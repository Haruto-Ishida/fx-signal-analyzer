from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fxtool import commentary, events, indicators, strategies, tracking
from fxtool.config import IndicatorParams, RiskParams

P, RISK = IndicatorParams(), RiskParams()
NOW_TOKYO_NOON = pd.Timestamp("2026-10-06 03:00")  # 火曜 日本時間12時（時間帯の注意が出ない）
NO_ALERTS = pd.DataFrame(columns=["source", "message"])
NO_VERDICTS = pd.DataFrame(columns=["source", "verdict"])


@pytest.fixture
def setup(random_walk):
    ind = indicators.add_all(random_walk, P)
    n = len(ind)
    views = [strategies.evaluate(ind, n, s, P, RISK) for s in strategies.STRATEGIES]
    return ind, n, views


def wait_all(views):
    return [replace(v, decision="様子見") for v in views]


def force(views, key, decision):
    out = []
    for v in views:
        if v.strategy.key == key:
            d = 1 if decision == "買い" else -1
            v = replace(v, decision=decision, direction=d, bars_ago=1, entry=150.0, atr=0.2,
                        sl=150.0 - d * 0.3, tp=150.0 + d * 0.6)
        out.append(v)
    return out


def build(ind, n, views, weights=None, alerts=NO_ALERTS, verdicts=NO_VERDICTS, now=NOW_TOKYO_NOON, tf="1h", **kw):
    return commentary.build(ind, n, views, weights or {}, alerts, verdicts, tf, 0.01, 3, P, RISK, now,
                            lambda t: t.strftime("%m/%d %H:%M"), **kw)


def all_text(c):
    return " ".join([c.headline, *c.reasons, *c.cautions])


def test_no_signal_is_wait(setup):
    ind, n, views = setup
    c = build(ind, n, wait_all(views))
    assert c.headline.startswith("様子見")
    assert c.cautions[-1] == commentary.NEWS_CAVEAT
    assert len(c.reasons) == 1  # 相場の状況の説明だけ


def test_signal_reason_has_levels_and_evidence_strength(setup):
    ind, n, views = setup
    vs = force(wait_all(views), "macd_cross", "売り")
    verdicts = pd.DataFrame({"source": ["macd_cross"], "verdict": ["偶然の範囲"], "dsr": [0.02], "pbo": [0.94],
                             "n_trials": [86]})
    c = build(ind, n, vs, verdicts=verdicts)
    assert "MACDクロスが売り" in c.headline and "根拠としては弱い" in c.headline
    assert "86通り試した中での結果" in c.headline
    r = c.reasons[0]
    assert "MACDがシグナル線を下抜け（1本前）→ 売り" in r
    assert "150.000" in r and "150.300" in r and "149.400" in r
    assert "-30.0 pips" in r and "+60.0 pips" in r  # ATR 0.2 × 1.5 / 3.0
    assert "「偶然の範囲」" in r and "86通り試した中で" in r and "DSR）2%" in r and "PBO）94%" in r


def test_promising_strategy_is_not_called_weak(setup):
    ind, n, views = setup
    vs = force(wait_all(views), "breakout", "買い")
    c = build(ind, n, vs, verdicts=pd.DataFrame({"source": ["breakout"], "verdict": ["有望（要継続監視）"]}))
    assert "根拠としては弱い" not in c.headline


def test_conflict_alert_and_low_weight(setup):
    ind, n, views = setup
    vs = force(force(wait_all(views), "ma_cross", "買い"), "rsi_reversal", "売り")
    alerts = pd.DataFrame({"source": ["rsi_reversal"], "message": ["RSI反転：直近の成績が悪化"]})
    c = build(ind, n, vs, weights={"rsi_reversal": 0.5, "ma_cross": 1.2}, alerts=alerts)
    text = " ".join(c.cautions)
    assert "判断が分かれています" in text
    assert "警告中：RSI反転：直近の成績が悪化" in text
    assert "重みは 0.50" in text
    assert len(c.cautions) <= commentary.MAX_CAUTIONS


def test_counter_trend_warning(setup):
    ind, n, views = setup
    last = ind.iloc[-1]
    up = last["sma_s"] > last["sma_l"] and last["close"] > last["sma_l"]
    down = last["sma_s"] <= last["sma_l"] and last["close"] <= last["sma_l"]
    if not (up or down):
        pytest.skip("このデータの最後はトレンドがはっきりしない")
    vs = force(wait_all(views), "bb_reversion", "売り" if up else "買い")
    c = build(ind, n, vs)
    assert any("トレンドと逆向き" in x for x in c.cautions)


def test_meta_labeling_reference_or_used(setup):
    ind, n, views = setup
    vs = force(wait_all(views), "breakout", "買い")
    c = build(ind, n, vs, meta={"breakout": (0.35, "見送り")}, meta_ok=False)
    r = " ".join(c.reasons)
    assert "ブレイクアウトは「見送り」（コスト後に勝つ確率 35%）" in r and "参考です" in r
    assert "見送り" not in c.headline  # 条件を満たしていないモデルの判定は結論に使わない

    c2 = build(ind, n, vs, meta={"breakout": (0.35, "見送り")}, meta_ok=True)
    assert "「見送り」と判定" in c2.headline and "判断に含めています" in " ".join(c2.reasons)
    c3 = build(ind, n, vs, meta={"breakout": (0.62, "乗る")}, meta_ok=True)
    assert "根拠としては弱い" not in c3.headline

    c4 = build(ind, n, vs)  # 判定がまだない
    assert any("判定はまだありません" in x for x in c4.reasons)


def test_halt_is_the_first_thing_said(setup):
    ind, n, views = setup
    vs = force(wait_all(views), "ma_cross", "買い")
    halt = tracking.Halt(True, "今日（日本時間）の損失が -2.3R に達したため、今日は新しいシグナルを出しません", -2.3, -2.3)
    c = build(ind, n, vs, halt=halt)
    assert c.headline.startswith("⛔ 停止中") and c.cautions[0].startswith("停止中：今日")


def test_scheduled_events(setup):
    ind, n, views = setup
    sched = events.load_schedule()
    now = pd.Timestamp("2026-10-28 17:30")  # FOMC（日本時間 10/29 3:00）の30分前
    up = events.upcoming("USDJPY=X", now, 48, sched, include_low=True)
    c = build(ind, n, wait_all(views), now=now, upcoming=up)
    text = " ".join(c.cautions)
    assert "まもなくFOMC（米国の金融政策）の結果発表です（日本時間 10/29（木）03:00）" in text
    assert "日銀の金融政策決定会合の結果発表" in text and "ごろ 12:00" in text

    nfp_now = pd.Timestamp("2026-11-06 13:15")  # 米雇用統計（冬時間 13:30 UTC）の15分前
    c2 = build(ind, n, wait_all(views), now=nfp_now, upcoming=events.upcoming("USDJPY=X", nfp_now, 48, sched))
    assert any("まもなく米雇用統計" in x and "22:30" in x for x in c2.cautions)


def test_low_importance_times_and_weekend(setup):
    ind, n, views = setup
    hot = ind.copy()
    hot.iloc[-1, hot.columns.get_loc("atr")] = hot["atr"].iloc[-100:].mean() * 2
    now = pd.Timestamp("2026-10-06 12:15")  # 米国の経済指標が多い時刻（夏時間 12:30 UTC）の15分前
    up = events.upcoming("USDJPY=X", now, 48, events.load_schedule(), include_low=True)
    c = build(hot, n, wait_all(views), now=now, upcoming=up)
    text = " ".join(c.cautions)
    assert "値動きが普段より大きく" in text and "米国の経済指標が多い時刻" in text
    c2 = build(ind, n, wait_all(views), now=pd.Timestamp("2026-10-09 12:00"))  # 金曜 日本時間21時
    assert any("週末" in x for x in c2.cautions)
    c3 = build(ind, n, wait_all(views), now=pd.Timestamp("2026-10-09 12:00"), tf="1d")
    assert not any("週末" in x or "時間帯" in x for x in c3.cautions)


def test_reported_numbers_match_indicators(setup):
    ind, n, views = setup
    c = build(ind, n, wait_all(views))
    assert f"RSIは {ind['rsi'].iat[-1]:.0f}" in c.reasons[0]


def test_no_placeholder_text_for_any_real_state(setup):
    """実際の各戦略の状態をそのまま使っても、nan や None などが文章に混ざらない。"""
    ind, n, views = setup
    verdicts = pd.DataFrame({"source": [s.key for s in strategies.STRATEGIES], "verdict": "偶然の範囲",
                             "dsr": [0.1, np.nan, 0.3, 0.2, 0.05], "pbo": [np.nan, 0.5, 0.2, 0.1, 0.9],
                             "n_trials": 86})
    for i in range(200, n, 97):
        sub = ind.iloc[: i + 1]
        vs = [strategies.evaluate(sub, i + 1, s, P, RISK) for s in strategies.STRATEGIES]
        text = all_text(build(sub, i + 1, vs, weights={s.key: 1.0 for s in strategies.STRATEGIES}, verdicts=verdicts,
                              meta={s.key: (0.4, "乗る") for s in strategies.STRATEGIES}))
        for bad in ("nan", "None", "inf"):
            assert bad not in text
