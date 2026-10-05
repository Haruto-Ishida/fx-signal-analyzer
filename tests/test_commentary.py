from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fxtool import commentary, indicators, strategies
from fxtool.config import IndicatorParams, RiskParams
from fxtool.tracking import MLStatus

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


def build(ind, n, views, weights=None, ml=(), alerts=NO_ALERTS, verdicts=NO_VERDICTS, now=NOW_TOKYO_NOON, tf="1h"):
    return commentary.build(ind, n, views, weights or {}, list(ml), alerts, verdicts, tf, 0.01, 3, P, RISK, now,
                            lambda t: t.strftime("%m/%d %H:%M"))


def all_text(c):
    return " ".join([c.headline, *c.reasons, *c.cautions])


def test_no_signal_is_wait(setup):
    ind, n, views = setup
    c = build(ind, n, wait_all(views))
    assert c.headline.startswith("様子見")
    assert c.cautions[-1] == commentary.NEWS_CAVEAT
    assert len(c.reasons) == 1  # 相場の状況の説明だけ


def test_signal_reason_has_levels_and_weak_evidence(setup):
    ind, n, views = setup
    vs = force(wait_all(views), "macd_cross", "売り")
    c = build(ind, n, vs, verdicts=pd.DataFrame({"source": ["macd_cross"], "verdict": ["偶然の範囲"]}))
    assert "MACDクロスが売り" in c.headline and "根拠としては弱い" in c.headline
    r = c.reasons[0]
    assert "MACDがシグナル線を下抜け（1本前）→ 売り" in r
    assert "150.000" in r and "150.300" in r and "149.400" in r
    assert "-30.0 pips" in r and "+60.0 pips" in r  # ATR 0.2 × 1.5 / 3.0
    assert any("「偶然の範囲」" in x for x in c.cautions)


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


def test_ml_is_mentioned_but_not_used_without_edge(setup):
    ind, n, views = setup
    t = ind.index[-1]
    ml = [MLStatus("gbm", "勾配ブースティング", 1, 0.62, "シグナルなし", False, t)]
    c = build(ind, n, wait_all(views), ml=ml)
    assert c.headline.startswith("様子見")
    assert any("62%" in r and "判断材料にはしていません" in r for r in c.reasons)

    ml_ok = [MLStatus("gbm", "勾配ブースティング", 1, 0.62, "買い", True, t)]
    c2 = build(ind, n, wait_all(views), ml=ml_ok)
    assert "機械学習（勾配ブースティング）が買い" in c2.headline
    assert "根拠としては弱い" not in c2.headline


def test_volatility_and_time_of_day(setup):
    ind, n, views = setup
    hot = ind.copy()
    hot.iloc[-1, hot.columns.get_loc("atr")] = hot["atr"].iloc[-100:].mean() * 2
    c = build(hot, n, wait_all(views), now=pd.Timestamp("2026-10-06 13:00"))  # 日本時間22時
    text = " ".join(c.cautions)
    assert "値動きが普段より大きく" in text and "米国の経済指標" in text
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
    for i in range(200, n, 97):
        sub = ind.iloc[: i + 1]
        vs = [strategies.evaluate(sub, i + 1, s, P, RISK) for s in strategies.STRATEGIES]
        text = all_text(build(sub, i + 1, vs, weights={s.key: 1.0 for s in strategies.STRATEGIES}))
        for bad in ("nan", "None", "inf"):
            assert bad not in text
