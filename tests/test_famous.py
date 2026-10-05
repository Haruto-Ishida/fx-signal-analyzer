"""Step 4：有名手法ライブラリのテスト（合成データだけ。実データでの検証はユーザーの承認後）。"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import famous, indicators, strategies, trials
from fxtool.config import IndicatorParams, RiskParams

P = IndicatorParams()


@pytest.fixture(scope="module")
def daily():
    df = make_path_ohlc(2500, sd=0.6, seed=12)
    df.index = pd.bdate_range("2010-01-01", periods=len(df))
    return df


def ind_for(df, st):
    return indicators.add_all(df, st.params(P))


@pytest.mark.parametrize("st", famous.FAMOUS, ids=lambda s: s.key)
def test_no_lookahead(daily, st):
    """後ろのデータを書き換えても、それより前のシグナル・決済シグナルは変わらない。"""
    full = ind_for(daily, st)
    tampered = daily.copy()
    tampered.iloc[1800:] *= 1.25
    part = ind_for(tampered, st)
    p = st.params(P)
    pd.testing.assert_series_equal(strategies.signals(full, st, p).iloc[:1800], strategies.signals(part, st, p).iloc[:1800])
    if st.exit_func is not None:
        assert strategies.exit_signals(full, st, p)[:1800] == strategies.exit_signals(part, st, p)[:1800]


@pytest.mark.parametrize("st", famous.FAMOUS, ids=lambda s: s.key)
def test_each_rule_trades_both_ways_and_is_documented(daily, st):
    s = strategies.signals(ind_for(daily, st), st, st.params(P))
    assert (s == 1).sum() > 0 and (s == -1).sum() > 0
    assert st.family == "famous" and st.source and st.spec.get("simplifications")
    assert "rule" in st.spec


def test_parabolic_sar_follows_trend():
    up = np.linspace(100, 120, 60)
    h, l = up + 0.2, up - 0.2
    sar, trend = famous.parabolic_sar(h, l)
    assert (trend[1:] == 1).all() and (sar[2:] < l[2:]).all()
    down = np.r_[up, np.linspace(120, 100, 40)]
    sar2, trend2 = famous.parabolic_sar(down + 0.2, down - 0.2)
    assert trend2[-1] == -1 and (np.diff(trend2) != 0).sum() == 1  # 1回だけ下向きに変わる


def test_turtle_s1_skips_some_breakouts(daily):
    st = famous.FAMOUS_BY_KEY["turtle_s1"]
    ind = ind_for(daily, st)
    s1 = strategies.signals(ind, st, st.params(P))
    buy, sell = famous._turtle_entries(ind, 20)
    raw = np.where(buy, 1, np.where(sell, -1, 0))
    taken = s1.to_numpy() != 0
    assert (s1.to_numpy()[taken] == raw[taken]).all()  # 出すシグナルは、20日ブレイクアウトそのもの
    assert 0 < taken.sum() < (raw != 0).sum()          # 勝ちのあと・仮の取引中のブレイクアウトは見送られている


def test_turtle_s1_skip_rule_on_a_handmade_path():
    """A：上へのブレイク（入る）→ 10日安値割れで勝って終わる → B：次の上へのブレイクは「直前が勝ち」なので見送る
    → B の仮の取引は急落で損切り → その急落での下へのブレイクは入る → C：その後の上へのブレイクも入る。"""
    c = np.r_[[100.0] * 30, [101.0],                      # 30本目：A
              np.linspace(101.5, 110, 15), [108, 106, 104, 103, 103],
              [103.0] * 15, np.linspace(104, 109, 5),     # 70本目：B（終値109が直前20本の高値を超える）
              [108, 100, 100, 100, 100], [100.0] * 25, [101.0], [101.0] * 5]
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.1, "low": np.minimum(o, c) - 0.1, "close": c},
                      index=pd.bdate_range("2020-01-01", periods=len(c)))
    st = famous.FAMOUS_BY_KEY["turtle_s1"]
    ind = indicators.add_all(df, st.params(P))
    s = famous.turtle_s1(ind, st.params(P)).to_numpy()
    buy, sell = famous._turtle_entries(ind, 20)
    a, b, crash, c_bar = 30, 68, 72, 101  # B は終値106.5が直前20本の高値（104.1）を超える68本目
    assert buy.iat[a] and s[a] == 1                      # A は入る
    assert buy.iat[b] and s[b] == 0                      # B は直前（A）が勝ちなので見送る
    assert sell.iat[crash] and s[crash] == -1            # B の仮の取引は損切り → 次の下へのブレイクは入る
    assert buy.iat[c_bar] and s[c_bar] == 1              # 直前（下へのブレイク）は負けなので C は入る


def make_bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="D")
    df["atr"] = 1.0
    return bt.Bars(df)


def test_exit_signal_closes_at_next_open():
    flat = [100, 100.2, 99.8, 100]
    bars = make_bars([flat, [100.5, 100.6, 100.4, 100.5], [100.5, 100.6, 100.4, 100.6], [100.7, 100.8, 100.6, 100.7], flat])
    rs, ru = bt.strategy_exits(famous.FAMOUS_BY_KEY["turtle_s2"], RiskParams(), bt.ExitRules())
    assert ru.max_bars == 0 and rs.tp_atr == bt.NO_BARRIER and rs.sl_atr == 2.0
    t = bt.simulate(bars, [1, 0, 0, 0, 0], rs, ru, bt.Costs(0, 0), 0.01, exit_sig=[0, 0, 1, 0, 0])
    r = t.iloc[0]
    assert (r.reason, r.exit_i, r.exit) == ("決済シグナル", 3, 100.7)   # 2本目の終値で決済シグナル → 3本目の始値
    t2 = bt.simulate(bars, [1, 0, 0, 0, 0], rs, ru, bt.Costs(0, 0), 0.01, exit_sig=[0, 0, -1, 0, 0])
    assert t2.iloc[0].reason == "期間末"  # 売りの決済シグナルでは、買いは決済しない


def test_famous_tab_evaluates_only_dev_data_after_approval(daily, tmp_path, monkeypatch):
    """承認ボタンの処理：日足の開発用データだけで6通りを検証し、判定と試行を記録する（合成データ・一時DB）。"""
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from dashboard import famous_tab
    from fxtool import data, holdout, tracking

    path = tmp_path / "f.db"
    real = data.connect
    monkeypatch.setattr(data, "connect", lambda *a, **k: real(path))
    monkeypatch.setattr(data, "now_utc", lambda: daily.index[-1] + pd.Timedelta(days=2))
    con = real(path)
    data.save(con, "EUR/JPY", "1d", daily)
    n0 = trials.count(con, "EUR/JPY", "1d")
    seen = {}
    orig = bt.evaluate_all

    def spy(raw, *a, **k):
        seen["last"] = raw.index[-1]
        return orig(raw, *a, **{**k, "runs": 10, "rw_paths": 2})

    monkeypatch.setattr(bt, "evaluate_all", spy)
    famous_tab.evaluate_and_save("EUR/JPY")
    b = holdout.get("EUR/JPY", "1d")
    assert seen["last"] < b.start                               # 最終テスト期間は使っていない
    v = tracking.load_verdicts(con, "EUR/JPY", "1d")
    assert set(v["source"]) == {s.key for s in famous.FAMOUS}
    assert trials.count(con, "EUR/JPY", "1d") == n0 + len(famous.FAMOUS)
    assert (v["n_trials"] == n0 + len(famous.FAMOUS)).all()     # 増える分も含めて割り引いている


def test_famous_strategies_can_be_evaluated_and_counted_once(daily):
    rules = bt.ExitRules(max_bars=20)
    rep = bt.evaluate_all(daily, P, RiskParams(), rules, bt.Costs(0.3, 0.2), 0.01, runs=10, k=3, rw_paths=2,
                          strategy_list=famous.FAMOUS)
    assert [r.strategy.key for r in rep.reports] == [s.key for s in famous.FAMOUS]
    for r in rep.reports:
        assert r.verdict[0]
        # 元のルールでは時間切れを使わない
        assert "期限" not in set(r.full_trades["reason"])
    hashes = {trials.config_hash("strategy", row["config"]) for row in rep.trials}
    planned = {trials.config_hash("strategy", row["config"]) for row in bt.planned_trials(P, RiskParams(), rules, famous.FAMOUS)}
    assert hashes == planned and len(planned) == len(famous.FAMOUS)  # 元のルールの版だけ ＝ 6通り
    assert all(row["config"]["spec"]["rule"] for row in rep.trials)
