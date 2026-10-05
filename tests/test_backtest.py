import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import indicators, strategies
from fxtool.config import IndicatorParams, RiskParams

PIP = 0.01
NO_COST = bt.Costs(0, 0)
RISK = RiskParams(sl_atr=1.0, tp_atr=2.0)


def make_bars(rows, atr=1.0) -> bt.Bars:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    df["atr"] = atr
    return bt.Bars(df)


FLAT = [100, 100.2, 99.8, 100]


def test_entry_is_next_open_and_take_profit():
    bars = make_bars([FLAT, [100.5, 100.6, 100.4, 100.5], [100.5, 102.6, 100.4, 102.5], FLAT])
    t = bt.simulate(bars, [1, 0, 0, 0], RISK, bt.ExitRules(), NO_COST, PIP)
    assert len(t) == 1
    r = t.iloc[0]
    assert (r.entry_i, r.entry, r.sl, r.tp) == (1, 100.5, 99.5, 102.5)
    assert (r.reason, r.exit_i, r.exit) == ("利確", 2, 102.5)
    assert r.pips == pytest.approx(200)


def test_costs_are_subtracted():
    bars = make_bars([FLAT, [100.5, 100.6, 100.4, 100.5], [100.5, 102.6, 100.4, 102.5], FLAT])
    t = bt.simulate(bars, [1, 0, 0, 0], RISK, bt.ExitRules(), bt.Costs(spread=0.3, slippage=0.2), PIP)
    assert t.iloc[0].pips == pytest.approx(200 - 0.7)


def test_short_stop_loss():
    bars = make_bars([FLAT, FLAT, [100, 101.2, 99.9, 101]])
    t = bt.simulate(bars, [-1, 0, 0], RISK, bt.ExitRules(), NO_COST, PIP)
    r = t.iloc[0]
    assert (r.reason, r.exit, r.pips) == ("損切り", 101, pytest.approx(-100))


def test_opposite_signal_reverses_position():
    bars = make_bars([FLAT, FLAT, FLAT, [100.3, 100.4, 100.2, 100.3], FLAT, FLAT])
    sig = [1, 0, -1, 0, 0, 0]
    t = bt.simulate(bars, sig, RISK, bt.ExitRules(opposite_exit=True), NO_COST, PIP)
    assert list(t.reason) == ["反対シグナル", "期間末"]
    first, second = t.iloc[0], t.iloc[1]
    assert (first.exit_i, first.exit) == (3, 100.3)       # 反対シグナルの次の足の始値で決済
    assert (second.dir, second.entry_i, second.entry) == (-1, 3, 100.3)  # 同じ価格でドテン


def test_same_direction_signal_ignored_while_holding():
    bars = make_bars([FLAT] * 6)
    t = bt.simulate(bars, [1, 0, 1, 0, 0, 0], RISK, bt.ExitRules(opposite_exit=False), NO_COST, PIP)
    assert len(t) == 1 and t.iloc[0].reason == "期間末"


def test_max_bars_and_window_end():
    bars = make_bars([FLAT] * 10)
    t = bt.simulate(bars, [1] + [0] * 9, RISK, bt.ExitRules(max_bars=3), NO_COST, PIP)
    assert (t.iloc[0].reason, t.iloc[0].exit_i) == ("期限", 3)
    t2 = bt.simulate(bars, [0, 0, 1] + [0] * 7, RISK, bt.ExitRules(), NO_COST, PIP, start=0, end=5)
    assert (t2.iloc[0].reason, t2.iloc[0].exit_i) == ("期間末", 4)  # 窓の外の足は使わない


def test_signal_on_last_bar_of_window_is_not_traded():
    bars = make_bars([FLAT] * 5)
    assert bt.simulate(bars, [0, 0, 0, 0, 1], RISK, bt.ExitRules(), NO_COST, PIP).empty
    assert bt.simulate(bars, [0, 0, 1, 0, 0], RISK, bt.ExitRules(), NO_COST, PIP, 0, 3).empty


def test_metrics():
    m = bt.metrics([10, -5, 20, -15, -10, 5])
    assert m["n"] == 6 and m["win_rate"] == pytest.approx(0.5)
    assert m["total"] == pytest.approx(5) and m["pf"] == pytest.approx(35 / 30)
    assert m["max_dd"] == pytest.approx(-25)  # 累積 10,5,25,10,0,5 → ピーク25から0
    assert bt.metrics([-3, -2])["max_dd"] == pytest.approx(-5)  # 開始時点の0もピーク
    assert math.isinf(bt.metrics([1, 2])["pf"])
    assert bt.metrics([])["n"] == 0


def test_equity_curve():
    assert bt.equity_curve([1, 3], [10, -4], 5).tolist() == [0, 10, 10, 6, 6]


def test_random_baseline_is_reproducible_and_matches_rate(random_walk):
    ind = indicators.add_all(random_walk, IndicatorParams())
    bars = bt.Bars(ind)
    w = [(bt.WARMUP, len(ind), 0.02)]
    r1, q = bt.random_baseline(bars, w, RISK, bt.ExitRules(), bt.Costs(), PIP, runs=30, seed=1, curves=True)
    r2, _ = bt.random_baseline(bars, w, RISK, bt.ExitRules(), bt.Costs(), PIP, runs=30, seed=1)
    pd.testing.assert_frame_equal(r1, r2)
    assert q.shape == (3, len(ind)) and (q[0] <= q[1]).all() and (q[1] <= q[2]).all()


def test_random_trading_has_zero_edge_minus_costs():
    """ランダム売買の1回あたりの期待値は「0 − 取引コスト」になる（エンジンに偏りがないことの確認）。

    損切り・利確を同じ幅にして、足の途中の値動きの細かさによる影響を受けないようにしている。
    """
    df = make_path_ohlc(6000, seed=3)
    bars = bt.Bars(indicators.add_all(df, IndicatorParams()))
    w = [(bt.WARMUP, len(df), 0.03)]
    risk = RiskParams(sl_atr=1.0, tp_atr=1.0)
    rules = bt.ExitRules(opposite_exit=False)
    free, _ = bt.random_baseline(bars, w, risk, rules, bt.Costs(0, 0), PIP, runs=200, seed=0)
    paid, _ = bt.random_baseline(bars, w, risk, rules, bt.Costs(1.0, 0.5), PIP, runs=200, seed=0)
    assert free["n"].sum() > 20000
    assert abs(free["total"].sum() / free["n"].sum()) < 0.5
    assert paid["total"].sum() / paid["n"].sum() == pytest.approx(free["total"].sum() / free["n"].sum() - 2.0)


def test_beat_ratio():
    assert bt.beat_ratio(5, pd.Series([1, 2, 3, 10])) == 0.75


def test_make_folds_cover_data_without_overlap():
    folds = bt.make_folds(1150, k=5, train_mult=2)
    assert folds[0][0][0] == bt.WARMUP
    for (a, b), (b2, c) in folds:
        assert a < b == b2 < c
        assert (b - a) == 2 * (c - b2) or c == 1150
    tests = [f[1] for f in folds]
    assert all(tests[i][1] == tests[i + 1][0] for i in range(len(tests) - 1))  # 検証区間は連続
    assert tests[-1][1] == 1150


def test_walk_forward_uses_only_past_for_selection(random_walk):
    """学習区間より後ろのデータを書き換えても、各区間で選ばれるパラメータは変わらない。"""
    st = strategies.STRATEGY_BY_KEY["ma_cross"]
    args = (IndicatorParams(), RISK, bt.ExitRules(), bt.Costs(), PIP)
    wf = bt.walk_forward(random_walk, st, *args, k=3, train_mult=2)
    assert len(wf.folds) == 3
    assert wf.test_metrics["n"] == len(wf.test_trades)

    first_train_end = wf.folds[0].train[1]
    tampered = random_walk.copy()
    tampered.iloc[first_train_end:, :] = tampered.iloc[first_train_end:, :] * 1.5  # 未来を大きく変える
    wf2 = bt.walk_forward(tampered, st, *args, k=3, train_mult=2)
    assert wf2.folds[0].overrides == wf.folds[0].overrides
    assert wf2.folds[0].train_metrics == wf.folds[0].train_metrics


def test_assess_labels():
    def fake(total, n, pos):
        return bt.WalkForwardResult([], pd.DataFrame(), dict(n=n, total=total), 0.0, pos)
    assert bt.assess(fake(100, 5, 5), 0.99, 5)[0] == "判断材料不足"
    assert bt.assess(fake(-1, 50, 2), 0.99, 5)[0] == "優位性なし"
    assert bt.assess(fake(100, 150, 4), 0.97, 5, dsr=0.99, pbo=0.1)[0] == "有望（要継続監視）"
    assert bt.assess(fake(100, 50, 4), 0.97, 5, dsr=0.99, pbo=0.1)[0] == "参考値"  # 取引100回未満
    assert bt.assess(fake(100, 50, 1), 0.97, 5)[0] == "不安定"
    assert bt.assess(fake(100, 50, 4), 0.60, 5)[0] == "偶然の範囲"
