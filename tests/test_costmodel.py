"""Step 2：時間帯によってスプレッドを広めに見積もる（要件定義書 5-2）のテスト。"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import costmodel, events
from fxtool.config import IndicatorParams, RiskParams

EMPTY = pd.DataFrame(columns=["bank", "start_date", "end_date"])


def mult_at(index, tf="1h", schedule=EMPTY):
    return pd.Series(costmodel.spread_multipliers(pd.DatetimeIndex(index), tf, schedule=schedule), index=index)


def test_nonfarm_payrolls_window():
    # 2026-10-02（第1金曜）の米雇用統計は 12:30 UTC。前後1時間（11:30〜13:30）に重なる足は3倍
    idx = pd.date_range("2026-10-02 09:00", "2026-10-02 15:00", freq="h")
    m = mult_at(idx)
    assert m["2026-10-02 10:00"] == 1 and m["2026-10-02 14:00"] == 1
    assert (m["2026-10-02 11:00":"2026-10-02 13:00"] == 3).all()


def test_rollover_follows_us_daylight_saving():
    summer = mult_at(pd.date_range("2026-10-06 19:00", periods=4, freq="h"))
    assert summer["2026-10-06 21:00"] == 2 and summer["2026-10-06 22:00"] == 1   # 17:00 米東部 = 21:00 UTC（夏）
    winter = mult_at(pd.date_range("2026-12-08 19:00", periods=5, freq="h"))
    assert winter["2026-12-08 22:00"] == 2 and winter["2026-12-08 21:00"] == 1   # 冬は 22:00 UTC


def test_weekend_open_hours():
    idx = pd.DatetimeIndex(["2026-10-09 18:00", "2026-10-09 19:00", "2026-10-11 23:00", "2026-10-12 00:00",
                            "2026-10-12 01:00", "2026-10-12 02:00"])
    m = mult_at(idx)
    assert m.tolist() == [1, 1, 3, 3, 3, 1]


def test_daily_bars_monday_and_fomc_from_schedule():
    idx = pd.date_range("2026-10-26", "2026-10-30", freq="B")  # 月〜金
    m = mult_at(idx, "1d", events.load_schedule())
    assert m["2026-10-26"] == 3                     # 月曜（週明け）
    assert m["2026-10-28"] == 3                     # FOMC（10/28 18:00 UTC）がある日
    assert m["2026-10-30"] == 3                     # 日銀会合（10/30 昼ごろ）がある日
    assert m["2026-10-27"] == 1 and m["2026-10-29"] == 1


def make_bars(rows, mult=None):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    df["atr"] = 1.0
    return bt.Bars(df, mult)


def test_trade_cost_uses_entry_and_exit_bars():
    flat = [100, 100.2, 99.8, 100]
    rows = [flat, [100.5, 100.6, 100.4, 100.5], [100.5, 102.6, 100.4, 102.5], flat]
    costs = bt.Costs(spread=1.0, slippage=0.2)
    plain = bt.simulate(make_bars(rows), [1, 0, 0, 0], RiskParams(1.0, 2.0), bt.ExitRules(), costs, 0.01)
    wide = bt.simulate(make_bars(rows, [1, 3, 2, 1]), [1, 0, 0, 0], RiskParams(1.0, 2.0), bt.ExitRules(), costs, 0.01)
    # 入る足（1本目：×3）と出る足（2本目：×2）のスプレッドの半分ずつ ＋ スリッページ×2
    assert plain.pips.iat[0] == pytest.approx(200 - 1.4)
    assert wide.pips.iat[0] == pytest.approx(200 - (0.5 * 3 + 0.5 * 2 + 0.4))
    with pytest.raises(ValueError):
        make_bars(rows, [1, 1])


def test_all_ones_is_the_same_as_constant_costs(random_walk):
    P, R, rules = IndicatorParams(), RiskParams(), bt.ExitRules(max_bars=48)
    a = bt.evaluate_all(random_walk, P, R, rules, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2)
    b = bt.evaluate_all(random_walk, P, R, rules, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2,
                        spread_mult=np.ones(len(random_walk)))
    for x, y in zip(a.reports, b.reports):
        assert x.full_metrics["total"] == pytest.approx(y.full_metrics["total"])
        assert x.wf.test_metrics["total"] == pytest.approx(y.wf.test_metrics["total"])


def test_wider_costs_never_improve_results(random_walk):
    P, R, rules = IndicatorParams(), RiskParams(), bt.ExitRules(max_bars=48)
    m = costmodel.spread_multipliers(random_walk.index, "1h", schedule=EMPTY)
    assert (m >= 1).all() and (m > 1).any()
    a = bt.evaluate_all(random_walk, P, R, rules, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2)
    b = bt.evaluate_all(random_walk, P, R, rules, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2, spread_mult=m)
    for x, y in zip(a.reports, b.reports):
        assert y.full_metrics["total"] <= x.full_metrics["total"] + 1e-9
