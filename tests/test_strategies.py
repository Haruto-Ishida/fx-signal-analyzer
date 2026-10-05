import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlc
from fxtool import indicators, strategies
from fxtool.config import IndicatorParams, RiskParams

P = IndicatorParams()


@pytest.mark.parametrize("strategy", strategies.STRATEGIES, ids=lambda s: s.key)
def test_signals_do_not_look_ahead(random_walk, strategy):
    full = strategies.signals(indicators.add_all(random_walk, P), strategy, P)
    part = strategies.signals(indicators.add_all(random_walk.iloc[:900], P), strategy, P)
    pd.testing.assert_series_equal(full.iloc[:900], part)


@pytest.mark.parametrize("strategy", strategies.STRATEGIES, ids=lambda s: s.key)
def test_every_strategy_fires_both_ways(random_walk, strategy):
    s = strategies.signals(indicators.add_all(random_walk, P), strategy, P)
    assert set(s.unique()) <= {-1, 0, 1}
    assert (s == 1).sum() > 0 and (s == -1).sum() > 0


def test_ma_cross_fires_once_at_golden_cross():
    close = np.r_[np.linspace(160, 140, 150), np.linspace(140, 170, 150)]
    d = indicators.add_all(make_ohlc(close), P)
    s = strategies.signals(d, strategies.STRATEGY_BY_KEY["ma_cross"], P)
    buys = np.flatnonzero(s.to_numpy() == 1)
    assert len(buys) == 1
    i = buys[0]
    assert d["sma_s"].iat[i] > d["sma_l"].iat[i] and d["sma_s"].iat[i - 1] <= d["sma_l"].iat[i - 1]
    assert (s == -1).sum() == 0


def test_signals_ignore_unconfirmed_bars(random_walk):
    d = indicators.add_all(random_walk, P)
    st = strategies.STRATEGY_BY_KEY["macd_cross"]
    raw = strategies.signals(d, st, P)
    last = int(np.flatnonzero(raw.to_numpy() != 0)[-1])
    masked = strategies.signals(d, st, P, n_confirmed=last)
    assert masked.iloc[last:].eq(0).all()


def bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    return df


def test_resolve_take_profit():
    df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [100.5, 102.5, 100.4, 102]])
    o = strategies.resolve_trade(df, 0, 1, sl=99, tp=102)
    assert (o.status, o.exit_price, o.bars_held) == ("利確", 102, 2)


def test_resolve_both_hit_counts_as_stop():
    df = bars([[100, 100, 100, 100], [100, 103, 98, 101]])
    assert strategies.resolve_trade(df, 0, 1, sl=99, tp=102).status == "損切り"


def test_resolve_gap_through_stop_exits_at_open():
    df = bars([[100, 100, 100, 100], [97, 97.5, 96, 97]])
    o = strategies.resolve_trade(df, 0, 1, sl=99, tp=102)
    assert (o.status, o.exit_price) == ("損切り", 97)


def test_resolve_short_side_and_open_trade():
    df = bars([[100, 100, 100, 100], [100, 100.5, 99.5, 99.8]])
    assert strategies.resolve_trade(df, 0, -1, sl=101, tp=98).status == "進行中"
    df2 = bars([[100, 100, 100, 100], [100, 100.2, 97.9, 98]])
    o = strategies.resolve_trade(df2, 0, -1, sl=101, tp=98)
    assert (o.status, o.exit_price) == ("利確", 98)


def test_resolve_max_bars():
    df = bars([[100, 100, 100, 100]] + [[100, 100.5, 99.5, 100.1]] * 5)
    o = strategies.resolve_trade(df, 0, 1, sl=99, tp=102, max_bars=3)
    assert (o.status, o.bars_held, o.exit_price) == ("期限", 3, 100.1)


def test_levels():
    assert strategies.levels(100, 1, 2, RiskParams(sl_atr=1.5, tp_atr=3)) == (97, 106)
    assert strategies.levels(100, -1, 2, RiskParams(sl_atr=1.5, tp_atr=3)) == (103, 94)


def test_evaluate_decision_window(random_walk):
    d = indicators.add_all(random_walk, P)
    st = strategies.STRATEGY_BY_KEY["macd_cross"]
    sig = strategies.signals(d, st, P)
    i = int(np.flatnonzero(sig.to_numpy() != 0)[-2])  # 少し前のシグナルを「最新の確定足」にする
    risk = RiskParams(sl_atr=50, tp_atr=50, valid_bars=3)  # 損切り・利確に届かない幅
    v = strategies.evaluate(d.iloc[: i + 1], i + 1, st, P, risk)
    assert v.bars_ago == 0 and v.decision == ("買い" if sig.iat[i] > 0 else "売り")
    assert v.sl < v.entry < v.tp if sig.iat[i] > 0 else v.tp < v.entry < v.sl

    v_old = strategies.evaluate(d.iloc[: i + 4], i + 4, st, P, risk)
    if v_old.signal_time == d.index[i]:  # 間に別のシグナルが出ていなければ
        assert v_old.bars_ago == 3 and v_old.decision == "様子見"
