import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlc
from fxtool import indicators as ind
from fxtool.config import IndicatorParams


def naive_wilder_rsi(close: list[float], n: int) -> list[float]:
    """教科書どおりの RSI（最初の n 本は単純平均、その後はワイルダー平滑化）。"""
    deltas = np.diff(close)
    out = [np.nan] * len(close)
    avg_g = np.mean([max(d, 0) for d in deltas[:n]])
    avg_l = np.mean([max(-d, 0) for d in deltas[:n]])
    out[n] = 100 - 100 / (1 + avg_g / avg_l)
    for i in range(n, len(deltas)):
        d = deltas[i]
        avg_g = (avg_g * (n - 1) + max(d, 0)) / n
        avg_l = (avg_l * (n - 1) + max(-d, 0)) / n
        out[i + 1] = 100 - 100 / (1 + avg_g / avg_l)
    return out


def test_sma():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]


def test_rsi_matches_textbook(random_walk):
    close = random_walk["close"]
    got = ind.rsi(close, 14).to_numpy()
    want = np.array(naive_wilder_rsi(close.tolist(), 14))
    np.testing.assert_allclose(got, want, rtol=1e-9, equal_nan=True)


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert ind.rsi(up, 14).dropna().eq(100).all()
    assert ind.rsi(down, 14).dropna().eq(0).all()


def test_atr_constant_range():
    df = make_ohlc([100.0] * 50, spread=0.5)  # 値幅はいつも 1.0
    assert ind.atr(df, 14).dropna().round(12).eq(1.0).all()


def test_true_range_uses_gap():
    df = pd.DataFrame({"open": [100, 105], "high": [101, 106], "low": [99, 104], "close": [100, 105]}, dtype=float)
    # 2本目：高値-安値=2 だが、前日終値100からの窓で 106-100=6
    assert ind.true_range(df).tolist() == [2.0, 6.0]


def test_add_all_columns_and_bands(random_walk):
    p = IndicatorParams()
    d = ind.add_all(random_walk, p)
    for col in ["sma_s", "sma_l", "rsi", "macd", "macd_sig", "macd_hist", "bb_up", "bb_lo", "atr", "dc_hi", "dc_lo"]:
        assert col in d and d[col].notna().sum() > 1000
    v = d.dropna()
    assert (v["bb_up"] >= v["bb_mid"]).all() and (v["bb_mid"] >= v["bb_lo"]).all()
    assert d["macd"].iloc[: p.macd_slow + p.macd_signal].isna().all()


def test_donchian_excludes_current_bar(random_walk):
    p = IndicatorParams(donchian_period=5)
    d = ind.add_all(random_walk, p)
    i = 100
    assert d["dc_hi"].iat[i] == pytest.approx(random_walk["high"].iloc[i - 5:i].max())
    assert d["dc_lo"].iat[i] == pytest.approx(random_walk["low"].iloc[i - 5:i].min())


def test_indicators_do_not_look_ahead(random_walk):
    """後ろのデータを足しても、過去の足の指標値は変わらない（未来の情報を使っていない）。"""
    p = IndicatorParams()
    full = ind.add_all(random_walk, p)
    part = ind.add_all(random_walk.iloc[:800], p)
    pd.testing.assert_frame_equal(full.iloc[:800], part)
