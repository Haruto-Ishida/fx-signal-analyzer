"""テクニカル指標の計算。入力は open/high/low/close 列を持つ DataFrame。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import IndicatorParams


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """ワイルダーの平滑化（RSI・ATR で使う）。最初の n 本は単純平均で初期化する。"""
    out = pd.Series(np.nan, index=s.index)
    values = s.to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < n:
        return out
    start = valid[0] + n - 1
    prev = values[valid[0]:start + 1].mean()
    res = np.full(len(values), np.nan)
    res[start] = prev
    for i in range(start + 1, len(values)):
        prev = prev + (values[i] - prev) / n
        res[i] = prev
    out[:] = res
    return out


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder((-delta).clip(lower=0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + gain / loss)
    r = r.where(loss != 0, 100.0)
    return r.where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(df), n)


def add_all(df: pd.DataFrame, p: IndicatorParams) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["sma_s"] = sma(close, p.sma_short)
    out["sma_l"] = sma(close, p.sma_long)

    out["rsi"] = rsi(close, p.rsi_period)

    macd = ema(close, p.macd_fast) - ema(close, p.macd_slow)
    sig = ema(macd, p.macd_signal)
    warmup = p.macd_slow + p.macd_signal  # EMA の立ち上がりは不安定なので使わない
    out["macd"] = macd
    out["macd_sig"] = sig
    out["macd_hist"] = macd - sig
    out.iloc[:warmup, out.columns.get_indexer(["macd", "macd_sig", "macd_hist"])] = np.nan

    mid = sma(close, p.bb_period)
    sd = close.rolling(p.bb_period).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_up"] = mid + p.bb_std * sd
    out["bb_lo"] = mid - p.bb_std * sd

    out["atr"] = atr(out, p.atr_period)

    # ドンチャンチャネル：その足を含まない、直前 n 本の高値・安値
    out["dc_hi"] = out["high"].rolling(p.donchian_period).max().shift()
    out["dc_lo"] = out["low"].rolling(p.donchian_period).min().shift()
    return out
