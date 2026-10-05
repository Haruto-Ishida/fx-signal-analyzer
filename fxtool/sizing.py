"""ポジションサイズの目安（研究ノート 6：1回の損失を口座残高の 1〜2% 以内にし、ATR に応じて調整する）。

損切りの幅は ATR の倍数なので、値動きが大きいときほど数量は小さくなる。注文は一切しない（表示のみ）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import MIN_UNITS


@dataclass
class Size:
    units: int                 # 取引数量の目安（通貨単位。MIN_UNITS 単位で切り捨て）
    risk_jpy: float            # 許容する損失額（円）
    loss_at_stop_jpy: float    # この数量で損切りになったときの損失額（円）
    note: str = ""


def quote_to_jpy(ticker: str, usdjpy: float | None) -> float | None:
    """決済通貨（ペアの後ろの通貨）1単位が何円か。"""
    quote = ticker.replace("=X", "")[3:6].upper()
    if quote == "JPY":
        return 1.0
    if quote == "USD":
        return usdjpy
    return None


def position_size(balance_jpy: float, risk_pct: float, entry: float, sl: float, ticker: str,
                  usdjpy: float | None) -> Size | None:
    """損切りまでの値幅 × 数量 × 円換算 ≤ 残高 × 許容損失% になる最大の数量。"""
    conv = quote_to_jpy(ticker, usdjpy)
    stop = abs(entry - sl)
    if conv is None or not stop or math.isnan(stop):
        return None
    risk_jpy = balance_jpy * risk_pct / 100
    raw_units = risk_jpy / (stop * conv)
    units = int(raw_units // MIN_UNITS * MIN_UNITS)
    if units == 0:
        return Size(0, risk_jpy, MIN_UNITS * stop * conv,
                    f"最小単位（{MIN_UNITS:,}通貨）でも許容損失を超えます")
    return Size(units, risk_jpy, units * stop * conv)
