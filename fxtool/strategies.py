"""戦略ルールとシグナル判定。

各戦略は「足の確定時に発生するイベント」として買い(+1)・売り(-1)・なし(0)を出す。
エントリーの目安はシグナルが出た足の終値、損切り・利確は ATR の倍数で決める。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import IndicatorParams, RiskParams


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift() <= b.shift())


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift() >= b.shift())


def _signal(buy: pd.Series, sell: pd.Series) -> pd.Series:
    return pd.Series(np.select([buy, sell], [1, -1], 0), index=buy.index, dtype=int)


def ma_cross(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    return _signal(_cross_up(d["sma_s"], d["sma_l"]), _cross_down(d["sma_s"], d["sma_l"]))


def macd_cross(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    return _signal(_cross_up(d["macd"], d["macd_sig"]), _cross_down(d["macd"], d["macd_sig"]))


def breakout(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    above = d["close"] > d["dc_hi"]
    below = d["close"] < d["dc_lo"]
    prev_above = above.shift(fill_value=False).astype(bool)
    prev_below = below.shift(fill_value=False).astype(bool)
    return _signal(above & ~prev_above, below & ~prev_below)


def rsi_reversal(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    r, prev = d["rsi"], d["rsi"].shift()
    return _signal((r >= p.rsi_low) & (prev < p.rsi_low), (r <= p.rsi_high) & (prev > p.rsi_high))


def bb_reversion(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    c, pc = d["close"], d["close"].shift()
    buy = (c > d["bb_lo"]) & (pc <= d["bb_lo"].shift())
    sell = (c < d["bb_up"]) & (pc >= d["bb_up"].shift())
    return _signal(buy, sell)


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    kind: str
    rule: str
    func: Callable[[pd.DataFrame, IndicatorParams], pd.Series]


STRATEGIES = [
    Strategy("ma_cross", "移動平均クロス", "順張り",
             "短期SMAが長期SMAを上抜けで買い（ゴールデンクロス）、下抜けで売り（デッドクロス）", ma_cross),
    Strategy("macd_cross", "MACDクロス", "順張り",
             "MACDがシグナル線を上抜けで買い、下抜けで売り", macd_cross),
    Strategy("breakout", "ブレイクアウト", "順張り",
             "終値が直前N本の最高値を超えたら買い、最安値を割ったら売り（ドンチャンチャネル）", breakout),
    Strategy("rsi_reversal", "RSI反転", "逆張り",
             "RSIが売られすぎラインを下から回復で買い、買われすぎラインを上から割り込みで売り", rsi_reversal),
    Strategy("bb_reversion", "ボリンジャー回帰", "逆張り",
             "終値が−σの外からバンド内に戻ったら買い、+σの外から戻ったら売り", bb_reversion),
]
STRATEGY_BY_KEY = {s.key: s for s in STRATEGIES}


def signals(ind: pd.DataFrame, strategy: Strategy, p: IndicatorParams, n_confirmed: int | None = None) -> pd.Series:
    """各足のシグナル。n_confirmed 以降（未確定の足）は 0 にする。ATR が計算できない足も 0。"""
    s = strategy.func(ind, p).where(ind["atr"].notna(), 0).astype(int)
    if n_confirmed is not None:
        s.iloc[n_confirmed:] = 0
    return s


def levels(entry: float, direction: int, atr: float, risk: RiskParams) -> tuple[float, float]:
    """(損切り, 利確) の価格。"""
    return entry - direction * risk.sl_atr * atr, entry + direction * risk.tp_atr * atr


@dataclass
class Outcome:
    status: str                     # "進行中" / "利確" / "損切り" / "期限"
    exit_time: pd.Timestamp | None
    exit_price: float | None
    bars_held: int


def exit_hit(direction: int, o: float, h: float, l: float, sl: float, tp: float) -> tuple[str, float] | None:
    """1本の足の中で損切り・利確に届いたかを調べ、(理由, 決済価格) を返す。

    - 損切りと利確の両方に届いた場合は、不利な側（損切り）が先とみなす。
    - 窓を開けて損切り・利確を飛び越えた場合は、その足の始値で決済したとみなす。
    """
    if direction > 0:
        if l <= sl:
            return "損切り", min(sl, o)
        if h >= tp:
            return "利確", max(tp, o)
    else:
        if h >= sl:
            return "損切り", max(sl, o)
        if l <= tp:
            return "利確", min(tp, o)
    return None


def resolve_trade(
    df: pd.DataFrame, i: int, direction: int, sl: float, tp: float, max_bars: int | None = None
) -> Outcome:
    """i 本目の終値で入ったポジションが、その後どこで決済されるかを調べる。

    max_bars を指定すると、その本数を経過した足の終値で決済する（"期限"）。
    """
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    for j in range(i + 1, len(df)):
        hit = exit_hit(direction, o[j], h[j], l[j], sl, tp)
        if hit:
            return Outcome(hit[0], df.index[j], hit[1], j - i)
        if max_bars is not None and j - i >= max_bars:
            return Outcome("期限", df.index[j], c[j], j - i)
    return Outcome("進行中", None, None, len(df) - 1 - i)


@dataclass
class StrategyView:
    """ある戦略の「今」の状態。"""
    strategy: Strategy
    decision: str                       # "買い" / "売り" / "様子見"
    direction: int                      # 直近シグナルの方向（なければ 0）
    signal_time: pd.Timestamp | None
    bars_ago: int | None                # 直近シグナルが最新の確定足から何本前か（0 = 最新の確定足）
    entry: float | None
    sl: float | None
    tp: float | None
    atr: float | None
    outcome: Outcome | None
    unrealized: float | None            # 進行中の場合の含み損益（価格差、方向込み）


def evaluate(
    ind: pd.DataFrame, n_confirmed: int, strategy: Strategy, p: IndicatorParams, risk: RiskParams
) -> StrategyView:
    """ind は未確定の足も含む指標付きデータ。シグナルは確定足だけで判定し、
    損切り・利確に届いたかは未確定の足の値動きも含めて調べる。"""
    sig = signals(ind, strategy, p, n_confirmed)
    hits = np.flatnonzero(sig.to_numpy() != 0)
    if len(hits) == 0:
        return StrategyView(strategy, "様子見", 0, None, None, None, None, None, None, None, None)

    i = int(hits[-1])
    direction = int(sig.iat[i])
    entry = float(ind["close"].iat[i])
    atr = float(ind["atr"].iat[i])
    sl, tp = levels(entry, direction, atr, risk)
    outcome = resolve_trade(ind, i, direction, sl, tp)
    bars_ago = n_confirmed - 1 - i
    unrealized = direction * (float(ind["close"].iat[-1]) - entry) if outcome.status == "進行中" else None

    active = outcome.status == "進行中" and bars_ago < risk.valid_bars
    decision = ("買い" if direction > 0 else "売り") if active else "様子見"
    return StrategyView(
        strategy, decision, direction, ind.index[i], bars_ago, entry, sl, tp, atr, outcome, unrealized
    )
