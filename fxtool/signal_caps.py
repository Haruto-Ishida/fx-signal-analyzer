"""キルスイッチの「1日のシグナルの回数」の上限を、通貨ペアごとに過去のデータから決める（2026-10-07 のユーザーの決定）。

決め方：「100日に1日くらいしか起きない回数」＝ 1日のシグナルの回数がその値以上になる日が、全体の1%以下になる最小の回数。
- 開発用データ（最終テスト期間より前）だけを使う。
- 1日 ＝ 日本時間の日付。データがある日（取引のある日）だけを数える。
- 1時間足と日足の両方のシグナルを合計する。ただし2つの時間足の開発用データは期間が重ならない
  （日足の境界は何年も前）ので、それぞれの「1日の回数の分布」を別々に求めて、独立とみなして足し合わせる（たたみ込み）。
- 5つの戦略（最初からある戦略）のシグナルで数える（本番で記録しているのはこの5つ）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data, indicators, strategies
from .config import PAIRS, TIMEFRAMES, IndicatorParams

TAIL = 0.01  # 100日に1日


@dataclass
class CapResult:
    pair: str
    cap: int                       # 上限（この回数に達したら、その日はそれ以上出さない）
    tail_prob: float               # 1日の回数が cap 以上になる日の割合（見込み）
    mean: float                    # 1日の平均の回数
    n_days: dict[str, int]         # 時間足ごとの、数えた日数
    dist: np.ndarray               # 1日の回数の分布（index = 回数）


def daily_counts(dev: pd.DataFrame, tf_key: str, p: IndicatorParams = IndicatorParams()) -> pd.Series:
    """データがある日（日本時間）ごとの、5つの戦略のシグナルの合計回数。シグナルの時刻は足が確定した時刻。"""
    ind = indicators.add_all(dev, p)
    total = sum((strategies.signals(ind, s, p) != 0).astype(int) for s in strategies.STRATEGIES)
    day = (ind.index + TIMEFRAMES[tf_key].bar + pd.Timedelta(hours=9)).normalize()
    c = total.groupby(day).sum()
    return c.iloc[1:-1] if len(c) > 2 else c  # 最初と最後の日は途中までしかないので除く


def distribution(counts: pd.Series) -> np.ndarray:
    v = counts.to_numpy(dtype=int)
    return np.bincount(v) / len(v) if len(v) else np.array([1.0])


def cap_from(dist: np.ndarray, tail: float = TAIL) -> tuple[int, float]:
    """P(回数 ≥ cap) ≤ tail になる最小の cap と、そのときの確率。"""
    sf = dist[::-1].cumsum()[::-1]  # sf[k] = P(回数 ≥ k)
    for k in range(len(sf)):
        if sf[k] <= tail:
            return max(k, 1), float(sf[k])
    return len(sf), 0.0


def compute(con: sqlite3.Connection, pair: str, tail: float = TAIL) -> CapResult:
    dists, n_days = [], {}
    for tf in TIMEFRAMES:
        dev = data.load_dev(con, pair, tf)
        c = daily_counts(dev[["open", "high", "low", "close"]], tf)
        n_days[tf] = len(c)
        dists.append(distribution(c))
    dist = dists[0]
    for d in dists[1:]:
        dist = np.convolve(dist, d)
    cap, prob = cap_from(dist, tail)
    return CapResult(pair, cap, prob, float((np.arange(len(dist)) * dist).sum()), n_days, dist)


def compute_all(con: sqlite3.Connection, pairs=tuple(PAIRS), tail: float = TAIL) -> list[CapResult]:
    return [compute(con, p, tail) for p in pairs]


def toml_section(results: list[CapResult]) -> str:
    lines = ["[signal_caps]",
             "# キルスイッチの「1日のシグナルの回数」の上限（通貨ペアごと）。",
             "# 過去の開発用データで「100日に1日くらいしか起きない回数」として計算した値（fxtool/signal_caps.py）。",
             "# ここにない通貨ペアは [kill_switch] の max_signals_per_day を使う"]
    for r in results:
        lines.append(f'"{r.pair}" = {r.cap}   # 1日の平均 {r.mean:.1f}回・{r.cap}回以上の日は約{r.tail_prob:.1%}')
    return "\n".join(lines) + "\n"
