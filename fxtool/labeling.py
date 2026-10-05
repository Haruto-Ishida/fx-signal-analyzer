"""ラベル（正解データ）の作り方と、時系列の検証の道具（研究ノート 2-2・3）。

- トリプルバリア：シグナルの次の足の始値で入り、利確・損切り・時間切れのどれに最初に届くかで結果を決める。
  バックテスト（backtest.simulate）と同じ判定（strategies.exit_hit）を使う。データの終わりまでに決着しない場合は、
  答えがまだないのでラベルを付けない。
- ラベルの期間（t0 = シグナルの足、t1 = 決済の足）を持たせ、パージ・エンバーゴ・重複度の重みに使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RiskParams
from .strategies import exit_hit, levels


def triple_barrier(ind: pd.DataFrame, events: pd.DataFrame, risk: RiskParams, max_bars: int, cost_pips: float,
                   pip: float) -> pd.DataFrame:
    """events：列 i（シグナルの足の行番号）と d（+1 買い / −1 売り）。

    戻り値：決着したものだけ。列 ev（events の何行目か）, i, d, t1（決済の足）, pips（コスト後）, gross（コスト前の pips）, reason。
    """
    o, h, l, c, atr = (ind[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    n = len(ind)
    rows = []
    for ev, (i, d) in enumerate(zip(events["i"].to_numpy(dtype=int), events["d"].to_numpy(dtype=int))):
        e = i + 1
        if e >= n or np.isnan(atr[i]):
            continue
        entry = o[e]
        sl, tp = levels(entry, d, atr[i], risk)
        for j in range(e, n):
            hit = exit_hit(d, o[j], h[j], l[j], sl, tp)
            if hit:
                reason, px = hit
                break
            if j - i >= max_bars:
                reason, px = "期限", c[j]
                break
        else:
            continue  # データの終わりまでに決着しない＝まだ答えがない
        gross = d * (px - entry) / pip
        rows.append((ev, i, d, j, gross - cost_pips, gross, reason))
    return pd.DataFrame(rows, columns=["ev", "i", "d", "t1", "pips", "gross", "reason"])


def uniqueness(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """平均の独自性（López de Prado）：ラベルの期間中、同時に有効なラベルの数 c の逆数 1/c の平均。
    重なりのないラベルは 1、まったく同じ期間のラベルが2つあれば それぞれ 0.5。"""
    t0, t1 = np.asarray(t0, dtype=int), np.asarray(t1, dtype=int)
    if len(t0) == 0:
        return np.array([])
    size = int(t1.max()) + 2
    diff = np.zeros(size)
    np.add.at(diff, t0, 1)
    np.add.at(diff, t1 + 1, -1)
    conc = np.cumsum(diff)[:-1]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    csum = np.r_[0.0, np.cumsum(inv)]
    return (csum[t1 + 1] - csum[t0]) / (t1 - t0 + 1)


def purged_train_mask(t0: np.ndarray, t1: np.ndarray, test_start: int, test_end: int, embargo: int) -> np.ndarray:
    """学習に使ってよいサンプル：ラベルの期間が検証期間 [test_start, test_end) と重ならず（パージ）、
    検証期間の直後 embargo 本以内に始まらない（エンバーゴ）もの。"""
    before = t1 < test_start
    after = t0 >= test_end + embargo
    return before | after
