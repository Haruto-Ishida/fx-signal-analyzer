"""分数階差分（要件定義書 4章の2。López de Prado『Advances in Financial Machine Learning』5章の考え方）。

価格そのもの（d=0）は「記憶」が残るが、時期によって水準が変わる（定常でない）ので機械学習に向かない。
1階差分（d=1、ふつうの値動き）は定常だが、過去の水準の情報が消える。
その間の 0 < d < 1 で差分をとると、定常に近づけながら、過去の情報をある程度残せる。

- 窓の長さを固定する方法（FFD）：重みが threshold より小さくなったところで打ち切る。
  その足と、それより前の決まった本数だけを使うので、未来のデータは使わない（テストで確認）。
- d は固定（初期値 0.4）。本来は「定常になる最小の d」を学習データで探すが、そのための検定（ADF）のライブラリを
  入れておらず、また d を選ぶこと自体も試行になるため、まずは固定値で始める。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_D = 0.4
DEFAULT_THRESHOLD = 1e-3
MAX_WIDTH = 500


def weights(d: float, threshold: float = DEFAULT_THRESHOLD, max_width: int = MAX_WIDTH) -> np.ndarray:
    """w[0]=1, w[k] = −w[k−1]·(d−k+1)/k。|w[k]| < threshold で打ち切る。w[k] は k 本前の値にかける重み。"""
    w = [1.0]
    k = 1
    while k < max_width:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < threshold:
            break
        w.append(nxt)
        k += 1
    return np.array(w)


def frac_diff(x: pd.Series, d: float = DEFAULT_D, threshold: float = DEFAULT_THRESHOLD,
              max_width: int = MAX_WIDTH) -> pd.Series:
    """FFD の分数階差分。最初の (窓の長さ − 1) 本と、窓の中に欠損がある足は NaN。"""
    w = weights(d, threshold, max_width)
    width = len(w)
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= width:
        # out[i] = Σ_k w[k]·v[i−k]（k = 0..width−1）
        win = np.lib.stride_tricks.sliding_window_view(v, width)  # win[j] = v[j .. j+width−1]
        out[width - 1:] = win @ w[::-1]
    return pd.Series(out, index=x.index, name=x.name)


def fd_feature(close: pd.Series, d: float = DEFAULT_D) -> pd.Series:
    """機械学習の特徴量用：対数価格の分数階差分（×100）。"""
    return frac_diff(np.log(close), d) * 100
