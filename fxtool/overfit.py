"""成績の指標と、バックテストの過学習のチェック（研究ノート 2-3・5）。

- シャープレシオ：日ごとの損益（決済日に計上、取引のない平日は 0）の 平均 ÷ 標準偏差。年率は √252 倍。
- PBO（Probability of Backtest Overfitting）：Bailey, Borwein, López de Prado & Zhu (2017) の CSCV。
  期間を S 区間に分け、半分の区間で一番良かった設定が、残り半分の区間で何位になるかを全通り調べる。
- DSR（Deflated Sharpe Ratio）：Bailey & López de Prado (2014)。試した数とリターン分布のゆがみ・裾の厚さを考慮して、
  「本当のシャープレシオが 0 より大きい確率」を求める。
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 252


# ---------------- 日ごとの損益とシャープレシオ ----------------

def business_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())


def daily_pnl(exit_times, pips, days: pd.DatetimeIndex) -> pd.Series:
    """決済した日に損益を計上した、平日ごとの損益（pips）。取引のない日は 0。"""
    s = pd.Series(np.asarray(pips, dtype=float), index=pd.DatetimeIndex(exit_times).normalize())
    return s.groupby(level=0).sum().reindex(days, fill_value=0.0)


def sharpe(daily: pd.Series | np.ndarray) -> float:
    """1日あたりのシャープレシオ（年率ではない）。ばらつきが 0 なら 0。"""
    x = np.asarray(daily, dtype=float)
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 else float(x.mean() / sd)


def annualized(sr_daily: float) -> float:
    return sr_daily * math.sqrt(PERIODS_PER_YEAR)


# ---------------- PBO（CSCV） ----------------

def pbo_cscv(matrix: pd.DataFrame, n_splits: int = 16) -> dict:
    """matrix：行＝日、列＝設定 の損益。戻り値：pbo、各分け方のロジット、選んだ設定の検証側の成績など。

    各分け方で「学習側（半分の区間）のシャープレシオが最大の設定」を選び、検証側（残り半分）での順位 ω を求める。
    λ = log(ω / (1 − ω))。λ ≤ 0（検証側で中位以下）の割合が PBO。
    """
    X = matrix.to_numpy(dtype=float)
    T, N = X.shape
    if N < 2 or T < n_splits * 2:
        return {"pbo": math.nan, "n_configs": N, "n_splits": n_splits, "logits": np.array([]),
                "is_sr": np.array([]), "oos_sr": np.array([]), "prob_oos_loss": math.nan}
    blocks = np.array_split(np.arange(T), n_splits)
    cnt = np.array([len(b) for b in blocks], dtype=float)                     # (S,)
    s1 = np.stack([X[b].sum(axis=0) for b in blocks])                       # (S, N)
    s2 = np.stack([(X[b] ** 2).sum(axis=0) for b in blocks])

    combos = np.array(list(itertools.combinations(range(n_splits), n_splits // 2)))
    is_mask = np.zeros((len(combos), n_splits))
    is_mask[np.arange(len(combos))[:, None], combos] = 1.0
    oos_mask = 1.0 - is_mask

    def sr(mask):
        n = mask @ cnt                                                      # (C,)
        m1 = (mask @ s1) / n[:, None]
        var = (mask @ s2) / n[:, None] - m1 ** 2
        sd = np.sqrt(np.clip(var, 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, m1 / sd, 0.0)

    is_sr, oos_sr = sr(is_mask), sr(oos_mask)                                # (C, N)
    best = is_sr.argmax(axis=1)
    chosen_oos = oos_sr[np.arange(len(combos)), best]
    # 検証側での順位（1 = 最下位、N = 最上位）。同じ値は平均の順位
    rank = (oos_sr < chosen_oos[:, None]).sum(axis=1) + ((oos_sr == chosen_oos[:, None]).sum(axis=1) + 1) / 2
    omega = rank / (N + 1)
    logits = np.log(omega / (1 - omega))
    return {
        "pbo": float((logits <= 0).mean()),
        "n_configs": N,
        "n_splits": n_splits,
        "logits": logits,
        "is_sr": is_sr[np.arange(len(combos)), best],
        "oos_sr": chosen_oos,
        "prob_oos_loss": float((chosen_oos < 0).mean()),
    }


# ---------------- DSR ----------------

def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """独立に n_trials 回試したとき、本当はどれも実力 0 でも出てしまう最大のシャープレシオの期待値（SR*）。"""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(sr: float, sr_benchmark: float, daily: np.ndarray) -> float:
    """シャープレシオが sr_benchmark を本当に上回っている確率（PSR）。分布のゆがみ・裾の厚さを考慮する。"""
    x = np.asarray(daily, dtype=float)
    T = len(x)
    if T < 3 or x.std() == 0:
        return math.nan
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))  # 正規分布なら 3
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return math.nan
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(T - 1) / math.sqrt(denom)))


def deflated_sharpe(daily: np.ndarray, n_trials: int, sr_variance: float) -> dict:
    """DSR：試した数で割り引いた基準（SR*）を、本当に上回っている確率。"""
    sr = sharpe(daily)
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    return {"sr": sr, "sr_annual": annualized(sr), "sr_star": sr_star, "sr_star_annual": annualized(sr_star),
            "dsr": probabilistic_sharpe(sr, sr_star, daily), "psr0": probabilistic_sharpe(sr, 0.0, daily),
            "n_trials": n_trials}
