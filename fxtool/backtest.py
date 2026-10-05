"""バックテスト・比較基準（何もしない／ランダム売買）・ウォークフォワード検証。

売買のルール
- シグナルが出た足の「次の足の始値」でエントリーする（シグナルの足が確定するまで注文できないため）。
- 損切り・利確はシグナルの足の ATR から決める。判定は strategies.exit_hit（フェーズ1と共通）。
- 反対シグナルが出たら次の足の始値で決済し、そのまま反対方向に入り直す（ドテン）。
- ポジションは同時に1つまで。持っている間の同じ方向のシグナルは無視する。
- 損益は pips。毎回の取引から往復の取引コスト（スプレッド + スリッページ×2）を引く。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import indicators, strategies
from .config import IndicatorParams, RiskParams
from .strategies import Strategy, exit_hit, levels

# ウォークフォワードのパラメータ候補で最も長い期間でも指標が計算できるよう、最初のこの本数は使わない
WARMUP = 150

PARAM_GRIDS: dict[str, list[dict]] = {
    "ma_cross": [dict(sma_short=s, sma_long=l) for s in (10, 20, 30) for l in (50, 75, 100)],
    "macd_cross": [dict(macd_fast=f, macd_slow=s, macd_signal=g)
                   for f, s in ((8, 17), (12, 26), (19, 39)) for g in (5, 9)],
    "breakout": [dict(donchian_period=n) for n in (10, 20, 40, 55)],
    "rsi_reversal": [dict(rsi_period=n, rsi_low=lo, rsi_high=100 - lo) for n in (9, 14, 21) for lo in (20, 25, 30)],
    "bb_reversion": [dict(bb_period=n, bb_std=s) for n in (14, 20, 30) for s in (1.5, 2.0, 2.5)],
}


def describe_params(key: str, ov: dict) -> str:
    if key == "ma_cross":
        return f"SMA {ov['sma_short']}/{ov['sma_long']}"
    if key == "macd_cross":
        return f"MACD {ov['macd_fast']}/{ov['macd_slow']}/{ov['macd_signal']}"
    if key == "breakout":
        return f"{ov['donchian_period']}本"
    if key == "rsi_reversal":
        return f"RSI{ov['rsi_period']} {ov['rsi_low']:g}/{ov['rsi_high']:g}"
    if key == "bb_reversion":
        return f"BB {ov['bb_period']} ±{ov['bb_std']:g}σ"
    return str(ov)


@dataclass(frozen=True)
class Costs:
    spread: float = 0.3      # 往復のスプレッド（pips）
    slippage: float = 0.2    # 片道のスリッページ（pips）

    @property
    def round_trip(self) -> float:
        return self.spread + 2 * self.slippage


@dataclass(frozen=True)
class ExitRules:
    opposite_exit: bool = True
    max_bars: int = 0        # 0 なら保有本数の上限なし


class Bars:
    """シミュレーション用に列を Python のリストにしたもの（ループを速くするため）。"""

    def __init__(self, ind: pd.DataFrame):
        self.index = ind.index
        self.n = len(ind)
        self.o = ind["open"].tolist()
        self.h = ind["high"].tolist()
        self.l = ind["low"].tolist()
        self.c = ind["close"].tolist()
        self.atr = ind["atr"].tolist()


TRADE_FIELDS = ["dir", "signal_i", "entry_i", "exit_i", "entry", "exit", "sl", "tp", "reason", "pips"]


def simulate_raw(
    bars: Bars, sig: list[int], risk: RiskParams, rules: ExitRules, costs: Costs, pip: float,
    start: int = 0, end: int | None = None,
) -> list[tuple]:
    """[start, end) の範囲だけで売買する。範囲の終わりで持っているポジションはその足の終値で決済する。"""
    end = bars.n if end is None else end
    o, h, l, c, atr = bars.o, bars.h, bars.l, bars.c, bars.atr
    cost = costs.round_trip
    out = []
    k = start
    while k < end - 1:
        d = sig[k]
        a = atr[k]
        if d == 0 or a != a:  # a != a は NaN
            k += 1
            continue
        e = k + 1
        entry = o[e]
        sl, tp = levels(entry, d, a, risk)
        j = e
        while True:
            hit = exit_hit(d, o[j], h[j], l[j], sl, tp)
            if hit:
                reason, px = hit
                exit_j, next_k = j, j
                break
            if rules.max_bars and j - e + 1 >= rules.max_bars:
                reason, px, exit_j, next_k = "期限", c[j], j, j
                break
            if rules.opposite_exit and sig[j] == -d and j + 1 < end:
                reason, px, exit_j, next_k = "反対シグナル", o[j + 1], j + 1, j
                break
            if j == end - 1:
                reason, px, exit_j, next_k = "期間末", c[j], j, end
                break
            j += 1
        out.append((d, k, e, exit_j, entry, px, sl, tp, reason, d * (px - entry) / pip - cost))
        k = next_k
    return out


def simulate(bars: Bars, sig: list[int], risk: RiskParams, rules: ExitRules, costs: Costs, pip: float,
             start: int = 0, end: int | None = None) -> pd.DataFrame:
    return pd.DataFrame(simulate_raw(bars, sig, risk, rules, costs, pip, start, end), columns=TRADE_FIELDS)


def metrics(pips) -> dict:
    """pips: 決済順に並んだ各取引の損益（pips）。"""
    p = np.asarray(pips, dtype=float)
    n = len(p)
    if n == 0:
        return dict(n=0, win_rate=math.nan, total=0.0, avg=math.nan, pf=math.nan, max_dd=0.0)
    gross_win = p[p > 0].sum()
    gross_loss = -p[p < 0].sum()
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = math.inf if gross_win > 0 else math.nan
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.maximum(eq, 0))  # 開始時点（0）もピークに含める
    return dict(
        n=n, win_rate=float((p > 0).mean()), total=float(eq[-1]), avg=float(p.mean()),
        pf=float(pf), max_dd=float(min(0.0, (eq - peak).min())),
    )


def equity_curve(exit_i, pips, n: int) -> np.ndarray:
    """決済した足ごとに損益を積み上げた、長さ n の累積損益（pips）。"""
    return np.cumsum(np.bincount(np.asarray(exit_i, dtype=int), weights=np.asarray(pips, dtype=float), minlength=n))


def signal_rate(sig: list[int], start: int, end: int) -> float:
    return float(np.count_nonzero(np.asarray(sig[start:end]))) / max(1, end - start)


def random_baseline(
    bars: Bars, windows: list[tuple[int, int, float]], risk: RiskParams, rules: ExitRules, costs: Costs,
    pip: float, runs: int = 200, seed: int = 0, curves: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """ランダム売買を runs 回行う。windows は (開始, 終了, 1本あたりのシグナル発生率) のリスト。

    方向は買い・売りを半々でランダムに決め、決済ルールとコストは戦略と同じにする。
    戻り値：試行ごとの成績（metrics の列）と、curves=True なら累積損益の 5/50/95% 点（3 × n）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    curve_arr = np.empty((runs, bars.n)) if curves else None
    for r in range(runs):
        sig = np.zeros(bars.n, dtype=int)
        for a, b, rate in windows:
            hits = rng.random(b - a) < rate
            sig[a:b][hits] = rng.choice([-1, 1], size=int(hits.sum()))
        sig_l = sig.tolist()
        trades = []
        for a, b, _ in windows:
            trades += simulate_raw(bars, sig_l, risk, rules, costs, pip, a, b)
        pips = [t[9] for t in trades]
        rows.append(metrics(pips))
        if curves:
            curve_arr[r] = equity_curve([t[3] for t in trades], pips, bars.n)
    q = np.percentile(curve_arr, [5, 50, 95], axis=0) if curves else None
    return pd.DataFrame(rows), q


def beat_ratio(value: float, random_totals: pd.Series) -> float:
    """ランダム売買の試行のうち、value が上回った割合（0〜1）。"""
    return float((random_totals < value).mean())


# ---------------- ウォークフォワード ----------------

@dataclass
class Candidate:
    overrides: dict
    params: IndicatorParams
    bars: Bars
    sig: list[int]


def candidates(raw: pd.DataFrame, strategy: Strategy, base: IndicatorParams,
               cache: dict | None = None) -> list[Candidate]:
    cache = {} if cache is None else cache
    out = []
    for ov in PARAM_GRIDS[strategy.key]:
        p = replace(base, **ov)
        if p not in cache:
            cache[p] = indicators.add_all(raw, p)
        ind = cache[p]
        out.append(Candidate(ov, p, Bars(ind), strategies.signals(ind, strategy, p).tolist()))
    return out


def make_folds(n: int, k: int, train_mult: int, start: int = WARMUP) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """[(学習区間), (検証区間)] を k 個。学習区間の長さは検証区間の train_mult 倍で、1区間ずつずらしていく。"""
    unit = (n - start) // (k + train_mult)
    if unit < 2:
        raise ValueError("データが少なすぎてウォークフォワードの区間を作れません")
    folds = []
    for i in range(k):
        a = start + i * unit
        b = a + train_mult * unit
        c = n if i == k - 1 else b + unit
        folds.append(((a, b), (b, c)))
    return folds


@dataclass
class Fold:
    train: tuple[int, int]
    test: tuple[int, int]
    overrides: dict
    train_metrics: dict
    test_trades: pd.DataFrame
    test_rate: float


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    test_trades: pd.DataFrame
    test_metrics: dict
    train_avg: float             # 学習区間（選ばれたパラメータ）の1取引あたり平均。区間が重なるので合計ではなく平均で比べる
    positive_folds: int


def walk_forward(
    raw: pd.DataFrame, strategy: Strategy, base: IndicatorParams, risk: RiskParams, rules: ExitRules,
    costs: Costs, pip: float, k: int = 5, train_mult: int = 2, min_trades: int = 5, cache: dict | None = None,
) -> WalkForwardResult:
    """各学習区間で損益合計が最大のパラメータを選び（取引が min_trades 回未満の候補は除外）、
    直後の検証区間でそのまま使ったときの成績を測る。"""
    cands = candidates(raw, strategy, base, cache)
    folds = []
    for (a, b), (b2, c) in make_folds(len(raw), k, train_mult):
        best, best_m, best_score = cands[0], None, -math.inf
        for cand in cands:
            m = metrics([t[9] for t in simulate_raw(cand.bars, cand.sig, risk, rules, costs, pip, a, b)])
            score = m["total"] if m["n"] >= min_trades else -math.inf
            if best_m is None or score > best_score:
                best, best_m, best_score = cand, m, score
        test = simulate(best.bars, best.sig, risk, rules, costs, pip, b2, c)
        folds.append(Fold((a, b), (b2, c), best.overrides, best_m, test, signal_rate(best.sig, b2, c)))

    test_trades = pd.concat([f.test_trades for f in folds], ignore_index=True)
    train_n = sum(f.train_metrics["n"] for f in folds)
    return WalkForwardResult(
        folds=folds,
        test_trades=test_trades,
        test_metrics=metrics(test_trades["pips"]),
        train_avg=float(sum(f.train_metrics["total"] for f in folds) / train_n) if train_n else math.nan,
        positive_folds=sum(f.test_trades["pips"].sum() > 0 for f in folds),
    )


@dataclass
class StrategyReport:
    strategy: Strategy
    full_trades: pd.DataFrame        # 全期間・初期パラメータ
    full_metrics: dict
    full_random: pd.DataFrame        # 同じ頻度のランダム売買（試行ごとの成績）
    full_beat: float
    wf: WalkForwardResult
    wf_random: pd.DataFrame          # 検証区間だけで、同じ頻度のランダム売買
    wf_beat: float
    verdict: tuple[str, str]


@dataclass
class Report:
    start: int
    reports: list[StrategyReport]
    baseline_random: pd.DataFrame    # 全戦略の平均頻度でのランダム売買
    baseline_curves: np.ndarray      # その累積損益の 5/50/95% 点
    k: int


def evaluate_all(
    raw: pd.DataFrame, base: IndicatorParams, risk: RiskParams, rules: ExitRules, costs: Costs, pip: float,
    runs: int = 200, k: int = 5, train_mult: int = 2,
) -> Report:
    cache: dict = {}
    ind = indicators.add_all(raw, base)
    cache[base] = ind
    bars = Bars(ind)
    n = len(ind)
    reports, rates = [], []
    for i, st in enumerate(strategies.STRATEGIES):
        sig = strategies.signals(ind, st, base).tolist()
        full = simulate(bars, sig, risk, rules, costs, pip, WARMUP, n)
        fm = metrics(full["pips"])
        rate = signal_rate(sig, WARMUP, n)
        rates.append(rate)
        full_rand, _ = random_baseline(bars, [(WARMUP, n, rate)], risk, rules, costs, pip, runs, seed=i)

        wf = walk_forward(raw, st, base, risk, rules, costs, pip, k, train_mult, cache=cache)
        wf_windows = [(f.test[0], f.test[1], f.test_rate) for f in wf.folds]
        wf_rand, _ = random_baseline(bars, wf_windows, risk, rules, costs, pip, runs, seed=100 + i)
        wf_beat = beat_ratio(wf.test_metrics["total"], wf_rand["total"])
        reports.append(StrategyReport(
            st, full, fm, full_rand, beat_ratio(fm["total"], full_rand["total"]),
            wf, wf_rand, wf_beat, assess(wf, wf_beat, k),
        ))

    base_rand, curves = random_baseline(
        bars, [(WARMUP, n, float(np.mean(rates)))], risk, rules, costs, pip, runs, seed=999, curves=True
    )
    return Report(WARMUP, reports, base_rand, curves, k)


def assess(wf: WalkForwardResult, beat: float, k: int) -> tuple[str, str]:
    """ウォークフォワードの検証区間の成績から、優位性がありそうかを機械的に判定する。"""
    m = wf.test_metrics
    if m["n"] < 20:
        return "判断材料不足", f"検証区間での取引が {m['n']} 回しかなく、偶然との区別がつきません。"
    if m["total"] <= 0:
        return "優位性なし", "コストを含めると、まだ見ていない期間でマイナスでした。"
    if beat >= 0.95 and wf.positive_folds >= math.ceil(k * 0.6):
        return "有望（要継続監視）", (
            f"検証区間でプラス、ランダム売買の {beat:.0%} を上回り、{wf.positive_folds}/{k} 区間で黒字でした。"
            "ただし複数の戦略を試した中での結果なので、偶然の可能性は残ります。"
        )
    if beat >= 0.95:
        return "不安定", f"合計ではランダム売買を上回りましたが、黒字の区間は {wf.positive_folds}/{k} だけで、一部の期間に頼った成績です。"
    return "偶然の範囲", f"検証区間ではプラスでしたが、ランダム売買の {beat:.0%} しか上回っておらず、偶然でも出る程度の成績です。"
