"""手法の自動生成：遺伝的アルゴリズム（GA。要件定義書 6章。ライブラリは DEAP）。

遺伝的アルゴリズム：戦略の部品（遺伝子）の組み合わせを、生き物の進化のように「良いものどうしを組み合わせる（交叉）」
「一部をランダムに変える（突然変異）」をくり返して、より良い組み合わせを探す方法。

過学習を防ぐための決まりごと
- 遺伝子は、既存の指標と、有名手法の部品（ブレイクアウトの期間・パラボリックSAR・DMI など）の組み合わせに限る。
- 1つの戦略は「エントリーのきっかけ 1つ ＋ 絞り込みの条件 最大2つ」（最大3条件）。条件が多いほど適応度を割り引く。
- 1回の探索で評価する組み合わせの数（予算）を事前に決め、それを超えない。評価した組み合わせはすべて試行として記録する。
- 適応度は、最終テスト期間を除いた開発用データだけで計算する。開発用データを時間順に区間に分け、
  どの区間でも安定して良い組み合わせを高く評価する（区間ごとのシャープレシオの平均 − ばらつき）。
- 乱数のシード値を記録し、同じ結果を再現できるようにする。
- 見つかった戦略は「何通り試した中で一番良かったもの」なので、採用の判断は別の検証（ウォークフォワード・PBO・DSR・
  ランダムとの比較・年別）に合格したものだけ、しかもユーザーの承認後にする。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from deap import base, creator, tools

from . import backtest as bt
from . import famous, indicators, overfit
from .config import IndicatorParams, RiskParams
from .strategies import Strategy, _cross_down, _cross_up, _signal

GENE_VERSION = "ga-v1"
N_SEGMENTS = 5
MIN_TRADES_FOR_FITNESS = 20


# ---------------- 遺伝子（部品）のカタログ ----------------

def _sma(c: pd.Series, n: int) -> pd.Series:
    return c.rolling(n).mean()


def _ema(c: pd.Series, n: int) -> pd.Series:
    return c.ewm(span=n, adjust=False).mean()


def _first(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(fill_value=False).astype(bool)


def trig_ma(s: int, l: int):
    def f(d):
        a, b = _sma(d["close"], s), _sma(d["close"], l)
        return _signal(_cross_up(a, b), _cross_down(a, b))
    return f


def trig_macd(fa: int, sl: int, sg: int):
    def f(d):
        m = _ema(d["close"], fa) - _ema(d["close"], sl)
        s = _ema(m, sg)
        m = m.where(pd.Series(np.arange(len(d)) >= sl + sg, index=d.index))
        return _signal(_cross_up(m, s), _cross_down(m, s))
    return f


def trig_breakout(n: int):
    def f(d):
        hh, ll = d["high"].rolling(n).max().shift(), d["low"].rolling(n).min().shift()
        return _signal(_first(d["close"] > hh), _first(d["close"] < ll))
    return f


def trig_rsi(n: int, lo: float):
    def f(d):
        r = indicators.rsi(d["close"], n)
        prev = r.shift()
        return _signal((r >= lo) & (prev < lo), (r <= 100 - lo) & (prev > 100 - lo))
    return f


def trig_bb(n: int, k: float):
    def f(d):
        mid = _sma(d["close"], n)
        sd = d["close"].rolling(n).std(ddof=0)
        up, lo = mid + k * sd, mid - k * sd
        c, pc = d["close"], d["close"].shift()
        return _signal((c > lo) & (pc <= lo.shift()), (c < up) & (pc >= up.shift()))
    return f


def trig_sar():
    return lambda d: famous.sar_reversal(d, IndicatorParams())


def trig_dmi():
    def f(d):
        m = famous.dmi(d)
        return _signal(_cross_up(m["pdi"], m["mdi"]), _cross_down(m["pdi"], m["mdi"]))
    return f


TRIGGERS: list[tuple[str, str, object]] = (
    [(f"ma_{s}_{l}", f"SMA{s}がSMA{l}を抜ける", trig_ma(s, l)) for s, l in ((10, 50), (20, 50), (20, 75), (20, 100), (30, 100))]
    + [(f"macd_{a}_{b}_{c}", f"MACD({a},{b},{c})がシグナル線を抜ける", trig_macd(a, b, c)) for a, b, c in ((8, 17, 9), (12, 26, 9), (19, 39, 9))]
    + [(f"breakout_{n}", f"終値が直前{n}本の高値・安値を抜ける", trig_breakout(n)) for n in (10, 20, 40, 55)]
    + [(f"rsi_{n}_{lo}", f"RSI{n}が{lo}・{100 - lo}から戻る", trig_rsi(n, lo)) for n, lo in ((14, 30), (14, 20), (9, 30))]
    + [(f"bb_{n}_{k:g}", f"終値がボリンジャー({n}, ±{k:g}σ)の内側に戻る", trig_bb(n, k)) for n, k in ((20, 2.0), (20, 2.5))]
    + [("sar", "パラボリックSARの向きが変わる", trig_sar()), ("dmi", "+DIと−DIがクロスする", trig_dmi())]
)


def filt_trend(n: int):
    def f(d):
        m = _sma(d["close"], n)
        return d["close"] > m, d["close"] < m
    return f


def filt_rsi_not_extreme():
    def f(d):
        r = indicators.rsi(d["close"], 14)
        return r < 70, r > 30
    return f


def filt_adx(op: str, x: float):
    def f(d):
        adx = famous.dmi(d)["adx"]
        ok = adx > x if op == "gt" else adx < x
        return ok, ok
    return f


def filt_vol(high: bool):
    def f(d):
        ratio = d["atr"] / d["atr"].rolling(100).mean()
        ok = ratio > 1 if high else ratio < 1
        return ok, ok
    return f


def filt_squeeze():
    def f(d):
        mid = _sma(d["close"], 20)
        sd = d["close"].rolling(20).std(ddof=0)
        width = 4 * sd / mid
        sq = (width <= width.rolling(125).min()).fillna(False).astype(float)
        ok = sq.rolling(20, min_periods=1).max().astype(bool)
        return ok, ok
    return f


FILTERS: list[tuple[str, str, object]] = [
    ("none", "（なし）", None),
    *[(f"trend_{n}", f"トレンドの向き（終値がSMA{n}の上なら買いだけ・下なら売りだけ）", filt_trend(n)) for n in (50, 100, 200)],
    ("rsi_not_extreme", "RSIが行き過ぎていない（買いはRSI<70、売りはRSI>30）", filt_rsi_not_extreme()),
    ("adx_gt_20", "トレンドがある（ADX>20）", filt_adx("gt", 20)),
    ("adx_gt_25", "トレンドが強い（ADX>25）", filt_adx("gt", 25)),
    ("adx_lt_20", "トレンドが弱い・もみ合い（ADX<20）", filt_adx("lt", 20)),
    ("vol_high", "値動きが普段より大きい（ATRが過去100本平均より大）", filt_vol(True)),
    ("vol_low", "値動きが普段より小さい（ATRが過去100本平均より小）", filt_vol(False)),
    ("squeeze", "直近にバンドの幅が半年で最も狭くなった（スクイーズ）", filt_squeeze()),
]
TRIGGER_IDS = [t[0] for t in TRIGGERS]
FILTER_IDS = [f[0] for f in FILTERS]


# ---------------- 遺伝子 → 戦略 ----------------

def canonical(genome) -> tuple[int, int, int]:
    """[きっかけ, 条件1, 条件2] を、同じ意味なら同じ形にそろえる（条件の順番は関係ない・同じ条件の重複はなし）。"""
    t, a, b = int(genome[0]), int(genome[1]), int(genome[2])
    fs = sorted({x for x in (a, b) if x != 0})
    fs += [0] * (2 - len(fs))
    return t, fs[0], fs[1]


def genes_of(genome) -> dict:
    t, a, b = canonical(genome)
    return {"trigger": TRIGGER_IDS[t], "filters": [FILTER_IDS[x] for x in (a, b) if x != 0], "version": GENE_VERSION}


def key_of(genome) -> str:
    return "ga_" + hashlib.sha1(json.dumps(genes_of(genome), sort_keys=True).encode()).hexdigest()[:10]


def complexity(genome) -> int:
    t, a, b = canonical(genome)
    return 1 + (a != 0) + (b != 0)


def describe(genome) -> str:
    t, a, b = canonical(genome)
    parts = [TRIGGERS[t][1]] + [FILTERS[x][1] for x in (a, b) if x != 0]
    return "きっかけ：" + parts[0] + ("。条件：" + "、".join(parts[1:]) if len(parts) > 1 else "")


def make_func(genome):
    t, a, b = canonical(genome)
    trig = TRIGGERS[t][2]
    flts = [FILTERS[x][2] for x in (a, b) if x != 0]

    def func(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
        s = trig(d)
        buy, sell = s == 1, s == -1
        for f in flts:
            ok_buy, ok_sell = f(d)
            buy &= ok_buy.fillna(False).astype(bool)
            sell &= ok_sell.fillna(False).astype(bool)
        return _signal(buy, sell)
    return func


def to_strategy(genome, source: str = "") -> Strategy:
    g = genes_of(genome)
    return Strategy(key_of(genome), f"GA戦略 {key_of(genome)[3:9]}", "GA", describe(genome), make_func(genome),
                    family="ga", source=source, spec={"ga_genes": g})


# ---------------- 適応度 ----------------

@dataclass
class Fitness:
    score: float
    n_trades: int
    segment_sharpe: list[float]
    total_pips: float


def fitness(ind: pd.DataFrame, genome, risk: RiskParams, rules: bt.ExitRules, costs: bt.Costs, pip: float,
            bars: bt.Bars) -> Fitness:
    """コスト後の成績。開発用データを時間順に N_SEGMENTS 区間に分け、区間ごとのシャープレシオ（年率）の
    平均 − 0.5 × ばらつき − 0.1 ×（条件の数 − 1）。取引が100回未満なら −1、20回未満なら評価しない（−10）。"""
    sig = make_func(genome)(ind, IndicatorParams()).where(ind["atr"].notna(), 0).astype(int).tolist()
    n = len(ind)
    trades = bt.simulate_raw(bars, sig, risk, rules, costs, pip, bt.WARMUP, n)
    n_tr = len(trades)
    total = float(sum(t[9] for t in trades))
    if n_tr < MIN_TRADES_FOR_FITNESS:
        return Fitness(-10.0, n_tr, [], total)
    days = overfit.business_days(ind.index[bt.WARMUP], ind.index[-1])
    daily = overfit.daily_pnl(ind.index[[t[3] for t in trades]], [t[9] for t in trades], days)
    segs = np.array_split(daily.to_numpy(), N_SEGMENTS)
    srs = [overfit.annualized(overfit.sharpe(s)) for s in segs]
    score = float(np.mean(srs) - 0.5 * np.std(srs) - 0.1 * (complexity(genome) - 1))
    if n_tr < bt.MIN_TRADES_RELIABLE:
        score -= 1.0
    return Fitness(score, n_tr, srs, total)


# ---------------- 探索 ----------------

@dataclass
class SearchResult:
    seed: int
    budget: int
    evaluated: pd.DataFrame          # 評価した組み合わせ（genome, key, 説明, 適応度, 取引回数, 区間ごとのシャープレシオ）
    top: list[tuple[int, int, int]]  # 適応度の高い順の組み合わせ（重複なし）
    trials: list[dict] = field(default_factory=list)  # 試行の記録用
    kind: str = "full"               # full（本格的な探索）/ neighborhood（上位候補の周辺だけ）
    trace: list[float] = field(default_factory=list)  # 評価した順の適応度（何通り目で最良が見つかったかの確認用）


def _setup_creator():
    if not hasattr(creator, "FxFitness"):
        creator.create("FxFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "FxIndividual"):
        creator.create("FxIndividual", list, fitness=creator.FxFitness)


class _Evaluator:
    """予算の範囲で組み合わせを評価する。同じ組み合わせは二度評価しない（試行も1つ）。"""

    def __init__(self, ind, risk, rules, costs, pip, budget, mult=None):
        self.ind, self.risk, self.rules, self.costs, self.pip = ind, risk, rules, costs, pip
        self.bars = bt.Bars(ind, mult)
        self.budget = budget
        self.cache: dict[tuple, Fitness] = {}

    @property
    def remaining(self) -> int:
        return self.budget - len(self.cache)

    def __call__(self, genome) -> float:
        g = canonical(genome)
        if g not in self.cache:
            if self.remaining <= 0:
                return -math.inf  # 予算を使い切ったら、新しい組み合わせは評価しない（試行にも数えない）
            self.cache[g] = fitness(self.ind, g, self.risk, self.rules, self.costs, self.pip, self.bars)
        return self.cache[g].score


def _result(ev: _Evaluator, seed: int, budget: int, kind: str, base_p: IndicatorParams, top_k: int = 10) -> SearchResult:
    from .trials import strategy_config
    rows = [{"genome": g, "key": key_of(g), "rule": describe(g), "fitness": f.score, "trades": f.n_trades,
             "total_pips": f.total_pips, "segment_sharpe": f.segment_sharpe} for g, f in ev.cache.items()]
    cols = ["genome", "key", "rule", "fitness", "trades", "total_pips", "segment_sharpe"]
    df = pd.DataFrame(rows, columns=cols).sort_values("fitness", ascending=False, kind="stable").reset_index(drop=True)
    top = [tuple(g) for g in df["genome"].head(top_k)]
    tr = [{"family": "strategy", "name": "ga", "purpose": f"ga_{kind}",
           "config": strategy_config(r["key"], base_p, ev.risk, ev.rules, spec={"ga_genes": genes_of(r["genome"])}),
           "metrics": {"fitness": r["fitness"], "trades": r["trades"], "total_pips": r["total_pips"], "seed": seed}}
          for r in rows]
    trace = [float(f.score) for f in ev.cache.values()]  # dict は入れた順を保つ＝評価した順
    return SearchResult(seed, budget, df, top, tr, kind, trace)


def search(dev: pd.DataFrame, risk: RiskParams, rules: bt.ExitRules, costs: bt.Costs, pip: float, budget: int,
           seed: int, pop_size: int = 40, holdout_start: pd.Timestamp | None = None, mult=None,
           base_p: IndicatorParams = IndicatorParams()) -> SearchResult:
    """本格的な探索。評価する組み合わせの数は budget まで。dev は開発用データ（最終テスト期間より前）。"""
    if holdout_start is not None and len(dev) and dev.index[-1] >= holdout_start:
        from .holdout import FinalTestAccessError
        raise FinalTestAccessError("最終テスト期間のデータが GA の探索に渡されています")
    _setup_creator()
    rnd = random.Random(seed)
    ind = indicators.add_all(dev, base_p)
    ev = _Evaluator(ind, risk, rules, costs, pip, budget, mult)

    def rand_genome():
        return creator.FxIndividual([rnd.randrange(len(TRIGGERS)), rnd.randrange(len(FILTERS)), rnd.randrange(len(FILTERS))])

    def mutate(ind_):
        k = rnd.randrange(3)
        ind_[k] = rnd.randrange(len(TRIGGERS) if k == 0 else len(FILTERS))
        return (ind_,)

    def mate(a, b):
        for k in range(3):
            if rnd.random() < 0.5:
                a[k], b[k] = b[k], a[k]
        return a, b

    pop = [rand_genome() for _ in range(pop_size)]
    for p in pop:
        p.fitness.values = (ev(p),)
    while ev.remaining > 0:
        before = ev.remaining
        # トーナメント選択（ランダムに選んだ3つの中で一番良いものを選ぶ）。乱数は seed で固定した rnd だけを使う
        offspring = [creator.FxIndividual(list(tools.selBest([rnd.choice(pop) for _ in range(3)], 1)[0]))
                     for _ in range(len(pop))]
        for i in range(0, len(offspring) - 1, 2):
            if rnd.random() < 0.6:
                mate(offspring[i], offspring[i + 1])
        for o in offspring:
            if rnd.random() < 0.3:
                mutate(o)
            o.fitness.values = (ev(o),)
        pop = tools.selBest(pop + offspring, pop_size)
        if ev.remaining == before:  # 新しい組み合わせが出てこなくなったら、ランダムな組み合わせを混ぜる
            for k in range(min(5, len(pop))):
                pop[-1 - k] = rand_genome()
                pop[-1 - k].fitness.values = (ev(pop[-1 - k]),)
            if ev.remaining == before:
                break
    return _result(ev, seed, budget, "full", base_p)


def neighbors(genome) -> list[tuple[int, int, int]]:
    """1つの遺伝子だけを変えた組み合わせ（周辺）。"""
    out = set()
    g = list(canonical(genome))
    for k in range(3):
        for v in range(len(TRIGGERS) if k == 0 else len(FILTERS)):
            h = g.copy()
            h[k] = v
            out.add(canonical(h))
    out.discard(canonical(genome))
    return sorted(out)


def neighborhood_search(dev: pd.DataFrame, previous_top: list, risk: RiskParams, rules: bt.ExitRules, costs: bt.Costs,
                        pip: float, budget: int, seed: int, holdout_start: pd.Timestamp | None = None, mult=None,
                        base_p: IndicatorParams = IndicatorParams(), skip=frozenset()) -> SearchResult:
    """定期的な再探索：これまでの上位候補の周辺（遺伝子を1つだけ変えたもの）のうち、まだ評価していないもの（skip にないもの）
    だけを、budget まで評価する。開発用データは境界で固定されていて変わらないので、評価済みの組み合わせを評価し直しても
    同じ結果になるだけ（予算と時間のむだ）なので、評価し直さない。"""
    if holdout_start is not None and len(dev) and dev.index[-1] >= holdout_start:
        from .holdout import FinalTestAccessError
        raise FinalTestAccessError("最終テスト期間のデータが GA の探索に渡されています")
    rnd = random.Random(seed)
    ind = indicators.add_all(dev, base_p)
    ev = _Evaluator(ind, risk, rules, costs, pip, budget, mult)
    cands = [canonical(g) for g in previous_top]
    for g in list(cands):
        nb = neighbors(g)
        rnd.shuffle(nb)
        cands += nb
    done = {canonical(g) for g in skip}
    for g in dict.fromkeys(cands):
        if ev.remaining <= 0:
            break
        if g not in done:
            ev(g)
    return _result(ev, seed, budget, "neighborhood", base_p)
