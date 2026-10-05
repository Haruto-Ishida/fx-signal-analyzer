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
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import indicators, overfit, strategies
from .config import IndicatorParams, RiskParams
from .strategies import Strategy, exit_hit, levels

# ウォークフォワードのパラメータ候補で最も長い期間でも指標が計算できるよう、最初のこの本数は使わない
WARMUP = 150

PARAM_GRIDS = strategies.PARAM_GRIDS  # 以前の場所からも参照できるように


def describe_params(key: str, ov: dict) -> str:
    if not ov:
        return "初期設定"
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
    return "・".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in ov.items())


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

    def __init__(self, ind: pd.DataFrame, mult=None):
        self.index = ind.index
        self.n = len(ind)
        self.o = ind["open"].tolist()
        self.h = ind["high"].tolist()
        self.l = ind["low"].tolist()
        self.c = ind["close"].tolist()
        self.atr = ind["atr"].tolist()
        # 足ごとのスプレッドの倍率（costmodel.spread_multipliers）。None なら一定のコスト
        self.mult = None if mult is None else [float(x) for x in mult]
        if self.mult is not None and len(self.mult) != self.n:
            raise ValueError("スプレッドの倍率の数が足の数と合いません")


TRADE_FIELDS = ["dir", "signal_i", "entry_i", "exit_i", "entry", "exit", "sl", "tp", "reason", "pips"]


def simulate_raw(
    bars: Bars, sig: list[int], risk: RiskParams, rules: ExitRules, costs: Costs, pip: float,
    start: int = 0, end: int | None = None, exit_sig: list[int] | None = None,
) -> list[tuple]:
    """[start, end) の範囲だけで売買する。範囲の終わりで持っているポジションはその足の終値で決済する。

    exit_sig（有名手法などの決済のルール）：その足の終値で exit_sig が持っている方向（または 2）なら、次の足の始値で決済する。"""
    end = bars.n if end is None else end
    o, h, l, c, atr = bars.o, bars.h, bars.l, bars.c, bars.atr
    cost = costs.round_trip
    mult = bars.mult  # 時間帯によるスプレッドの倍率（なければ一定のコスト）
    half, slip2 = costs.spread / 2, 2 * costs.slippage
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
            if exit_sig is not None and exit_sig[j] in (d, 2) and j + 1 < end:
                reason, px, exit_j, next_k = "決済シグナル", o[j + 1], j + 1, j
                break
            if rules.opposite_exit and sig[j] == -d and j + 1 < end:
                reason, px, exit_j, next_k = "反対シグナル", o[j + 1], j + 1, j
                break
            if j == end - 1:
                reason, px, exit_j, next_k = "期間末", c[j], j, end
                break
            j += 1
        c_trade = cost if mult is None else half * (mult[e] + mult[exit_j]) + slip2  # 入る足と出る足のスプレッドの半分ずつ
        out.append((d, k, e, exit_j, entry, px, sl, tp, reason, d * (px - entry) / pip - c_trade))
        k = next_k
    return out


def simulate(bars: Bars, sig: list[int], risk: RiskParams, rules: ExitRules, costs: Costs, pip: float,
             start: int = 0, end: int | None = None, exit_sig: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(simulate_raw(bars, sig, risk, rules, costs, pip, start, end, exit_sig), columns=TRADE_FIELDS)


NO_BARRIER = 1e6  # 「使わない」損切り・利確（ATR の 100万倍。実際には届かない）


def strategy_exits(st: Strategy, risk: RiskParams, rules: ExitRules) -> tuple[RiskParams, ExitRules]:
    """戦略に元のルールの決済（exits）があればそれを、なければ共通の設定を使う。"""
    e = st.exits
    if not e:
        return risk, rules
    def barrier(name, default):
        v = e.get(name, default)
        return NO_BARRIER if v is None else float(v)
    r = RiskParams(sl_atr=barrier("sl_atr", risk.sl_atr), tp_atr=barrier("tp_atr", risk.tp_atr),
                   valid_bars=risk.valid_bars)
    ru = ExitRules(opposite_exit=bool(e.get("opposite_exit", rules.opposite_exit)),
                   max_bars=int(e.get("max_bars", rules.max_bars)))
    return r, ru


def metrics(pips) -> dict:
    """pips: 決済順に並んだ各取引の損益（pips）。"""
    p = np.asarray(pips, dtype=float)
    n = len(p)
    if n == 0:
        return dict(n=0, win_rate=math.nan, total=0.0, avg=math.nan, pf=math.nan, max_dd=0.0,
                    avg_win=math.nan, avg_loss=math.nan, payoff=math.nan)
    gross_win = p[p > 0].sum()
    gross_loss = -p[p < 0].sum()
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = math.inf if gross_win > 0 else math.nan
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.maximum(eq, 0))  # 開始時点（0）もピークに含める
    avg_win = float(p[p > 0].mean()) if (p > 0).any() else math.nan
    avg_loss = float(p[p < 0].mean()) if (p < 0).any() else math.nan
    return dict(
        n=n, win_rate=float((p > 0).mean()), total=float(eq[-1]), avg=float(p.mean()),
        pf=float(pf), max_dd=float(min(0.0, (eq - peak).min())),
        avg_win=avg_win, avg_loss=avg_loss,
        payoff=avg_win / -avg_loss if not (math.isnan(avg_win) or math.isnan(avg_loss)) else math.nan,
    )


MIN_TRADES_RELIABLE = 100  # 研究ノート5：これ未満の取引回数の成績は「参考値」


def equity_curve(exit_i, pips, n: int) -> np.ndarray:
    """決済した足ごとに損益を積み上げた、長さ n の累積損益（pips）。"""
    return np.cumsum(np.bincount(np.asarray(exit_i, dtype=int), weights=np.asarray(pips, dtype=float), minlength=n))


def signal_rate(sig: list[int], start: int, end: int) -> float:
    return float(np.count_nonzero(np.asarray(sig[start:end]))) / max(1, end - start)


def random_baseline(
    bars: Bars, windows: list[tuple[int, int, float]], risk: RiskParams, rules: ExitRules, costs: Costs,
    pip: float, runs: int = 200, seed: int = 0, curves: bool = False, exit_sig: list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """ランダム売買を runs 回行う。windows は (開始, 終了, 1本あたりのシグナル発生率) のリスト。

    方向は買い・売りを半々でランダムに決め、決済ルール（exit_sig を含む）とコストは戦略と同じにする。
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
            trades += simulate_raw(bars, sig_l, risk, rules, costs, pip, a, b, exit_sig)
        pips = [t[9] for t in trades]
        rows.append(metrics(pips))
        if curves:
            curve_arr[r] = equity_curve([t[3] for t in trades], pips, bars.n)
    q = np.percentile(curve_arr, [5, 50, 95], axis=0) if curves else None
    return pd.DataFrame(rows), q


def beat_ratio(value: float, random_totals: pd.Series) -> float:
    """ランダム売買の試行のうち、value が上回った割合（0〜1）。"""
    return float((random_totals < value).mean())


# ---------------- ランダムウォーク（研究ノート 1-1） ----------------

def random_walk_ohlc(ohlc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """値動きの順番だけをランダムに並べ替えた価格。1本ごとの形（窓・高値・安値・終値の動き）と
    値動きの大きさの分布はそのままで、流れ（トレンド・パターン）だけがなくなる。時刻はそのまま。"""
    o, h, l, c = (ohlc[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    prev_close = np.r_[o[0], c[:-1]]
    gap, up, down, body = o - prev_close, h - o, l - o, c - o
    order = rng.permutation(np.arange(1, len(ohlc)))
    order = np.r_[0, order]
    gap, up, down, body = gap[order], up[order], down[order], body[order]
    gap[0] = 0.0
    close = o[0] + np.cumsum(gap + body)
    open_ = close - body
    return pd.DataFrame({"open": open_, "high": open_ + up, "low": open_ + down, "close": close}, index=ohlc.index)


def random_walk_baseline(ohlc: pd.DataFrame, base: IndicatorParams, risk: RiskParams, rules: ExitRules,
                         costs: Costs, pip: float, paths: int = 50, seed: int = 0,
                         strategy_list: list[Strategy] | None = None, mult=None) -> pd.DataFrame:
    """ランダムウォークの価格で各戦略（初期パラメータ）を動かした成績。行＝試行、列＝戦略の損益合計（pips）。
    時刻は本物と同じなので、時間帯によるスプレッドの倍率（mult）も同じものを使う。"""
    rng = np.random.default_rng(seed)
    sts = list(strategy_list) if strategy_list is not None else strategies.STRATEGIES
    rows = []
    for _ in range(paths):
        ind = indicators.add_all(random_walk_ohlc(ohlc, rng), base)
        bars = Bars(ind, mult)
        row = {}
        for st in sts:
            p = st.params(base)  # その戦略の初期パラメータ（最初からある5つの戦略では共通の初期値）
            ind_p = ind if p == base else indicators.add_all(ind[["open", "high", "low", "close"]], p)
            bars_p = bars if p == base else Bars(ind_p, mult)
            rs, ru = strategy_exits(st, risk, rules)
            sig = strategies.signals(ind_p, st, p).tolist()
            row[st.key] = sum(t[9] for t in simulate_raw(bars_p, sig, rs, ru, costs, pip, WARMUP, len(ind_p),
                                                         strategies.exit_signals(ind_p, st, p)))
        rows.append(row)
    return pd.DataFrame(rows)


def yearly(trades: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """決済した年ごとの成績。"""
    if trades.empty:
        return pd.DataFrame(columns=["year", "n", "total", "pf"])
    years = pd.DatetimeIndex(index[trades["exit_i"].to_numpy(dtype=int)]).year
    out = []
    for y, g in trades.groupby(years):
        m = metrics(g["pips"])
        out.append({"year": int(y), "n": m["n"], "total": m["total"], "pf": m["pf"]})
    return pd.DataFrame(out)


# ---------------- ウォークフォワード ----------------

@dataclass
class Candidate:
    overrides: dict
    params: IndicatorParams
    bars: Bars
    sig: list[int]
    exit_sig: list[int] | None = None


def candidates(raw: pd.DataFrame, strategy: Strategy, base: IndicatorParams,
               cache: dict | None = None, mult=None) -> list[Candidate]:
    cache = {} if cache is None else cache
    out = []
    for ov in strategy.grid():
        p = replace(base, **ov)
        if p not in cache:
            cache[p] = indicators.add_all(raw, p)
        ind = cache[p]
        out.append(Candidate(ov, p, Bars(ind, mult), strategies.signals(ind, strategy, p).tolist(),
                             strategies.exit_signals(ind, strategy, p)))
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
    candidate_metrics: list[dict] = field(default_factory=list)  # 学習区間での各候補の成績（Strategy.grid() の順）


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
    mult=None,
) -> WalkForwardResult:
    """各学習区間で損益合計が最大のパラメータを選び（取引が min_trades 回未満の候補は除外）、
    直後の検証区間でそのまま使ったときの成績を測る。戦略に元のルールの決済（exits・exit_func）があればそれを使う。"""
    risk, rules = strategy_exits(strategy, risk, rules)
    cands = candidates(raw, strategy, base, cache, mult)
    folds = []
    for (a, b), (b2, c) in make_folds(len(raw), k, train_mult):
        best, best_m, best_score = cands[0], None, -math.inf
        cand_ms = []
        for cand in cands:
            m = metrics([t[9] for t in simulate_raw(cand.bars, cand.sig, risk, rules, costs, pip, a, b, cand.exit_sig)])
            cand_ms.append(m)
            score = m["total"] if m["n"] >= min_trades else -math.inf
            if best_m is None or score > best_score:
                best, best_m, best_score = cand, m, score
        test = simulate(best.bars, best.sig, risk, rules, costs, pip, b2, c, best.exit_sig)
        folds.append(Fold((a, b), (b2, c), best.overrides, best_m, test, signal_rate(best.sig, b2, c), cand_ms))

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
    full_trades: pd.DataFrame        # 開発用データ全体・初期パラメータ
    full_metrics: dict
    full_random: pd.DataFrame        # 同じ頻度のランダム売買（試行ごとの成績）
    full_beat: float
    wf: WalkForwardResult
    wf_random: pd.DataFrame          # 検証区間だけで、同じ頻度のランダム売買
    wf_beat: float
    verdict: tuple[str, str]
    full_daily: pd.Series = None     # 日ごとの損益（開発用データ全体・初期パラメータ）
    wf_daily: pd.Series = None       # 日ごとの損益（ウォークフォワードの検証区間）
    full_sharpe: float = math.nan    # 年率
    wf_sharpe: float = math.nan      # 年率
    rw_beat: float = math.nan        # ランダムウォークの価格での成績を上回った割合
    yearly_full: pd.DataFrame = None
    yearly_wf: pd.DataFrame = None
    dsr: dict = field(default_factory=dict)   # 検証区間の日ごとの損益にもとづく DSR
    pbo: dict = field(default_factory=dict)   # この戦略のパラメータ候補だけでの PBO


@dataclass
class Report:
    start: int
    reports: list[StrategyReport]
    baseline_random: pd.DataFrame    # 全戦略の平均頻度でのランダム売買
    baseline_curves: np.ndarray      # その累積損益の 5/50/95% 点
    k: int
    trials: list[dict] = field(default_factory=list)  # 試行の記録用（trials.record に渡す）
    grid_daily: pd.DataFrame = None  # 全パラメータ候補の日ごとの損益（列＝"戦略|パラメータ"）
    random_walk: pd.DataFrame = None  # ランダムウォークの価格での各戦略の成績（行＝試行）
    pbo_all: dict = field(default_factory=dict)  # 全候補（戦略 × パラメータ）での PBO
    n_trials: int = 0                # DSR に使った「試した数」
    sr_variance: float = math.nan    # 候補どうしのシャープレシオ（1日あたり）のばらつき
    best_config: str = ""            # 開発用データ全体でシャープレシオが最も高かった候補
    best_dsr: dict = field(default_factory=dict)


def evaluate_all(
    raw: pd.DataFrame, base: IndicatorParams, risk: RiskParams, rules: ExitRules, costs: Costs, pip: float,
    runs: int = 200, k: int = 5, train_mult: int = 2, holdout_start: pd.Timestamp | None = None,
    rw_paths: int = 50, n_trials: int | None = None, strategy_list: list[Strategy] | None = None,
    spread_mult=None,
) -> Report:
    """raw は開発用データ（最終テスト期間より前）。holdout_start を渡すと、それ以降の足が混ざっていたら止める。

    n_trials は DSR に使う「これまでに試した組み合わせの数」。省略するとこの評価で試した候補の数を使う
    （記録を使う場合は、試行を記録してから finalize() をもう一度呼ぶ）。
    """
    if holdout_start is not None and len(raw) and raw.index[-1] >= holdout_start:
        from .holdout import FinalTestAccessError
        raise FinalTestAccessError(f"最終テスト期間（{holdout_start} 以降）のデータがバックテストに渡されています")
    cache: dict = {}
    ind = indicators.add_all(raw, base)
    cache[base] = ind
    bars = Bars(ind, spread_mult)
    n = len(ind)
    days = overfit.business_days(ind.index[WARMUP], ind.index[-1])
    reports, rates, grid = [], [], {}
    sts = list(strategy_list) if strategy_list is not None else strategies.STRATEGIES
    rw = (random_walk_baseline(raw, base, risk, rules, costs, pip, rw_paths, seed=7, strategy_list=sts,
                               mult=spread_mult) if rw_paths else pd.DataFrame())
    for i, st in enumerate(sts):
        # その戦略の初期パラメータと、元のルールの決済（なければ共通の設定）
        p_st = st.params(base)
        if p_st not in cache:
            cache[p_st] = indicators.add_all(raw, p_st)
        ind_st = cache[p_st]
        bars_st = bars if p_st == base else Bars(ind_st, spread_mult)
        rs, ru = strategy_exits(st, risk, rules)
        xs = strategies.exit_signals(ind_st, st, p_st)
        sig = strategies.signals(ind_st, st, p_st).tolist()
        full = simulate(bars_st, sig, rs, ru, costs, pip, WARMUP, n, xs)
        fm = metrics(full["pips"])
        rate = signal_rate(sig, WARMUP, n)
        rates.append(rate)
        full_rand, _ = random_baseline(bars_st, [(WARMUP, n, rate)], rs, ru, costs, pip, runs, seed=i, exit_sig=xs)

        wf = walk_forward(raw, st, base, risk, rules, costs, pip, k, train_mult, cache=cache, mult=spread_mult)
        wf_windows = [(f.test[0], f.test[1], f.test_rate) for f in wf.folds]
        wf_rand, _ = random_baseline(bars_st, wf_windows, rs, ru, costs, pip, runs, seed=100 + i, exit_sig=xs)
        wf_beat = beat_ratio(wf.test_metrics["total"], wf_rand["total"])

        def exit_times(t: pd.DataFrame) -> pd.Index:
            return ind.index[t["exit_i"].to_numpy(dtype=int)]

        full_daily = overfit.daily_pnl(exit_times(full), full["pips"], days)
        wf_days = overfit.business_days(ind.index[wf.folds[0].test[0]], ind.index[-1])
        wt = wf.test_trades
        wf_daily = overfit.daily_pnl(exit_times(wt), wt["pips"], wf_days)

        # PBO 用：この戦略の全パラメータ候補を、開発用データ全体で動かした日ごとの損益
        for cand in candidates(raw, st, base, cache, spread_mult):
            t = simulate(cand.bars, cand.sig, rs, ru, costs, pip, WARMUP, n, cand.exit_sig)
            grid[f"{st.key}|{describe_params(st.key, cand.overrides)}"] = overfit.daily_pnl(exit_times(t), t["pips"], days)

        reports.append(StrategyReport(
            st, full, fm, full_rand, beat_ratio(fm["total"], full_rand["total"]), wf, wf_rand, wf_beat, ("", ""),
            full_daily=full_daily, wf_daily=wf_daily,
            full_sharpe=overfit.annualized(overfit.sharpe(full_daily)),
            wf_sharpe=overfit.annualized(overfit.sharpe(wf_daily)),
            rw_beat=beat_ratio(fm["total"], rw[st.key]) if len(rw) else math.nan,
            yearly_full=yearly(full, ind.index), yearly_wf=yearly(wt, ind.index),
        ))

    base_rand, curves = random_baseline(
        bars, [(WARMUP, n, float(np.mean(rates)))], risk, rules, costs, pip, runs, seed=999, curves=True
    )
    rep = Report(WARMUP, reports, base_rand, curves, k, trial_rows(reports, base, risk, rules),
                 grid_daily=pd.DataFrame(grid), random_walk=rw)
    rep.pbo_all = overfit.pbo_cscv(rep.grid_daily)
    for r in reports:
        cols = [c for c in rep.grid_daily.columns if c.startswith(r.strategy.key + "|")]
        r.pbo = overfit.pbo_cscv(rep.grid_daily[cols])
    return finalize(rep, n_trials if n_trials is not None else rep.grid_daily.shape[1])


def finalize(rep: Report, n_trials: int) -> Report:
    """試した数 n_trials を使って DSR を計算し、判定を決める（試行を記録した後に呼び直してよい）。"""
    srs = rep.grid_daily.apply(overfit.sharpe)
    rep.n_trials = int(n_trials)
    rep.sr_variance = float(srs.var(ddof=1)) if len(srs) > 1 else 0.0
    rep.best_config = str(srs.idxmax()) if len(srs) else ""
    if rep.best_config:
        rep.best_dsr = overfit.deflated_sharpe(rep.grid_daily[rep.best_config].to_numpy(), rep.n_trials, rep.sr_variance)
    for r in rep.reports:
        r.dsr = overfit.deflated_sharpe(r.wf_daily.to_numpy(), rep.n_trials, rep.sr_variance)
        r.verdict = assess(r.wf, r.wf_beat, rep.k, r.dsr.get("dsr", math.nan), r.pbo.get("pbo", math.nan),
                           r.rw_beat, r.yearly_wf)
    return rep


def planned_trials(base: IndicatorParams, risk: RiskParams, rules: ExitRules,
                   strategy_list: list[Strategy] | None = None) -> list[dict]:
    """evaluate_all が試す設定（各戦略のパラメータ候補。初期パラメータも候補に含まれる）。"""
    from .trials import strategy_config

    sts = list(strategy_list) if strategy_list is not None else strategies.STRATEGIES
    return [{"family": "strategy", "name": st.key,
             "config": strategy_config(st.key, replace(base, **ov), *strategy_exits(st, risk, rules), spec=st.spec)}
            for st in sts for ov in st.grid()]


def trial_rows(reports: list[StrategyReport], base: IndicatorParams, risk: RiskParams, rules: ExitRules) -> list[dict]:
    """全期間（初期パラメータ）と、ウォークフォワードで試した全候補を、試行として記録する形にする。
    戦略ごとの初期パラメータと決済のルールで記録する（最初からある5つの戦略は共通の設定なので、以前と同じ形）。"""
    from .trials import strategy_config

    rows = []
    for r in reports:
        st = r.strategy
        rs, ru = strategy_exits(st, risk, rules)
        rows.append({"family": "strategy", "name": st.key, "purpose": "full_period",
                     "config": strategy_config(st.key, st.params(base), rs, ru, spec=st.spec),
                     "metrics": {**r.full_metrics, "beat_random": r.full_beat}})
        for j, ov in enumerate(st.grid()):
            rows.append({"family": "strategy", "name": st.key, "purpose": "wf_candidate",
                         "config": strategy_config(st.key, replace(base, **ov), rs, ru, spec=st.spec),
                         "metrics": {"train_total_by_fold": [f.candidate_metrics[j]["total"] for f in r.wf.folds],
                                     "selected_folds": [i for i, f in enumerate(r.wf.folds) if f.overrides == ov]}})
    return rows


VERDICT_VERSION = "v2-rw-yearly"   # 判定の基準を変えたら上げる（保存済みの判定を計算し直すため）
WORST_YEAR_SHARE = 0.5             # 1年の損失が、検証区間の利益合計のこの割合を超えたら「極端に悪い年」（暫定。ユーザーと相談）


def worst_year(yearly_df: pd.DataFrame | None, total: float) -> tuple[int, float] | None:
    """極端に悪い年（1年の損失が利益合計の WORST_YEAR_SHARE を超える）があれば (年, 損益)。"""
    if yearly_df is None or len(yearly_df) == 0 or total <= 0:
        return None
    w = yearly_df.loc[yearly_df["total"].idxmin()]
    return (int(w["year"]), float(w["total"])) if -w["total"] > WORST_YEAR_SHARE * total else None


def assess(wf: WalkForwardResult, beat: float, k: int, dsr: float = math.nan, pbo: float = math.nan,
           rw_beat: float = math.nan, yearly_df: pd.DataFrame | None = None) -> tuple[str, str]:
    """ウォークフォワードの検証区間の成績から、優位性がありそうかを機械的に判定する（要件定義書 6-3 の合格基準）。

    「有望」は、取引100回以上・ランダム売買の95%以上を上回る・ランダムウォークの価格での成績の95%以上を上回る・
    黒字の区間が6割以上・極端に悪い年がない・DSR（試した数で割り引いたシャープレシオ）95%以上・
    PBO（パラメータ選びの過学習の確率）50%未満、をすべて満たすときだけ。
    """
    m = wf.test_metrics
    if m["n"] < 20:
        return "判断材料不足", f"検証区間での取引が {m['n']} 回しかなく、偶然との区別がつきません。"
    if m["total"] <= 0:
        return "優位性なし", "コストを含めると、まだ見ていない期間でマイナスでした。"
    if beat < 0.95:
        return "偶然の範囲", f"検証区間ではプラスでしたが、ランダム売買の {beat:.0%} しか上回っておらず、偶然でも出る程度の成績です。"
    if not math.isnan(rw_beat) and rw_beat < 0.95:
        return "偶然の範囲", (f"ランダム売買には勝ちましたが、でたらめに動く価格（ランダムウォーク）で同じルールを動かした成績の "
                          f"{rw_beat:.0%} しか上回っておらず、このルールが値動きの特徴をとらえているとは言えません。")
    if wf.positive_folds < math.ceil(k * 0.6):
        return "不安定", f"合計ではランダム売買を上回りましたが、黒字の区間は {wf.positive_folds}/{k} だけで、一部の期間に頼った成績です。"
    bad = worst_year(yearly_df, m["total"])
    if bad:
        return "不安定", (f"合計ではプラスですが、{bad[0]}年に {bad[1]:+,.0f} pips と、利益合計（{m['total']:+,.0f} pips）の"
                        f"{WORST_YEAR_SHARE:.0%}を超える損失を出した年があります。")
    base = f"検証区間でプラス、ランダム売買の {beat:.0%} を上回り、{wf.positive_folds}/{k} 区間で黒字でした。"
    if m["n"] < MIN_TRADES_RELIABLE:
        return "参考値", base + f"ただし取引が {m['n']} 回と少ない（{MIN_TRADES_RELIABLE}回未満）ため、参考値として扱います。"
    if math.isnan(dsr) or dsr < 0.95:
        d = "計算できません" if math.isnan(dsr) else f"{dsr:.0%}"
        return "偶然の可能性", base + (f"ただし、試した組み合わせの数で割り引くと、偶然ではない確率（DSR）は {d} で、"
                                   "95%に届きません。たくさん試した中から良いものを選んだだけの可能性があります。")
    if not math.isnan(pbo) and pbo >= 0.5:
        return "過学習の疑い", base + (f"ただし、パラメータ選びが過学習している確率（PBO）が {pbo:.0%} と高く、"
                                   "過去に一番良かったパラメータが、この先も良いとは言えません。")
    return "有望（要継続監視）", base + (
        f"試した数で割り引いても偶然ではない確率（DSR）は {dsr:.0%}"
        + ("" if math.isnan(pbo) else f"、パラメータ選びの過学習の確率（PBO）は {pbo:.0%}")
        + "でした。それでも将来の利益は保証されないため、本番の記録で確かめ続けます。"
    )
