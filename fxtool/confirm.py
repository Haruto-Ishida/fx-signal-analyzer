"""SMA20/100 クロス候補の「確認の検証」（2026-10-07 のユーザーの決定。事前登録：preregistration/sma20_100_confirm.json）。

この候補は、Yahoo Finance の 1時間足（2023-12-19〜2026-03-16 の開発用データ）の全通りの評価で選んだ。
それより前の期間（2013-01-01〜2023-12-18）は、選ぶときに一度も見ていないデータなので、ここで **1回だけ** 試す。

決まりごと（事前登録の内容。検証が終わるまで変更しない）
- ルール・損切り・利確・時間切れ・コスト・期間・合格基準は、データを見る前に固定し、git にコミットした
  （preregistration/sma20_100_confirm.json）。結果を見てから変えない。
- 固定したルールの計算が変わっていないことを、合成データでの「指紋」（fingerprint）で毎回確かめる（テスト）。
- データ元のずれ（Dukascopy と Yahoo、Dukascopy と HistData）を先に確かめ、基準を超えたら検証しないで止める。
- 判定は2通り：① 累計の試行数で割り引いた判定（いつもの厳しい基準）② 事前に1つに決めたルールを1回だけ試した場合の判定。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import costmodel, ga, indicators, overfit
from .config import ROOT, IndicatorParams, RiskParams

PREREG_PATH = ROOT / "preregistration" / "sma20_100_confirm.json"


# ---------------- 事前登録 ----------------

def load_prereg(path: Path | None = None) -> dict:
    return json.loads((path or PREREG_PATH).read_text(encoding="utf-8"))


def genome(prereg: dict) -> tuple[int, int, int]:
    g = prereg["rule"]["genes"]
    fs = [ga.FILTER_IDS.index(f) for f in g["filters"]]
    return ga.canonical([ga.TRIGGER_IDS.index(g["trigger"]), *(fs + [0, 0])[:2]])


def frozen(prereg: dict):
    """事前登録の内容から、戦略・指標のパラメータ・損切りと利確・決済ルール・コストを作る。"""
    r = prereg["rule"]
    p = IndicatorParams(**r["indicator_params"])
    risk = RiskParams(sl_atr=r["exits"]["sl_atr"], tp_atr=r["exits"]["tp_atr"])
    rules = bt.ExitRules(opposite_exit=r["exits"]["opposite_exit"], max_bars=r["exits"]["max_bars"])
    costs = bt.Costs(r["costs"]["spread_pips"], r["costs"]["slippage_pips"])
    return ga.to_strategy(genome(prereg)), p, risk, rules, costs


def fingerprint(prereg: dict) -> str:
    """固定したルールの計算の指紋：決まった乱数の合成データでシグナルと取引を計算し、その結果のハッシュ。
    ルールや、指標・売買シミュレーションの計算が変わると値が変わる（テストで、登録した値と同じことを確かめる）。"""
    st, p, risk, rules, costs = frozen(prereg)
    rng = np.random.default_rng(20261007)
    n = 4000
    close = 150 + np.cumsum(rng.normal(0, 0.08, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.05, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.05, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                      index=pd.date_range("2020-01-06", periods=n, freq="h"))
    ind = indicators.add_all(df, p)
    sig = st.func(ind, p).where(ind["atr"].notna(), 0).astype(int).tolist()
    trades = bt.simulate_raw(bt.Bars(ind), sig, risk, rules, costs, 0.01, bt.WARMUP, n)
    payload = json.dumps([[t[0], t[1], t[3], t[8], round(t[9], 6)] for t in trades], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------- データ元のずれの確認 ----------------

def compare_sources(a: pd.DataFrame, b: pd.DataFrame, pip: float) -> dict:
    """2つのデータ元の1時間足（UTC・足の開始時刻）を比べる。共通の時刻の割合と、終値の差（pips）。"""
    common = a.index.intersection(b.index)
    base = min(len(a), len(b))
    diff = (a.loc[common, "close"] - b.loc[common, "close"]).abs() / pip
    return {"n_a": len(a), "n_b": len(b), "n_common": len(common),
            "match_ratio": len(common) / base if base else 0.0,
            "median_close_diff_pips": float(diff.median()) if len(diff) else math.nan,
            "p95_close_diff_pips": float(diff.quantile(0.95)) if len(diff) else math.nan}


def consistency_ok(res: dict, limits: dict) -> tuple[bool, str]:
    if res["match_ratio"] < limits["min_match_ratio"]:
        return False, f"共通の時刻が {res['match_ratio']:.1%}（{limits['min_match_ratio']:.0%} 以上が必要）"
    if not res["median_close_diff_pips"] <= limits["max_median_close_diff_pips"]:
        return False, f"終値の差の中央値が {res['median_close_diff_pips']:.2f} pips（{limits['max_median_close_diff_pips']} pips 以下が必要）"
    return True, "OK"


# ---------------- 確認の検証 ----------------

@dataclass
class Confirmation:
    n_trades: int
    total_pips: float
    win_rate: float
    sr_daily: float
    sr_annual: float
    random_beat: float
    rw_beat: float
    yearly: pd.DataFrame
    positive_year_ratio: float
    worst_year: tuple[int, float] | None
    psr_single: float              # ② 1回だけ試した場合：シャープレシオが 0 より大きい確率
    dsr_cumulative: float          # ① 累計の試行数で割り引いた DSR
    n_trials_cumulative: int
    verdict_single: tuple[str, list[str]] = ("", [])
    verdict_cumulative: tuple[str, list[str]] = ("", [])
    segments: list[float] = field(default_factory=list)   # 期間を5つに分けた、それぞれの損益（pips）


def run(ohlc: pd.DataFrame, prereg: dict, n_trials_cumulative: int, runs: int = 500, rw_paths: int = 100) -> Confirmation:
    """ohlc：1時間足（UTC・足の開始時刻・中値）。確認の期間の前に、指標の準備のための足（WARMUP 本以上）を含めること。
    確認の期間の外の足では売買しない。"""
    st, p, risk, rules, costs = frozen(prereg)
    per = prereg["test_period"]
    t0, t1 = pd.Timestamp(per["from"]), pd.Timestamp(per["to"])
    start = int(ohlc.index.searchsorted(t0))
    if start < bt.WARMUP:
        raise ValueError(f"確認の期間の前に、準備のための足が {bt.WARMUP} 本以上必要です（いまは {start} 本）")
    raw = ohlc[(ohlc.index < t1 + pd.Timedelta(hours=1))].iloc[start - bt.WARMUP:]
    raw = raw[["open", "high", "low", "close"]]
    mult = costmodel.spread_multipliers(raw.index, "1h") if prereg["rule"]["costs"]["time_varying"] else None
    ind = indicators.add_all(raw, p)
    bars = bt.Bars(ind, mult)
    n = len(ind)
    sig = st.func(ind, p).where(ind["atr"].notna(), 0).astype(int).tolist()
    trades = bt.simulate(bars, sig, risk, rules, costs, 0.01, bt.WARMUP, n)
    m = bt.metrics(trades["pips"])
    days = overfit.business_days(ind.index[bt.WARMUP], ind.index[-1])
    daily = overfit.daily_pnl(ind.index[trades["exit_i"].to_numpy(dtype=int)], trades["pips"], days).to_numpy()
    sr = overfit.sharpe(daily)
    rate = bt.signal_rate(sig, bt.WARMUP, n)
    rand, _ = bt.random_baseline(bars, [(bt.WARMUP, n, rate)], risk, rules, costs, 0.01, runs, seed=1)
    rw = bt.random_walk_baseline(raw, p, risk, rules, costs, 0.01, rw_paths, seed=2, strategy_list=[st], mult=mult)
    yearly = bt.yearly(trades, ind.index)
    pos_years = float((yearly["total"] > 0).mean()) if len(yearly) else 0.0
    worst = bt.worst_year(yearly, m["total"])
    psr = overfit.probabilistic_sharpe(sr, 0.0, daily)
    dsr = overfit.deflated_sharpe(daily, n_trials_cumulative, prereg["cumulative"]["sr_variance"])["dsr"]
    segs = [float(s.sum()) for s in np.array_split(daily, 5)]
    c = Confirmation(m["n"], m["total"], m["win_rate"], sr, overfit.annualized(sr), bt.beat_ratio(m["total"], rand["total"]),
                     bt.beat_ratio(m["total"], rw[st.key]), yearly, pos_years, worst, psr, dsr, n_trials_cumulative,
                     segments=segs)
    c.verdict_single = judge(c, prereg, c.psr_single)
    c.verdict_cumulative = judge(c, prereg, c.dsr_cumulative)
    return c


def judge(c: Confirmation, prereg: dict, prob: float) -> tuple[str, list[str]]:
    """事前登録の合格基準。prob は ② なら PSR（1回だけ試した場合）、① なら累計の試行数での DSR。"""
    k = prereg["pass_criteria"]
    checks = [
        (c.n_trades >= k["min_trades"], f"取引 {c.n_trades} 回（{k['min_trades']} 回以上）"),
        (c.total_pips > 0, f"コスト後の損益 {c.total_pips:+,.0f} pips（プラス）"),
        (c.random_beat >= k["min_random_beat"], f"ランダム売買を上回った割合 {c.random_beat:.0%}（{k['min_random_beat']:.0%} 以上）"),
        (c.rw_beat >= k["min_rw_beat"], f"ランダムウォークを上回った割合 {c.rw_beat:.0%}（{k['min_rw_beat']:.0%} 以上）"),
        (c.positive_year_ratio >= k["min_positive_year_ratio"],
         f"黒字の年の割合 {c.positive_year_ratio:.0%}（{k['min_positive_year_ratio']:.0%} 以上）"),
        (c.worst_year is None, "極端に悪い年なし" if c.worst_year is None else
         f"{c.worst_year[0]}年に {c.worst_year[1]:+,.0f} pips（利益合計の50%を超える損失）"),
        (not math.isnan(prob) and prob >= k["min_probability"], f"偶然ではない確率 {prob:.0%}（{k['min_probability']:.0%} 以上）"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return ("合格" if not failed else "不合格"), [("✅ " if ok else "❌ ") + msg for ok, msg in checks]


def trial_row(prereg: dict) -> dict:
    """この確認の試行（1つ）。設定は全通りの評価のときと同じ。データ元と期間が違うので、別の試行として数える。"""
    from .trials import strategy_config
    st, p, risk, rules, costs = frozen(prereg)
    return {"family": "strategy", "name": "ga", "purpose": "confirmation",
            "config": strategy_config(st.key, p, risk, rules, spec=st.spec)}
