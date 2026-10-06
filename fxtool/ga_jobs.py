"""GA の探索の実行と保存（画面のボタン、週末の自動の再探索）。

- 使うのは開発用データだけ（最終テスト期間は使わない）。
- 評価した組み合わせはすべて試行として記録する。
- 上位の候補は ga_candidates に保存するだけで、シグナルには使わない（採用は検証とユーザーの承認のあと）。
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest as bt
from . import costmodel, data, ga, jobs, trials
from . import settings as user_settings
from .tracking import JST, ts_str

TOP_SAVE = 20
WEEKLY_TOP = 10


def run(con: sqlite3.Connection, pair: str, tf: str, kind: str = "full", seed: int | None = None,
        budget: int | None = None, origin: str = "dashboard", now: pd.Timestamp | None = None) -> int:
    """探索して保存し、ga_runs の id を返す。budget は設定ファイルの上限を超えられない。"""
    s = user_settings.load()
    if kind == "exhaustive":
        budget, seed = search_space_size(), 0   # 全通り（乱数は使わない）
    else:
        cap = s.trial_budget.ga_full_search if kind == "full" else s.trial_budget.ga_weekly_search
        budget = cap if budget is None else min(int(budget), cap)
        seed = random.SystemRandom().randrange(2**31) if seed is None else int(seed)
    now = data.now_utc() if now is None else now
    dev = data.load_dev(con, pair, tf)[["open", "high", "low", "close"]]
    b = data.boundary(con, pair, tf)
    costs, pip = jobs.default_costs(pair), jobs.pip_size(jobs.PAIRS[pair])
    mult = costmodel.spread_multipliers(dev.index, tf, s.costs, s.event_filter)
    rules = jobs.exit_rules(tf)
    if kind == "exhaustive":
        res = ga.exhaustive_search(dev, jobs.RISK, rules, costs, pip, holdout_start=b.start, mult=mult, base_p=jobs.P)
    elif kind == "full":
        res = ga.search(dev, jobs.RISK, rules, costs, pip, budget, seed, holdout_start=b.start, mult=mult, base_p=jobs.P)
    elif kind == "neighborhood":
        prev = previous_top(con, pair, tf, WEEKLY_TOP)
        if not prev:
            raise ValueError("前回の探索の結果がないため、周辺の再探索はできません（先に本格的な探索をしてください）")
        res = ga.neighborhood_search(dev, prev, jobs.RISK, rules, costs, pip, budget, seed, holdout_start=b.start,
                                     mult=mult, base_p=jobs.P, skip=evaluated_genomes(con, pair, tf))
    else:
        raise ValueError(kind)
    if res.trials:
        trials.record(con, res.trials, pair=pair, tf=tf, origin=origin, now=now, data_from=dev.index[0],
                      data_to=dev.index[-1], n_bars=len(dev), note=f"GA {kind} seed={seed} budget={budget}")
    cur = con.execute(
        """INSERT INTO ga_runs (pair, tf, kind, seed, budget, n_evaluated, started_at, data_from, data_to, origin, trace)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pair, tf, kind, seed, budget, len(res.evaluated), ts_str(now), ts_str(dev.index[0]), ts_str(dev.index[-1]), origin,
         json.dumps([round(x, 5) for x in res.trace])))
    run_id = cur.lastrowid
    if res.evaluated.empty:
        con.commit()
        return run_id
    con.executemany(
        """INSERT INTO ga_candidates (run_id, rank, key, genes, rule, fitness, trades, total_pips, segment_sharpe)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(run_id, i + 1, r.key, json.dumps(ga.genes_of(r.genome), ensure_ascii=False), r.rule, float(r.fitness),
          int(r.trades), float(r.total_pips), json.dumps([round(x, 4) for x in r.segment_sharpe]))
         for i, r in enumerate(res.evaluated.head(TOP_SAVE).itertuples())])
    con.commit()
    return run_id


VALIDATE_TOP = 5


def validate(con: sqlite3.Connection, pair: str, tf: str, run_id: int, top_n: int = VALIDATE_TOP,
             origin: str = "dashboard", now: pd.Timestamp | None = None, runs: int = 200, rw_paths: int = 20) -> int:
    """探索の上位 top_n 個を、ほかの戦略と同じ検証（ウォークフォワード・ランダム売買・ランダムウォーク・PBO・DSR・年別）に
    かけ、結果を ga_candidates に保存する。開発用データだけを使う。探索のときと同じ設定なので、試行数は増えない。
    DSR は、その時点の累計の試行数（GA の探索で試した分を含む）で割り引く。"""
    now = data.now_utc() if now is None else now
    s = user_settings.load()
    cands = load_candidates(con, run_id).head(top_n)
    sts = [ga.to_strategy(genome_from_genes(json.loads(g))) for g in cands["genes"]]
    dev = data.load_dev(con, pair, tf)[["open", "high", "low", "close"]]
    b = data.boundary(con, pair, tf)
    costs, pip = jobs.default_costs(pair), jobs.pip_size(jobs.PAIRS[pair])
    mult = costmodel.spread_multipliers(dev.index, tf, s.costs, s.event_filter)
    rules = jobs.exit_rules(tf)
    planned = bt.planned_trials(jobs.P, jobs.RISK, rules, sts)
    n_trials = trials.count_with(con, pair, tf, planned, dev.index[0], dev.index[-1])
    rep = bt.evaluate_all(dev, jobs.P, jobs.RISK, rules, costs, pip, runs=runs, holdout_start=b.start,
                          rw_paths=rw_paths, n_trials=n_trials, strategy_list=sts, spread_mult=mult)
    trials.record(con, rep.trials, pair=pair, tf=tf, origin=origin, now=now, data_from=dev.index[0],
                  data_to=dev.index[-1], n_bars=len(dev), note=f"GA の候補の検証（run {run_id}）")
    for rank, r in zip(cands["rank"], rep.reports):
        y = {int(row.year): round(float(row.total), 1) for row in r.yearly_wf.itertuples()} if len(r.yearly_wf) else {}
        con.execute(
            """UPDATE ga_candidates SET verdict=?, reason=?, wf_total=?, wf_n=?, wf_beat=?, rw_beat=?, dsr=?, pbo=?,
               n_trials=?, yearly=?, validated_at=? WHERE run_id=? AND rank=?""",
            (r.verdict[0], r.verdict[1], float(r.wf.test_metrics["total"]), int(r.wf.test_metrics["n"]), float(r.wf_beat),
             jobs._num(r.rw_beat), jobs._num(r.dsr.get("dsr")), jobs._num(r.pbo.get("pbo")), rep.n_trials,
             json.dumps(y), ts_str(now), run_id, int(rank)))
    con.commit()
    return rep.n_trials


def genome_from_genes(genes: dict) -> tuple[int, int, int]:
    fs = [ga.FILTER_IDS.index(f) for f in genes["filters"]]
    return ga.canonical([ga.TRIGGER_IDS.index(genes["trigger"]), *(fs + [0, 0])[:2]])


def previous_top(con, pair, tf, k: int) -> list[tuple[int, int, int]]:
    """これまでのすべての探索の中で、適応度が高い順の k 個（同じ組み合わせは1つ）。
    開発用データは固定なので、別の回の適応度もそのまま比べられる。"""
    rows = con.execute(
        """SELECT c.genes FROM ga_candidates c JOIN ga_runs r ON r.id = c.run_id
           WHERE r.pair=? AND r.tf=? ORDER BY c.fitness DESC""", (pair, tf)).fetchall()
    out: list[tuple[int, int, int]] = []
    for (g,) in rows:
        try:
            gen = genome_from_genes(json.loads(g))
        except ValueError:  # 部品のカタログが変わって、もう作れない組み合わせ
            continue
        if gen not in out:
            out.append(gen)
        if len(out) >= k:
            break
    return out


def evaluated_genomes(con, pair, tf) -> set[tuple[int, int, int]]:
    """これまでに評価した（試行として記録した）GA の組み合わせ。"""
    out = set()
    for (cfg,) in con.execute("SELECT config FROM trials WHERE pair=? AND tf=? AND config LIKE '%ga_genes%'", (pair, tf)):
        genes = json.loads(cfg).get("spec", {}).get("ga_genes")
        if genes and genes.get("version") == ga.GENE_VERSION:
            try:
                out.add(genome_from_genes(genes))
            except ValueError:
                continue
    return out


def search_space_size() -> int:
    return len(ga.all_genomes())


def review_projection(n_now: int, years: int, done_now: bool, per_review_other: int = 40) -> dict:
    """「全通りの評価 ＋ 年1回の見直し（開発用データを新しくして全通りを評価し直す）」のときの、years 年後の累計試行数と
    合格の基準の厳しさ。データの期間が変わると別の試行として数えるので、見直しのたびに全通り分（＋最初からある戦略の候補・
    機械学習の約 per_review_other 通り）増える。done_now は、いまの開発用データで全通りの評価をすでにしたか。"""
    from . import overfit
    space = search_space_size()
    n = n_now + (0 if done_now else space) + years * (space + per_review_other)
    base = overfit.expected_max_sharpe(max(n_now, 2), 1.0)
    return {"years": years, "trials": n, "strict": overfit.expected_max_sharpe(n, 1.0) / base}


@dataclass
class Convergence:
    best_so_far: list[float]   # n 通り目までの最良の適応度
    best_at: int               # 最終的な最良が初めて見つかった順番（1始まり）
    near_at: int               # 最終的な最良との差が tol 以内に初めて入った順番
    last_gain_at: int          # 最良が最後に更新された順番（＝best_at）
    n: int


def convergence(trace: list[float], tol: float = 0.05) -> Convergence | None:
    """評価した順の適応度から、何通り目で最良が見つかったか・それ以降は改善しなかったかを調べる。"""
    x = np.array([v for v in trace], dtype=float)
    if len(x) == 0:
        return None
    x[~np.isfinite(x)] = -10.0
    bsf = np.maximum.accumulate(x)
    best = bsf[-1]
    best_at = int(np.argmax(bsf >= best)) + 1
    near_at = int(np.argmax(bsf >= best - tol * max(abs(best), 1e-9))) + 1
    return Convergence(bsf.tolist(), best_at, near_at, best_at, len(x))


def suggest_budget(conv: Convergence, cap: int) -> int:
    """再探索の1回の数の目安：最良が見つかるまでにかかった数の半分（周辺だけを探すので、ゼロからの探索より少なくて済む）を、
    10通り単位で切り上げ、30〜cap の範囲にする。あくまで目安（ユーザーと相談して決める）。"""
    n = math.ceil(conv.best_at / 2 / 10) * 10
    return int(min(max(n, 30), cap))


def projection(n_now: int, s: user_settings.Settings, years: float, full_runs: int = 1) -> dict:
    """years 年後の累計試行数の見込み（1つの通貨ペア・時間足あたり）と、合格の基準の厳しさ。
    - 単純な見込み：本格的な探索 full_runs 回 ＋ 再探索（間隔ごとに1回×上限の数）
    - 実際の上限：GA の組み合わせは全部で search_space_size() 通りしかなく、同じ組み合わせは二度数えないので、
      GA の分はそれを超えない
    - 基準の厳しさ：偶然でも出るシャープレシオの水準（SR*）が、いまの試行数のときの何倍になるか"""
    from . import overfit
    tb = s.trial_budget
    runs = math.ceil(years * 52 / tb.ga_interval_weeks) if tb.ga_weekly_enabled else 0
    naive_ga = full_runs * tb.ga_full_search + runs * tb.ga_weekly_search
    capped_ga = min(naive_ga, search_space_size())
    out = {"years": years, "re_search_runs": runs, "naive": n_now + naive_ga, "capped": n_now + capped_ga}
    base = overfit.expected_max_sharpe(max(n_now, 2), 1.0)
    out["strict_naive"] = overfit.expected_max_sharpe(out["naive"], 1.0) / base
    out["strict_capped"] = overfit.expected_max_sharpe(out["capped"], 1.0) / base
    return out


def load_runs(con, pair, tf) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM ga_runs WHERE pair=? AND tf=? ORDER BY id DESC", con, params=(pair, tf))


def load_candidates(con, run_id: int) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM ga_candidates WHERE run_id=? ORDER BY rank", con, params=(run_id,))


def weekly_due(con, pair, tf, now: pd.Timestamp, s: user_settings.Settings | None = None) -> bool:
    """定期的な再探索をするか：設定で有効・日本時間の土曜・本格的な探索の結果がある・前回の探索（本格的・再探索とも）から
    間隔（初期値3週間）がたっている、をすべて満たすとき。土曜にしか動かないので、「間隔−1日」以上たっていればよい。"""
    s = s or user_settings.load()
    tb = s.trial_budget
    if not tb.ga_weekly_enabled or tb.ga_weekly_search <= 0:
        return False
    if (now + JST).weekday() != 5:
        return False
    runs = load_runs(con, pair, tf)
    if runs.empty or not (runs["kind"] == "full").any():
        return False
    last = pd.Timestamp(runs["started_at"].iloc[0])
    return now - last >= pd.Timedelta(days=7 * tb.ga_interval_weeks - 1)


def projected_trials_per_year(s: user_settings.Settings | None = None, full_runs_per_year: int = 1) -> int:
    """1年分の GA の試行数の見込み（1つの通貨ペア・時間足あたり、単純な最大。実際は search_space_size() が上限）。"""
    s = s or user_settings.load()
    tb = s.trial_budget
    runs = math.ceil(52 / tb.ga_interval_weeks) if tb.ga_weekly_enabled else 0
    return full_runs_per_year * tb.ga_full_search + runs * tb.ga_weekly_search
