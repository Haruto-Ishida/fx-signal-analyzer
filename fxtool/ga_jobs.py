"""GA の探索の実行と保存（画面のボタン、週末の自動の再探索）。

- 使うのは開発用データだけ（最終テスト期間は使わない）。
- 評価した組み合わせはすべて試行として記録する。
- 上位の候補は ga_candidates に保存するだけで、シグナルには使わない（採用は検証とユーザーの承認のあと）。
"""
from __future__ import annotations

import json
import random
import sqlite3

import pandas as pd

from . import costmodel, data, ga, jobs, trials
from . import settings as user_settings
from .tracking import JST, ts_str

TOP_SAVE = 20
WEEKLY_TOP = 10


def run(con: sqlite3.Connection, pair: str, tf: str, kind: str = "full", seed: int | None = None,
        budget: int | None = None, origin: str = "dashboard", now: pd.Timestamp | None = None) -> int:
    """探索して保存し、ga_runs の id を返す。budget は設定ファイルの上限を超えられない。"""
    s = user_settings.load()
    cap = s.trial_budget.ga_full_search if kind == "full" else s.trial_budget.ga_weekly_search
    budget = cap if budget is None else min(int(budget), cap)
    seed = random.SystemRandom().randrange(2**31) if seed is None else int(seed)
    now = data.now_utc() if now is None else now
    dev = data.load_dev(con, pair, tf)[["open", "high", "low", "close"]]
    b = data.boundary(con, pair, tf)
    costs, pip = jobs.default_costs(pair), jobs.pip_size(jobs.PAIRS[pair])
    mult = costmodel.spread_multipliers(dev.index, tf, s.costs, s.event_filter)
    rules = jobs.exit_rules(tf)
    if kind == "full":
        res = ga.search(dev, jobs.RISK, rules, costs, pip, budget, seed, holdout_start=b.start, mult=mult, base_p=jobs.P)
    elif kind == "neighborhood":
        prev = previous_top(con, pair, tf, WEEKLY_TOP)
        if not prev:
            raise ValueError("前回の探索の結果がないため、周辺の再探索はできません（先に本格的な探索をしてください）")
        res = ga.neighborhood_search(dev, prev, jobs.RISK, rules, costs, pip, budget, seed, holdout_start=b.start,
                                     mult=mult, base_p=jobs.P)
    else:
        raise ValueError(kind)
    trials.record(con, res.trials, pair=pair, tf=tf, origin=origin, now=now, data_from=dev.index[0],
                  data_to=dev.index[-1], n_bars=len(dev), note=f"GA {kind} seed={seed} budget={budget}")
    cur = con.execute(
        """INSERT INTO ga_runs (pair, tf, kind, seed, budget, n_evaluated, started_at, data_from, data_to, origin)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pair, tf, kind, seed, budget, len(res.evaluated), ts_str(now), ts_str(dev.index[0]), ts_str(dev.index[-1]), origin))
    run_id = cur.lastrowid
    con.executemany(
        """INSERT INTO ga_candidates (run_id, rank, key, genes, rule, fitness, trades, total_pips, segment_sharpe)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(run_id, i + 1, r.key, json.dumps(ga.genes_of(r.genome), ensure_ascii=False), r.rule, float(r.fitness),
          int(r.trades), float(r.total_pips), json.dumps([round(x, 4) for x in r.segment_sharpe]))
         for i, r in enumerate(res.evaluated.head(TOP_SAVE).itertuples())])
    con.commit()
    return run_id


def genome_from_genes(genes: dict) -> tuple[int, int, int]:
    fs = [ga.FILTER_IDS.index(f) for f in genes["filters"]]
    return ga.canonical([ga.TRIGGER_IDS.index(genes["trigger"]), *(fs + [0, 0])[:2]])


def previous_top(con, pair, tf, k: int) -> list[tuple[int, int, int]]:
    row = con.execute("SELECT id FROM ga_runs WHERE pair=? AND tf=? ORDER BY id DESC LIMIT 1", (pair, tf)).fetchone()
    if not row:
        return []
    genes = [json.loads(g) for (g,) in con.execute(
        "SELECT genes FROM ga_candidates WHERE run_id=? ORDER BY rank LIMIT ?", (row[0], k))]
    out = []
    for g in genes:
        try:
            out.append(genome_from_genes(g))
        except ValueError:  # 部品のカタログが変わって、もう作れない組み合わせ
            continue
    return out


def load_runs(con, pair, tf) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM ga_runs WHERE pair=? AND tf=? ORDER BY id DESC", con, params=(pair, tf))


def load_candidates(con, run_id: int) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM ga_candidates WHERE run_id=? ORDER BY rank", con, params=(run_id,))


def weekly_due(con, pair, tf, now: pd.Timestamp, s: user_settings.Settings | None = None) -> bool:
    """週末（日本時間の土日）で、設定で有効になっていて、本格的な探索の結果があり、6日以上再探索していないとき。"""
    s = s or user_settings.load()
    if not s.trial_budget.ga_weekly_enabled or s.trial_budget.ga_weekly_search <= 0:
        return False
    if (now + JST).weekday() < 5:
        return False
    runs = load_runs(con, pair, tf)
    if runs.empty:
        return False
    last = pd.Timestamp(runs["started_at"].iloc[0])
    return now - last >= pd.Timedelta(days=6)


def projected_trials_per_year(s: user_settings.Settings | None = None, full_runs_per_year: int = 1) -> int:
    """1年分の試行数の見込み（1つの通貨ペア・時間足あたり、最大）。"""
    s = s or user_settings.load()
    weekly = 52 * s.trial_budget.ga_weekly_search if s.trial_budget.ga_weekly_enabled else 0
    return full_runs_per_year * s.trial_budget.ga_full_search + weekly
