"""Step 5：遺伝的アルゴリズム（GA）のテスト（合成データだけ。実データでの探索はしない）。"""
import json
import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import ga, indicators, strategies, trials
from fxtool.config import IndicatorParams, RiskParams
from fxtool.holdout import FinalTestAccessError

P = IndicatorParams()
RULES = bt.ExitRules(max_bars=20)
COSTS = bt.Costs(0.3, 0.2)


@pytest.fixture(scope="module")
def daily():
    df = make_path_ohlc(1800, sd=0.6, seed=21)
    df.index = pd.bdate_range("2012-01-01", periods=len(df))
    return df


def all_genomes():
    return sorted({ga.canonical([t, a, b]) for t in range(len(ga.TRIGGERS))
                   for a in range(len(ga.FILTERS)) for b in range(len(ga.FILTERS))})


def test_canonical_form_and_size_of_the_search_space():
    assert ga.canonical([3, 5, 2]) == ga.canonical([3, 2, 5]) == (3, 2, 5)
    assert ga.canonical([3, 4, 4]) == (3, 4, 0) and ga.canonical([3, 0, 4]) == (3, 4, 0)
    assert ga.complexity([3, 0, 0]) == 1 and ga.complexity([3, 2, 5]) == 3   # 最大3条件
    assert ga.key_of([3, 5, 2]) == ga.key_of([3, 2, 5])
    n = len(all_genomes())
    nt, nf = len(ga.TRIGGERS), len(ga.FILTERS) - 1
    assert n == nt * (1 + nf + nf * (nf - 1) // 2)


@pytest.mark.parametrize("t", range(len(ga.TRIGGERS)), ids=lambda t: ga.TRIGGER_IDS[t])
def test_triggers_have_no_lookahead(daily, t):
    full = indicators.add_all(daily, P)
    tampered = daily.copy()
    tampered.iloc[1300:] *= 1.2
    part = indicators.add_all(tampered, P)
    g = (t, 0, 0)
    pd.testing.assert_series_equal(ga.make_func(g)(full, P).iloc[:1300], ga.make_func(g)(part, P).iloc[:1300])


@pytest.mark.parametrize("f", range(1, len(ga.FILTERS)), ids=lambda f: ga.FILTER_IDS[f])
def test_filters_have_no_lookahead_and_only_remove_signals(daily, f):
    full = indicators.add_all(daily, P)
    tampered = daily.copy()
    tampered.iloc[1300:] *= 1.2
    part = indicators.add_all(tampered, P)
    g = (0, f, 0)
    s = ga.make_func(g)(full, P)
    pd.testing.assert_series_equal(s.iloc[:1300], ga.make_func(g)(part, P).iloc[:1300])
    base = ga.make_func((0, 0, 0))(full, P)
    assert ((s == 0) | (s == base)).all()  # 絞り込みの条件はシグナルを減らすだけ


def test_search_respects_budget_is_reproducible_and_records_every_trial(daily):
    r1 = ga.search(daily, RiskParams(), RULES, COSTS, 0.01, budget=30, seed=7, pop_size=12)
    r2 = ga.search(daily, RiskParams(), RULES, COSTS, 0.01, budget=30, seed=7, pop_size=12)
    assert len(r1.evaluated) == 30                       # 予算ちょうど（それ以上は評価しない）
    assert list(r1.evaluated["key"]) == list(r2.evaluated["key"])   # 同じシードなら同じ結果
    assert len(r1.trials) == 30
    hashes = {trials.config_hash("strategy", t["config"]) for t in r1.trials}
    assert len(hashes) == 30                             # 同じ組み合わせを二重に数えない
    assert r1.evaluated["fitness"].is_monotonic_decreasing
    r3 = ga.search(daily, RiskParams(), RULES, COSTS, 0.01, budget=30, seed=8, pop_size=12)
    assert list(r3.evaluated["key"]) != list(r1.evaluated["key"])


def test_search_refuses_final_test_data(daily):
    with pytest.raises(FinalTestAccessError):
        ga.search(daily, RiskParams(), RULES, COSTS, 0.01, budget=5, seed=1, holdout_start=daily.index[1000])
    with pytest.raises(FinalTestAccessError):
        ga.neighborhood_search(daily, [(0, 0, 0)], RiskParams(), RULES, COSTS, 0.01, budget=5, seed=1,
                               holdout_start=daily.index[1000])


def test_neighborhood_search_stays_near_previous_top(daily):
    top = [(0, 1, 0), (5, 0, 0)]
    r = ga.neighborhood_search(daily, top, RiskParams(), RULES, COSTS, 0.01, budget=25, seed=3)
    assert len(r.evaluated) == 25 and r.kind == "neighborhood"
    allowed = set(top)
    for g in top:
        allowed |= set(ga.neighbors(g))
    assert {tuple(g) for g in r.evaluated["genome"]} <= allowed
    assert {tuple(g) for g in r.evaluated["genome"]} >= set(top)   # 前回の上位候補は必ず評価し直す


def test_fitness_penalizes_complexity_and_few_trades(daily):
    ind = indicators.add_all(daily, P)
    bars = bt.Bars(ind)
    f1 = ga.fitness(ind, (0, 0, 0), RiskParams(), RULES, COSTS, 0.01, bars)
    assert len(f1.segment_sharpe) == ga.N_SEGMENTS
    expected = np.mean(f1.segment_sharpe) - 0.5 * np.std(f1.segment_sharpe) - (1.0 if f1.n_trades < 100 else 0.0)
    assert f1.score == pytest.approx(expected)
    f3 = ga.fitness(ind, (0, 1, 4), RiskParams(), RULES, COSTS, 0.01, bars)
    if f3.n_trades >= ga.MIN_TRADES_FOR_FITNESS:
        assert f3.score == pytest.approx(np.mean(f3.segment_sharpe) - 0.5 * np.std(f3.segment_sharpe) - 0.2
                                         - (1.0 if f3.n_trades < 100 else 0.0))


def test_ga_strategy_runs_through_the_normal_validation(daily):
    st = ga.to_strategy((0, 1, 0))
    assert st.family == "ga" and st.spec["ga_genes"]["trigger"] == ga.TRIGGER_IDS[0]
    rep = bt.evaluate_all(daily, P, RiskParams(), RULES, COSTS, 0.01, runs=10, k=3, rw_paths=2, strategy_list=[st])
    assert rep.reports[0].verdict[0]
    # 探索のときの試行と、検証のときの試行は同じ設定なので二重に数えない
    r = ga.SearchResult(0, 1, pd.DataFrame(), [])
    from fxtool.trials import strategy_config
    cfg = strategy_config(st.key, P, RiskParams(), RULES, spec=st.spec)
    assert trials.config_hash("strategy", cfg) == trials.config_hash("strategy", rep.trials[0]["config"])


# ---------------- 保存・週末の再探索・設定 ----------------

@pytest.fixture
def tmp_db(daily, tmp_path, monkeypatch):
    from fxtool import data
    path = tmp_path / "g.db"
    real = data.connect
    monkeypatch.setattr(data, "connect", lambda *a, **k: real(path))
    monkeypatch.setattr(data, "now_utc", lambda: daily.index[-1] + pd.Timedelta(days=2))
    con = real(path)
    data.save(con, "EUR/JPY", "1d", daily)
    yield con
    con.close()


def test_ga_job_uses_dev_data_only_and_records_trials(tmp_db, daily, monkeypatch):
    from fxtool import data, ga_jobs, holdout
    seen = {}
    orig = ga.search

    def spy(dev, *a, **k):
        seen["last"] = dev.index[-1]
        return orig(dev, *a, **{**k, "pop_size": 10})

    monkeypatch.setattr(ga, "search", spy)
    n0 = trials.count(tmp_db, "EUR/JPY", "1d")
    run_id = ga_jobs.run(tmp_db, "EUR/JPY", "1d", "full", seed=5, budget=25)
    assert seen["last"] < holdout.get("EUR/JPY", "1d").start
    assert trials.count(tmp_db, "EUR/JPY", "1d") == n0 + 25
    c = ga_jobs.load_candidates(tmp_db, run_id)
    assert len(c) == 20 and list(c["rank"]) == list(range(1, 21))
    # 予算は設定ファイルの上限（500）を超えられない
    assert ga_jobs.run.__defaults__ is not None
    top = ga_jobs.previous_top(tmp_db, "EUR/JPY", "1d", 10)
    assert len(top) == 10 and tuple(ga_jobs.genome_from_genes(ga.genes_of(top[0]))) == top[0]
    # 週末の再探索：前回の上位候補の周辺だけ
    run2 = ga_jobs.run(tmp_db, "EUR/JPY", "1d", "neighborhood", seed=6, budget=15)
    r = tmp_db.execute("SELECT kind, n_evaluated FROM ga_runs WHERE id=?", (run2,)).fetchone()
    assert r == ("neighborhood", 15)


def test_budget_is_capped_by_settings(tmp_db, monkeypatch):
    from fxtool import ga_jobs, settings
    seen = {}

    def fake(dev, risk, rules, costs, pip, budget, seed, **k):
        seen["budget"] = budget
        return ga.SearchResult(seed, budget, pd.DataFrame(columns=["genome", "key", "rule", "fitness", "trades",
                                                                   "total_pips", "segment_sharpe"]), [])
    monkeypatch.setattr(ga, "search", fake)
    ga_jobs.run(tmp_db, "EUR/JPY", "1d", "full", seed=1, budget=10_000)
    assert seen["budget"] == 500


def test_re_search_runs_every_3_weeks_on_saturday(tmp_db, tmp_path):
    from fxtool import ga_jobs, settings
    sat = pd.Timestamp("2026-10-10 03:00")  # 日本時間の土曜 12:00
    on = settings.load()
    assert on.trial_budget.ga_weekly_enabled is True and on.trial_budget.ga_interval_weeks == 3
    assert not ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", sat, on)   # 本格的な探索の結果がまだない
    tmp_db.execute("""INSERT INTO ga_runs (pair, tf, kind, seed, budget, n_evaluated, started_at, origin)
                      VALUES ('EUR/JPY', '1d', 'full', 1, 500, 500, '2026-09-19 03:00:00', 'dashboard')""")
    assert ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", sat, on)                                    # 3週間後の土曜
    assert not ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", pd.Timestamp("2026-10-03 03:00"), on)  # 2週間後の土曜
    assert not ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", pd.Timestamp("2026-10-11 03:00"), on)  # 日曜
    assert not ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", pd.Timestamp("2026-10-09 03:00"), on)  # 金曜
    tmp_db.execute("""INSERT INTO ga_runs (pair, tf, kind, seed, budget, n_evaluated, started_at, origin)
                      VALUES ('EUR/JPY', '1d', 'neighborhood', 1, 150, 150, '2026-10-10 02:00:00', 'schedule')""")
    assert not ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", sat, on)                                # 今日もう実行した
    assert ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", pd.Timestamp("2026-10-31 01:00"), on)       # その3週間後
    p = tmp_path / "s.toml"
    p.write_text("[trial_budget]\nga_weekly_enabled = false\n", encoding="utf-8")
    off = settings.load(p)
    assert not ga_jobs.weekly_due(tmp_db, "EUR/JPY", "1d", pd.Timestamp("2026-10-31 01:00"), off)
    assert ga_jobs.projected_trials_per_year(on) == 500 + 18 * 150
    assert ga_jobs.projected_trials_per_year(off) == 500


def test_projection_is_capped_by_the_search_space():
    from fxtool import ga_jobs, settings
    s = settings.load()
    p1 = ga_jobs.projection(86, s, 1)
    p4 = ga_jobs.projection(86, s, 4)
    assert p1["re_search_runs"] == 18 and p4["re_search_runs"] == 70
    assert p1["naive"] == 86 + 500 + 18 * 150 and p4["naive"] == 86 + 500 + 70 * 150
    assert p1["capped"] == p4["capped"] == 86 + ga_jobs.search_space_size()
    assert 1 < p1["strict_capped"] <= p1["strict_naive"] < p4["strict_naive"]


def test_convergence_and_budget_suggestion():
    from fxtool import ga_jobs
    trace = [0.1, 0.3, 0.2, 0.5, 0.4, 0.49, 0.5, 0.1] + [0.0] * 92
    c = ga_jobs.convergence(trace)
    assert c.best_at == 4 and c.n == 100 and c.best_so_far[-1] == 0.5
    assert c.near_at == 4
    c2 = ga_jobs.convergence([0.0] * 50 + [0.98, 1.0] + [0.0] * 448)
    assert c2.best_at == 52 and c2.near_at == 51
    assert ga_jobs.suggest_budget(c2, 150) == 30            # 52 の半分 → 30（下限）
    assert ga_jobs.suggest_budget(ga_jobs.convergence([0.0] * 400 + [1.0]), 150) == 150  # 上限
    assert ga_jobs.convergence([]) is None


def test_re_search_skips_already_evaluated_and_trace_is_saved(tmp_db, monkeypatch):
    from fxtool import ga_jobs
    orig = ga.search
    monkeypatch.setattr(ga, "search", lambda dev, *a, **k: orig(dev, *a, **{**k, "pop_size": 10}))
    r1 = ga_jobs.run(tmp_db, "EUR/JPY", "1d", "full", seed=4, budget=30)
    done1 = ga_jobs.evaluated_genomes(tmp_db, "EUR/JPY", "1d")
    assert len(done1) == 30
    trace = json.loads(tmp_db.execute("SELECT trace FROM ga_runs WHERE id=?", (r1,)).fetchone()[0])
    assert len(trace) == 30
    n1 = trials.count(tmp_db, "EUR/JPY", "1d")
    r2 = ga_jobs.run(tmp_db, "EUR/JPY", "1d", "neighborhood", seed=5, budget=20)
    new = {ga_jobs.genome_from_genes(json.loads(g)) for (g,) in tmp_db.execute("SELECT genes FROM ga_candidates WHERE run_id=?", (r2,))}
    assert new and not (new & done1)                                    # 評価済みの組み合わせは評価し直さない
    done2 = ga_jobs.evaluated_genomes(tmp_db, "EUR/JPY", "1d")
    assert len(done2) == 50 and trials.count(tmp_db, "EUR/JPY", "1d") == n1 + 20   # 20通りすべて新しい組み合わせ
    # 上位候補は、これまでのすべての回の中から選ぶ
    best = tmp_db.execute("SELECT genes FROM ga_candidates ORDER BY fitness DESC LIMIT 1").fetchone()[0]
    assert ga_jobs.previous_top(tmp_db, "EUR/JPY", "1d", 1) == [ga_jobs.genome_from_genes(json.loads(best))]

def test_settings_reject_non_boolean_and_over_budget(tmp_path):
    from fxtool import settings
    p = tmp_path / "s.toml"
    p.write_text('[trial_budget]\nga_weekly_enabled = "false"\nga_full_search = 5000\n', encoding="utf-8")
    s = settings.load(p)
    assert s.trial_budget.ga_weekly_enabled is True and s.trial_budget.ga_full_search == 500
    assert len(s.warnings) == 2
