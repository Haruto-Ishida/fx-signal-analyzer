"""Step 1：戦略と予測エンジンの共通の形のテスト。"""
import numpy as np
import pandas as pd
import pytest

from fxtool import backtest as bt
from fxtool import engines, strategies, trials
from fxtool.config import IndicatorParams, RiskParams

P, RISK = IndicatorParams(), RiskParams()
RULES = bt.ExitRules(max_bars=48)


def sma_cross_custom(d, p):
    """テスト用の「別の種類の戦略」：終値が SMA20 を上抜け・下抜け。"""
    above = d["close"] > d["sma_s"]
    prev = above.shift(fill_value=False).astype(bool)
    return strategies._signal(above & ~prev, ~above & prev & d["sma_s"].notna())


CUSTOM = strategies.Strategy("custom_sma", "テスト戦略", "順張り", "終値がSMAを抜けたら", sma_cross_custom,
                             family="ga", source="テスト", spec={"genes": ["close>sma_s"]})


def test_builtin_strategies_keep_their_grids_and_config_hashes():
    for s in strategies.STRATEGIES:
        assert s.family == "builtin" and s.grid() == strategies.PARAM_GRIDS[s.key]
    # 最初からある戦略の設定の形（ハッシュ）は変わらない ＝ 試行数が二重に数えられない
    cfg = trials.strategy_config("ma_cross", P, RISK, RULES)
    assert "spec" not in cfg
    assert trials.config_hash("strategy", cfg) == trials.config_hash(
        "strategy", trials.strategy_config("ma_cross", P, RISK, RULES, spec={}))


def test_any_strategy_can_be_evaluated_and_recorded(random_walk):
    rep = bt.evaluate_all(random_walk, P, RISK, RULES, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2,
                          strategy_list=[CUSTOM])
    assert [r.strategy.key for r in rep.reports] == ["custom_sma"]
    assert rep.grid_daily.columns.tolist() == ["custom_sma|初期設定"]
    assert rep.random_walk.columns.tolist() == ["custom_sma"]
    rows = rep.trials
    assert all(r["config"]["spec"] == {"genes": ["close>sma_s"]} for r in rows)
    planned = bt.planned_trials(P, RISK, RULES, [CUSTOM])
    assert {trials.config_hash("strategy", r["config"]) for r in planned} == \
           {trials.config_hash("strategy", r["config"]) for r in rows}


def test_default_evaluation_is_unchanged_by_the_refactor(random_walk):
    """strategy_list を省略したときは、最初からある5つの戦略で、以前と同じ評価になる。"""
    a = bt.evaluate_all(random_walk, P, RISK, RULES, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2)
    b = bt.evaluate_all(random_walk, P, RISK, RULES, bt.Costs(), 0.01, runs=10, k=3, rw_paths=2,
                        strategy_list=strategies.STRATEGIES)
    assert [r.strategy.key for r in a.reports] == [s.key for s in strategies.STRATEGIES]
    for x, y in zip(a.reports, b.reports):
        assert x.full_metrics == y.full_metrics and x.wf.test_metrics == y.wf.test_metrics
    assert len(bt.planned_trials(P, RISK, RULES)) == sum(len(g) for g in strategies.PARAM_GRIDS.values())


@pytest.mark.parametrize("kind", list(engines.LADDER))
def test_engine_interface(kind):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + rng.normal(size=400) > 0).astype(float)
    w = rng.uniform(0.5, 1.0, 400)
    e = engines.Engine(kind).train(X, y, w)
    p = e.predict(X)
    assert p.shape == (400,) and ((0 <= p) & (p <= 1)).all()
    assert np.corrcoef(p, X[:, 0])[0, 1] > 0.5  # 効いている特徴量を学習できている
    with pytest.raises(RuntimeError):
        engines.Engine(kind).predict(X)
    with pytest.raises(ValueError):
        engines.Engine("lstm")
