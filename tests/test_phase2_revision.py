"""フェーズ2改修（研究ノート反映）のテスト：シャープレシオ・PBO・DSR・ランダムウォーク・年別・判定。"""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import overfit
from fxtool.config import IndicatorParams, RiskParams

P, RISK = IndicatorParams(), RiskParams()


# ---------------- シャープレシオ ----------------

def test_daily_pnl_and_sharpe():
    days = overfit.business_days(pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-09"))  # 月〜金
    d = overfit.daily_pnl(pd.to_datetime(["2026-10-05 03:00", "2026-10-05 20:00", "2026-10-08 10:00"]),
                          [10, -4, 6], days)
    assert d.tolist() == [6, 0, 0, 6, 0]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert overfit.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1))
    assert overfit.sharpe(np.zeros(5)) == 0
    assert overfit.annualized(0.1) == pytest.approx(0.1 * math.sqrt(252))


# ---------------- DSR ----------------

def test_expected_max_sharpe_matches_simulation():
    """N 本の「実力ゼロ」の戦略を作ったときの最大シャープレシオの平均が、理論値（SR*）と合う。"""
    rng = np.random.default_rng(0)
    T, N = 500, 50
    maxes, all_sr = [], []
    for _ in range(200):
        srs = [overfit.sharpe(rng.normal(0, 1, T)) for _ in range(N)]
        maxes.append(max(srs))
        all_sr += srs
    theory = overfit.expected_max_sharpe(N, float(np.var(all_sr)))
    assert np.mean(maxes) == pytest.approx(theory, rel=0.1)
    assert overfit.expected_max_sharpe(1, 0.01) == 0
    assert overfit.expected_max_sharpe(100, 0.01) > overfit.expected_max_sharpe(10, 0.01)


def test_dsr_falls_as_more_trials_are_counted():
    rng = np.random.default_rng(1)
    daily = rng.normal(0.08, 1, 750)  # わずかにプラス
    few = overfit.deflated_sharpe(daily, 2, 0.003)["dsr"]
    many = overfit.deflated_sharpe(daily, 500, 0.003)["dsr"]
    assert few > many
    assert overfit.probabilistic_sharpe(0.0, 0.0, daily) == pytest.approx(0.5)


# ---------------- PBO ----------------

def test_pbo_is_about_half_for_pure_noise():
    rng = np.random.default_rng(2)
    m = pd.DataFrame(rng.normal(0, 1, (800, 20)))
    assert 0.3 < overfit.pbo_cscv(m)["pbo"] < 0.7


def test_pbo_is_low_when_one_config_really_works():
    rng = np.random.default_rng(3)
    m = pd.DataFrame(rng.normal(0, 1, (800, 20)))
    m[0] += 0.5  # どの期間でも明らかに良い
    assert overfit.pbo_cscv(m)["pbo"] < 0.05


def test_pbo_is_high_when_past_winner_reverses():
    """前半に強い設定は後半に弱い（その逆も）ので、学習側の勝者は検証側で負ける。"""
    rng = np.random.default_rng(4)
    T = 800
    trend = np.where(np.arange(T) < T // 2, 1.0, -1.0)
    m = pd.DataFrame({"a": trend + rng.normal(0, 1, T), "b": -trend + rng.normal(0, 1, T)})
    assert overfit.pbo_cscv(m)["pbo"] > 0.7


# ---------------- ランダムウォーク・年別・指標 ----------------

def test_random_walk_keeps_bar_shapes_but_shuffles_order():
    df = make_path_ohlc(500, seed=5)
    rw = bt.random_walk_ohlc(df, np.random.default_rng(0))
    assert rw.index.equals(df.index) and rw["open"].iat[0] == df["open"].iat[0]
    np.testing.assert_allclose(np.sort((rw["close"] - rw["open"]).to_numpy()),
                               np.sort((df["close"] - df["open"]).to_numpy()), atol=1e-9)
    assert (rw["high"] >= rw[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (rw["low"] <= rw[["open", "close"]].min(axis=1) + 1e-9).all()
    assert not np.allclose(rw["close"].to_numpy(), df["close"].to_numpy())


def test_yearly_sums_to_total():
    idx = pd.date_range("2024-12-30", periods=6, freq="D")
    t = pd.DataFrame({"exit_i": [0, 1, 3, 5], "pips": [10.0, -5.0, 7.0, 1.0]})
    y = bt.yearly(t, idx)
    assert y["year"].tolist() == [2024, 2025] and y["total"].sum() == pytest.approx(13)


def test_metrics_payoff():
    m = bt.metrics([10, -5, 20, -10])
    assert m["avg_win"] == 15 and m["avg_loss"] == -7.5 and m["payoff"] == pytest.approx(2.0)


# ---------------- 判定 ----------------

def fake_wf(total, n, pos):
    return bt.WalkForwardResult([], pd.DataFrame(), dict(n=n, total=total), 0.0, pos)


def test_assess_requires_everything_for_promising():
    assert bt.assess(fake_wf(100, 150, 4), 0.99, 5, dsr=0.97, pbo=0.2)[0] == "有望（要継続監視）"
    assert bt.assess(fake_wf(100, 80, 4), 0.99, 5, dsr=0.97, pbo=0.2)[0] == "参考値"        # 100回未満
    assert bt.assess(fake_wf(100, 150, 4), 0.99, 5, dsr=0.70, pbo=0.2)[0] == "偶然の可能性"  # DSR 不足
    assert bt.assess(fake_wf(100, 150, 4), 0.99, 5, dsr=0.97, pbo=0.6)[0] == "過学習の疑い"  # PBO 高い
    assert bt.assess(fake_wf(100, 150, 4), 0.90, 5, dsr=0.97, pbo=0.2)[0] == "偶然の範囲"
    assert bt.assess(fake_wf(100, 150, 2), 0.99, 5, dsr=0.97, pbo=0.2)[0] == "不安定"
    assert bt.assess(fake_wf(-5, 150, 4), 0.99, 5, dsr=0.97, pbo=0.2)[0] == "優位性なし"


def test_evaluate_all_produces_overfit_checks(random_walk):
    rep = bt.evaluate_all(random_walk, P, RISK, bt.ExitRules(max_bars=48), bt.Costs(), 0.01, runs=20, k=3, rw_paths=5)
    n_grid = sum(len(g) for g in bt.PARAM_GRIDS.values())
    assert rep.grid_daily.shape[1] == n_grid and rep.n_trials == n_grid
    assert rep.random_walk.shape == (5, 5)
    for r in rep.reports:
        assert r.pbo["n_configs"] == len(bt.PARAM_GRIDS[r.strategy.key])
        assert 0 <= r.dsr["dsr"] <= 1 and not math.isnan(r.wf_sharpe)
        assert r.yearly_full["total"].sum() == pytest.approx(r.full_metrics["total"])
        assert r.verdict[0]
    # ランダムな値動きでは「有望」は出ない
    assert not any(r.verdict[0].startswith("有望") for r in rep.reports)
    # 試した数を多く数えるほど DSR は下がる
    before = {r.strategy.key: r.dsr["dsr"] for r in rep.reports}
    rep2 = bt.finalize(rep, 1000)
    assert all(r.dsr["dsr"] <= before[r.strategy.key] + 1e-12 for r in rep2.reports)
