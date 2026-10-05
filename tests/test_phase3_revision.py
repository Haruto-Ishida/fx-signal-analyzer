"""フェーズ3改修（研究ノート反映）のテスト：トリプルバリア・重複度の重み・パージ・メタラベリング・モデルの段階的な比較。"""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import indicators, labeling, ml_eval, strategies
from fxtool.config import IndicatorParams, RiskParams
from test_ml import autocorrelated_ohlc

P, RISK = IndicatorParams(), RiskParams()
COSTS = bt.Costs(0.3, 0.2)


# ---------------- トリプルバリア ----------------

def test_triple_barrier_matches_backtest_engine(random_walk):
    """1つのシグナルの決済は、バックテストのエンジン（反対シグナルなし・時間切れあり）とまったく同じになる。"""
    ind = indicators.add_all(random_walk, P)
    sig = strategies.signals(ind, strategies.STRATEGY_BY_KEY["macd_cross"], P).to_numpy()
    idx = np.flatnonzero(sig)[5:25]
    events = pd.DataFrame({"i": idx, "d": sig[idx]})
    tb = labeling.triple_barrier(ind, events, RISK, 48, COSTS.round_trip, 0.01)
    bars = bt.Bars(ind)
    for _, row in tb.iterrows():
        one = np.zeros(len(ind), dtype=int)
        one[row.i] = row.d
        t = bt.simulate(bars, one.tolist(), RISK, bt.ExitRules(opposite_exit=False, max_bars=48), COSTS, 0.01,
                        row.i, len(ind))
        assert (t.iloc[0].exit_i, round(t.iloc[0].pips, 9)) == (row.t1, round(row.pips, 9))


def test_triple_barrier_drops_unresolved_at_end():
    # 価格は動かず、バリアの幅（ATR）は広い → 利確・損切りには届かず、時間切れでしか決着しない
    ind = pd.DataFrame({"open": 100.0, "high": 100.01, "low": 99.99, "close": 100.0, "atr": 1.0},
                       index=pd.date_range("2025-01-01", periods=300, freq="h"))
    events = pd.DataFrame({"i": [150, 290], "d": [1, 1]})  # 290 は残り9本しかなく、まだ答えがない
    tb = labeling.triple_barrier(ind, events, RISK, 48, 0.0, 0.01)
    assert tb["i"].tolist() == [150] and tb["reason"].iat[0] == "期限" and tb["t1"].iat[0] == 150 + 48


# ---------------- 重複度の重み・パージ ----------------

def test_uniqueness():
    assert labeling.uniqueness([0, 5], [2, 7]).tolist() == [1.0, 1.0]           # 重なりなし
    assert labeling.uniqueness([0, 0], [3, 3]).tolist() == [0.5, 0.5]           # まったく同じ期間
    u = labeling.uniqueness([0, 2], [3, 5])                                     # 2本だけ重なる
    assert u[0] == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)


def test_purged_train_mask():
    t0 = np.array([0, 10, 20, 30, 40, 52, 60])
    t1 = np.array([5, 22, 25, 35, 45, 55, 70])
    m = labeling.purged_train_mask(t0, t1, test_start=20, test_end=40, embargo=10)
    # 0：検証より前に決着 → 使う／10：検証期間にかかる → パージ／20,30：検証期間 → 外す
    # 40：検証の直後（エンバーゴ内）→ 外す／52：まだエンバーゴ内（40+10=50 より後なので使う）／60 → 使う
    assert m.tolist() == [True, False, False, False, False, True, True]


# ---------------- 未来の情報を使わない ----------------

def test_walk_forward_predictions_do_not_use_the_future():
    """1つ目の検証区間より後のデータを書き換えても、1つ目の区間の予測は変わらない（重みの計算も含めて）。"""
    df = make_path_ohlc(2500, seed=8)

    def preds(data):
        ind = indicators.add_all(data, P)
        smp = ml_eval.build_meta(ind, "1h", P, RISK, 48, COSTS.round_trip, 0.01)
        pred, folds, _, _ = ml_eval.walk_forward_predict(smp, "logistic", k=3)
        first = pred[pred["fold"] == 0]
        return smp.t0[first["s"].to_numpy()], first["p"].to_numpy(), folds

    t_a, p_a, folds = preds(df)
    cut = folds[0][1] + 1
    tampered = df.copy()
    tampered.iloc[cut:] = tampered.iloc[cut:] * 1.3
    t_b, p_b, _ = preds(tampered)
    keep = t_b < folds[0][1]
    np.testing.assert_array_equal(t_a, t_b[keep])
    np.testing.assert_allclose(p_a, p_b[keep])


# ---------------- メタラベリング・方向予測 ----------------

def synthetic_meta(n=3000, signal=1.5, seed=0) -> ml_eval.Samples:
    """1つ目の特徴量が大きいほど勝ちやすい、作り物のメタラベリングのデータ。"""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    win = (signal * X["a"] + rng.normal(size=n)) > 0.5
    t0 = np.arange(n) * 2
    pips = np.where(win, 30.0, -15.0)
    return ml_eval.Samples("meta", X, win.astype(float).to_numpy(), t0, t0 + 1, pips, np.full(n, np.nan),
                           np.array(["ma_cross"] * n), pd.date_range("2020-01-01", periods=2 * n + 5, freq="h"), 48)


def test_meta_learns_when_there_is_something_to_learn():
    smp = synthetic_meta()
    r = ml_eval.evaluate(smp, "logistic", 0.5, runs=50, importance=False, purged=False)
    assert r.logloss < r.base_logloss and r.auc > 0.7
    assert r.trade_metrics["avg"] > r.all_metrics["avg"] and r.beat >= 0.95


def test_meta_finds_nothing_in_noise():
    smp = synthetic_meta(signal=0.0, seed=1)
    rep = ml_eval.evaluate_all(smp, 0.5, runs=50)
    assert not any(r.ok for r in rep.results.values())


def test_direction_learns_autocorrelation_and_rejects_noise():
    ind = indicators.add_all(autocorrelated_ohlc(4000, phi=0.3, seed=5), P)
    smp = ml_eval.build_direction(ind, "1h", 1.0, 24, COSTS, 0.01)
    r = ml_eval.evaluate(smp, "logistic", 0.55, runs=30, importance=False, purged=False)
    assert r.acc > 0.55 and r.logloss < r.base_logloss

    noise = indicators.add_all(make_path_ohlc(4000, seed=6), P)
    smp2 = ml_eval.build_direction(noise, "1h", 1.0, 24, COSTS, 0.01)
    rep = ml_eval.evaluate_all(smp2, 0.55, runs=30)
    assert not any(x.ok for x in rep.results.values())


def test_direction_trading_holds_one_position_at_a_time():
    ind = indicators.add_all(make_path_ohlc(3000, seed=7), P)
    smp = ml_eval.build_direction(ind, "1h", 1.0, 24, COSTS, 0.01)
    r = ml_eval.evaluate(smp, "logistic", 0.51, runs=10, importance=False, purged=False)
    # 予測のたびに入るのではなく、ポジションがあるあいだは新しく入らない → 取引数は確信度を超えた予測の数より少ない
    n_signals = int(((r.pred["p"] >= 0.51) | (r.pred["p"] <= 0.49)).sum())
    assert 0 < r.trade_metrics["n"] < n_signals


def test_build_meta_features_and_labels(random_walk):
    ind = indicators.add_all(random_walk, P)
    smp = ml_eval.build_meta(ind, "1h", P, RISK, 48, COSTS.round_trip, 0.01)
    assert {"dir", "from_ma_cross", "ret_1_x_dir", "rsi_x_dir"} <= set(smp.X.columns)
    assert not smp.X.isna().any().any()
    np.testing.assert_array_equal(smp.y, (smp.pips_long > 0).astype(float))
    assert (np.diff(smp.t0) >= 0).all() and (smp.t1 > smp.t0).all()
    # 方向を掛けた特徴量は、方向と元の特徴量の積
    np.testing.assert_allclose(smp.X["ret_1_x_dir"], smp.X["ret_1"] * smp.X["dir"])


def test_quantile_binner_reproduces_sklearn_binning():
    """前処理つきの勾配ブースティングは、重みなしなら scikit-learn の勾配ブースティングと同じ予測になる
    （区切りの計算を先にしているだけで、モデルは同じ）。"""
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=(600, 3)), rng.integers(0, 2, 600), np.round(rng.normal(size=600), 1)]
    y = (X[:, 0] + rng.normal(size=600) > 0).astype(float)
    pipe = ml_eval.LADDER["gbm"][1]
    raw = pipe.steps[-1][1]
    a = clone(raw).fit(X, y).predict_proba(X)[:, 1]
    b = clone(pipe).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_allclose(a, b)
    binned = ml_eval.QuantileBinner().fit(X[:300]).transform(X[300:])
    assert all(len(np.unique(binned[:, j])) <= 255 for j in range(X.shape[1]))


# ---------------- モデルの段階的な比較 ----------------

def fake_result(kind, total):
    return ml_eval.Result(kind, ml_eval.LADDER[kind][0], "meta", 0.4, pd.DataFrame(), [], 0, 0, 0, (0, 0), 0, 0, 0, 0,
                          pd.DataFrame(), {"total": total}, {}, np.array([]), 0, pd.Series(dtype=float))


def test_complex_model_only_when_it_beats_simpler():
    r = {"logistic": fake_result("logistic", 100), "rf": fake_result("rf", 90), "gbm": fake_result("gbm", 150)}
    assert ml_eval.choose_model(r)[0] == "gbm"
    r = {"logistic": fake_result("logistic", 100), "rf": fake_result("rf", 120), "gbm": fake_result("gbm", 110)}
    assert ml_eval.choose_model(r)[0] == "rf"  # 勾配ブースティングはランダムフォレストに勝てていない
    r = {"logistic": fake_result("logistic", 100), "rf": fake_result("rf", 50), "gbm": fake_result("gbm", 80)}
    adopted, notes = ml_eval.choose_model(r)
    assert adopted == "logistic" and len(notes) == 2


def test_report_records_trials_and_deflates(random_walk):
    smp = synthetic_meta(n=1500, signal=1.0, seed=3)
    rep = ml_eval.evaluate_all(smp, 0.45, runs=20)
    assert {t["name"] for t in rep.trials} == set(ml_eval.LADDER)
    assert all(t["config"]["label"] == "triple_barrier_meta" for t in rep.trials)
    before = rep.results[rep.adopted].dsr["dsr"]
    ml_eval.finalize(rep, smp, 1000)
    assert rep.results[rep.adopted].dsr["dsr"] <= before
