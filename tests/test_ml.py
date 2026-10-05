import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import indicators, ml
from fxtool.config import IndicatorParams, RiskParams

P = IndicatorParams()
ARGS = dict(risk=RiskParams(), costs=bt.Costs(0.3, 0.2), pip=0.01, k=3, runs=50)


def ind_of(df):
    return indicators.add_all(df, P)


def autocorrelated_ohlc(n: int, phi: float, seed: int = 0) -> pd.DataFrame:
    """前の足と同じ方向に動きやすい（予測できる）データ。"""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.normal(0, 0.1, n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    close = 150 + np.cumsum(r)
    open_ = np.r_[150, close[:-1]]
    wick = np.abs(rng.normal(0, 0.03, n))
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick, "close": close,
    }, index=pd.date_range("2020-01-01", periods=n, freq="h"))


def test_target():
    y = ml.make_target(pd.Series([1.0, 2.0, 1.5, 1.5, 3.0]), 1)
    assert y.tolist()[:4] == [1.0, 0.0, 0.0, 1.0] and np.isnan(y.iat[4])
    y2 = ml.make_target(pd.Series([1.0, 2.0, 1.5, 1.5, 3.0]), 2)
    assert y2.tolist()[:3] == [1.0, 0.0, 1.0] and y2.iloc[3:].isna().all()


@pytest.mark.parametrize("tf_key", ["1h", "1d"])
def test_features_do_not_look_ahead(tf_key):
    df = make_path_ohlc(1200, seed=1)
    full = ml.make_features(ind_of(df), tf_key)
    tampered = df.copy()
    tampered.iloc[800:] *= 1.3
    part = ml.make_features(ind_of(tampered), tf_key)
    pd.testing.assert_frame_equal(full.iloc[:800], part.iloc[:800])
    assert full.iloc[200:].notna().all().all()


def test_folds_have_purge_gap():
    folds = ml.make_folds(start=100, end=5100, h=4, k=5, test_frac=0.5)
    assert len(folds) == 5
    for (a, b), (ts, te) in folds:
        assert a == 100 and ts - b == 4  # 学習の最後の h 本を外している
    assert folds[0][1][0] == 2600 and folds[-1][1][1] == 5100
    assert all(folds[i][1][1] == folds[i + 1][1][0] for i in range(4))


def test_predictions_do_not_use_data_after_their_fold():
    """1つ目の検証区間より後ろのデータを書き換えても、1つ目の区間の予測は変わらない。"""
    df = make_path_ohlc(3000, seed=2)
    ev = ml.evaluate_model(ind_of(df), "1h", "logistic", 1, 0.55, **ARGS)
    te0 = ev.folds[0].test[1]
    tampered = df.copy()
    tampered.iloc[te0 + 1:] *= 1.2
    ev2 = ml.evaluate_model(ind_of(tampered), "1h", "logistic", 1, 0.55, **ARGS)
    p1 = ev.pred[ev.pred["fold"] == 0]
    p2 = ev2.pred[ev2.pred["fold"] == 0]
    np.testing.assert_allclose(p1["p"], p2["p"])


@pytest.mark.parametrize("kind", ["logistic", "gbm"])
def test_no_signal_on_pure_noise(kind):
    df = make_path_ohlc(4000, seed=4)
    ev = ml.evaluate_model(ind_of(df), "1h", kind, 1, 0.55, **ARGS)
    assert not ev.ok
    assert ml.decision(ev) == "シグナルなし"
    assert 0.44 < ev.acc < 0.56


def test_detects_real_predictability():
    df = autocorrelated_ohlc(4000, phi=0.3, seed=5)
    ev = ml.evaluate_model(ind_of(df), "1h", "logistic", 1, 0.55, **ARGS)
    names = [c[0] for c in ev.checks]
    passed = dict(zip(names, [c[1] for c in ev.checks]))
    assert ev.acc > 0.55
    assert passed["正解率が基準を上回る"] and passed["確率の質が基準より良い"]
    assert ev.importance.index[0] == "ret_1"  # 効いているのは直近の値動き
    assert ev.momentum_acc > 0.55  # 「直前と同じ方向」という単純な基準もこのデータでは当たる


def test_decision_respects_threshold():
    df = autocorrelated_ohlc(3000, phi=0.3, seed=6)
    ev = ml.evaluate_model(ind_of(df), "1h", "logistic", 1, 0.55, **ARGS)
    ev.checks = [("x", True, "")]
    for p, want in [(0.6, "買い"), (0.4, "売り"), (0.52, "様子見")]:
        ev.current_p = p
        assert ml.decision(ev) == want
    ev.checks = [("x", False, "")]
    ev.current_p = 0.9
    assert ml.decision(ev) == "シグナルなし"


def test_calibration_table():
    pred = pd.DataFrame({"p": np.linspace(0.3, 0.7, 100), "y": [0, 1] * 50})
    cal = ml.calibration(pred, bins=5)
    assert len(cal) == 5 and cal["件数"].sum() == 100
