"""Step 6：GA の候補の検証・合格基準（ランダムウォーク・年別）・分数階差分。"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import backtest as bt
from fxtool import fracdiff, ga, ml, trials


# ---------------- 合格基準 ----------------

def wf_with(total, n=150, pos=5):
    return bt.WalkForwardResult([], pd.DataFrame(), {"n": n, "total": total}, 0.0, pos)


def test_assess_requires_beating_random_walk():
    wf = wf_with(500.0)
    assert bt.assess(wf, 0.99, 5, 0.99, 0.1)[0] == "有望（要継続監視）"
    v = bt.assess(wf, 0.99, 5, 0.99, 0.1, rw_beat=0.6)
    assert v[0] == "偶然の範囲" and "ランダムウォーク" in v[1]
    assert bt.assess(wf, 0.99, 5, 0.99, 0.1, rw_beat=0.97)[0] == "有望（要継続監視）"


def test_assess_rejects_an_extremely_bad_year():
    wf = wf_with(500.0)
    ok = pd.DataFrame({"year": [2020, 2021, 2022], "total": [300.0, 400.0, -200.0]})
    assert bt.assess(wf, 0.99, 5, 0.99, 0.1, 0.99, ok)[0] == "有望（要継続監視）"
    bad = pd.DataFrame({"year": [2020, 2021, 2022], "total": [600.0, 400.0, -500.0]})
    v = bt.assess(wf, 0.99, 5, 0.99, 0.1, 0.99, bad)
    assert v[0] == "不安定" and "2022年" in v[1]
    assert bt.worst_year(bad, 500.0) == (2022, -500.0) and bt.worst_year(ok, 500.0) is None


def test_verdicts_are_recomputed_when_rules_change():
    from fxtool import costmodel, jobs
    assert jobs.VERDICT_STATE == f"{costmodel.COST_MODEL_VERSION}|{bt.VERDICT_VERSION}"


# ---------------- GA の候補の検証 ----------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    from fxtool import data
    df = make_path_ohlc(1800, sd=0.6, seed=21)
    df.index = pd.bdate_range("2012-01-01", periods=len(df))
    path = tmp_path / "v.db"
    real = data.connect
    monkeypatch.setattr(data, "connect", lambda *a, **k: real(path))
    monkeypatch.setattr(data, "now_utc", lambda: df.index[-1] + pd.Timedelta(days=2))
    con = real(path)
    data.save(con, "EUR/JPY", "1d", df)
    yield con
    con.close()


def test_validate_top_candidates_adds_no_trials(tmp_db, monkeypatch):
    from fxtool import ga_jobs
    orig = ga.search
    monkeypatch.setattr(ga, "search", lambda dev, *a, **k: orig(dev, *a, **{**k, "pop_size": 10}))
    run_id = ga_jobs.run(tmp_db, "EUR/JPY", "1d", "full", seed=2, budget=20)
    n1 = trials.count(tmp_db, "EUR/JPY", "1d")
    n_used = ga_jobs.validate(tmp_db, "EUR/JPY", "1d", run_id, top_n=3, runs=10, rw_paths=2)
    assert trials.count(tmp_db, "EUR/JPY", "1d") == n1 == n_used   # 探索のときと同じ設定なので増えない
    c = ga_jobs.load_candidates(tmp_db, run_id)
    assert c["verdict"].head(3).notna().all() and c["verdict"].iloc[3:].isna().all()
    assert (c["n_trials"].head(3) == n1).all()
    json.loads(c["yearly"].iloc[0])


# ---------------- 分数階差分 ----------------

def test_fracdiff_weights_match_known_cases():
    assert np.allclose(fracdiff.weights(1.0), [1.0, -1.0])           # d=1 はふつうの差分
    assert np.allclose(fracdiff.weights(0.0), [1.0])                 # d=0 は元の値
    w = fracdiff.weights(0.5, threshold=1e-6, max_width=4)
    assert np.allclose(w, [1, -0.5, -0.125, -0.0625])
    assert len(fracdiff.weights(0.4)) > 10 and abs(fracdiff.weights(0.4)[-1]) >= fracdiff.DEFAULT_THRESHOLD


def test_frac_diff_values_and_no_lookahead():
    x = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=800)) + 100)
    d1 = fracdiff.frac_diff(x, 1.0)
    assert np.allclose(d1.iloc[1:], x.diff().iloc[1:]) and math.isnan(d1.iloc[0])
    fd = fracdiff.frac_diff(x, 0.4)
    w = fracdiff.weights(0.4)
    i = 700
    assert fd.iloc[i] == pytest.approx(sum(w[k] * x.iloc[i - k] for k in range(len(w))))
    assert fd.iloc[:len(w) - 1].isna().all() and fd.iloc[len(w) - 1:].notna().all()
    y = x.copy()
    y.iloc[600:] *= 1.5
    pd.testing.assert_series_equal(fd.iloc[:600], fracdiff.frac_diff(y, 0.4).iloc[:600])


def test_fracdiff_keeps_memory_and_is_more_stable_than_price():
    rng = np.random.default_rng(3)
    x = pd.Series(np.log(100 + np.cumsum(rng.normal(0, 0.3, 4000))))
    fd = fracdiff.frac_diff(x, 0.4).dropna()
    d1 = x.diff().dropna()
    # 元の価格との相関（記憶）は、ふつうの差分より大きい
    assert np.corrcoef(fd, x.loc[fd.index])[0, 1] > abs(np.corrcoef(d1, x.loc[d1.index])[0, 1])
    # 前半と後半の平均の差（水準のずれ）は、元の価格より小さい（標準偏差で割って比べる）
    def drift(s):
        h = len(s) // 2
        return abs(s.iloc[:h].mean() - s.iloc[h:].mean()) / s.std()
    assert drift(fd) < drift(x.loc[fd.index])


def test_make_features_fracdiff_is_optional():
    from fxtool import indicators
    from fxtool.config import IndicatorParams
    df = make_path_ohlc(900, sd=0.2, seed=5)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")
    ind = indicators.add_all(df, IndicatorParams())
    base = ml.make_features(ind, "1h")
    plus = ml.make_features(ind, "1h", fracdiff=True)
    assert "fd_close" not in base.columns                         # 本番（初期値）は今まで通り
    assert list(plus.columns) == list(base.columns) + ["fd_close"]
    pd.testing.assert_frame_equal(plus[base.columns], base)
    assert ml.feature_label("fd_close").startswith("分数階差分")
