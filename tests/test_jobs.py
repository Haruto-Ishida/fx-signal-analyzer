import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import data, jobs, strategies, tracking
from fxtool.config import ML_HORIZON, IndicatorParams

PAIR, TF = "USD/JPY", "1h"
FULL = make_path_ohlc(2200, seed=11)  # 2020-01-01 00:00 から1時間ごと


class World:
    """時計を進めると、その時点までのデータが「取得できる」ようになる偽の環境。"""

    def __init__(self, tmp_path, monkeypatch, start_bars: int):
        self.now = FULL.index[start_bars] + pd.Timedelta(minutes=30)  # start_bars 本目が形成中
        monkeypatch.setattr(data, "now_utc", lambda: self.now)
        monkeypatch.setattr(data, "download", lambda ticker, tf, period: FULL[FULL.index <= self.now])
        monkeypatch.setattr(jobs, "MODELS_DIR", tmp_path / "data" / "models")
        monkeypatch.setattr(jobs, "ROOT", tmp_path)
        self.con = data.connect(tmp_path / "t.db")

    def advance(self, hours: int):
        self.now += pd.Timedelta(hours=hours)

    def cycle(self, **kw):
        r = jobs.run_cycle(self.con, PAIR, TF, now=self.now, force=True, **kw)
        assert r.error is None, r.error
        return r


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch, start_bars=1500)


def q(con, sql, *args):
    return con.execute(sql, args).fetchall()


def test_first_cycle_trains_and_logs_only_latest(world):
    r = world.cycle()
    assert sorted(r.retrained) == ["gbm", "logistic"]
    assert r.data_added == 1501
    preds = tracking.load_predictions(world.con, PAIR, TF)
    assert len(preds) == 2  # 各モデル、最新の確定足の1件だけ
    assert (preds["bar_time"] == FULL.index[1499]).all()
    assert len(tracking.latest_weights(world.con, PAIR, TF)) == len(strategies.STRATEGIES)
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    assert (sl["signal_time"] == FULL.index[1499]).all()  # 過去にさかのぼって記録しない
    v = tracking.load_verdicts(world.con, PAIR, TF)
    assert r.verdicts_updated and sorted(v["source"]) == sorted(s.key for s in strategies.STRATEGIES)
    ml_status = tracking.latest_ml_status(world.con, PAIR, TF)
    assert {m.model for m in ml_status} == {"logistic", "gbm"}
    assert all(m.bar_time == FULL.index[1499] for m in ml_status)
    world.advance(5)
    assert not world.cycle().verdicts_updated  # 24本経つまでは再計算しない


def test_predictions_never_use_models_trained_on_that_bar(world):
    world.cycle()
    for _ in range(4):
        world.advance(30)
        world.cycle()
    preds = tracking.load_predictions(world.con, PAIR, TF)
    reg = tracking.load_registry(world.con, PAIR, TF).set_index("version")
    assert len(reg) >= 6  # 30本ごとに進めて、24本ごとに再学習
    through = reg.loc[preds["version"], "trained_through"].to_numpy()
    assert (preds["bar_time"].to_numpy() > through).all()
    # 時計を進めた分の足がすべて予測されている（取りこぼしなし）
    per_model = preds.groupby("model")["bar_time"].apply(lambda s: s.is_unique and len(s))
    assert (per_model == 1 + 4 * 30).all()


def test_new_bars_are_predicted_by_the_old_model_before_retraining(world):
    world.cycle()
    v1 = {k: jobs.latest_model(world.con, PAIR, TF, k)["version"] for k in ("logistic", "gbm")}
    world.advance(30)
    r = world.cycle()
    assert sorted(r.retrained) == ["gbm", "logistic"]
    preds = tracking.load_predictions(world.con, PAIR, TF)
    new = preds[preds["bar_time"] > FULL.index[1499]]
    assert len(new) == 60
    for k in ("logistic", "gbm"):
        assert (new.loc[new["model"] == k, "version"] == v1[k]).all()


def test_strategy_signals_logged_exactly_once(world):
    world.cycle()
    world.advance(200)
    world.cycle()
    world.cycle()  # 同じ時刻にもう一度：重複しない
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    sl = sl[~sl["source"].str.startswith("ml_")]

    from fxtool import indicators
    ind = indicators.add_all(FULL.iloc[:1700], IndicatorParams())
    expected = 0
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, IndicatorParams()).to_numpy()
        expected += int(np.count_nonzero(sig[1499:1700]))
    assert len(sl) == expected and expected > 0
    assert sl["weight"].notna().all()


def test_outcomes_and_labels_are_filled_correctly(world):
    world.cycle()
    world.advance(300)
    world.cycle()
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    done = sl[sl["status"] != "進行中"]
    assert len(done) > 0
    cost = jobs.default_costs(PAIR).round_trip
    want = done["direction"] * (done["exit_price"] - done["entry"]) / 0.01 - cost
    np.testing.assert_allclose(done["pips"], want)
    assert set(done["status"]) <= {"利確", "損切り", "期限"}

    preds = tracking.load_predictions(world.con, PAIR, TF)
    last_conf = FULL.index[1799 - 1]
    filled = preds.dropna(subset=["y"])
    assert (preds.loc[preds["y"].isna(), "bar_time"] > last_conf - pd.Timedelta(hours=ML_HORIZON)).all()
    close = FULL["close"]
    for _, row in filled.sample(20, random_state=0).iterrows():
        i = FULL.index.get_loc(row["bar_time"])
        assert row["y"] == float(close.iat[i + ML_HORIZON] > close.iat[i])


def test_throttle_and_job_runs(world):
    world.cycle()
    r = jobs.run_cycle(world.con, PAIR, TF, now=world.now)  # force なし・直後
    assert r.skipped
    runs = tracking.load_job_runs(world.con)
    assert len(runs) == 1 and runs["error"].isna().all()


# ---------------- 重み・警告 ----------------

def test_weights():
    assert tracking.weight_from_z(0) == 1
    assert tracking.weight_from_z(10) == 1.75 and tracking.weight_from_z(-10) == 0.25
    assert tracking.z_score(np.array([1.0, 2.0])) == 0  # 取引が少ないときは中立
    assert tracking.weighted_score({"a": "買い", "b": "売り"}, {"a": 1.5, "b": 0.5}) == pytest.approx(0.5)
    assert tracking.weighted_score({"a": "様子見"}, {"a": 1.0}) == 0


def test_recent_performance_weights_average_to_one(random_walk):
    from fxtool import backtest as bt
    from fxtool import indicators
    ind = indicators.add_all(random_walk, IndicatorParams())
    rec = tracking.recent_performance(ind, IndicatorParams(), jobs.RISK, bt.Costs(), 0.01, 720)
    assert np.mean([r.weight for r in rec]) == pytest.approx(1.0)
    assert all(0.1 < r.weight < 2.0 for r in rec)


def test_alert_sync_dedupes_and_resolves(tmp_path):
    con = data.connect(tmp_path / "a.db")
    t = pd.Timestamp("2026-10-05 10:00")
    c = tracking.Condition("ma_cross", "recent_bad", "warning", "悪化")
    assert tracking.sync_alerts(con, PAIR, TF, [c], t) == 1
    assert tracking.sync_alerts(con, PAIR, TF, [c], t + pd.Timedelta(hours=1)) == 0  # 同じ警告は増やさない
    assert len(tracking.load_alerts(con, PAIR, TF, active_only=True)) == 1
    tracking.sync_alerts(con, PAIR, TF, [], t + pd.Timedelta(hours=2))  # 条件が消えたら解決
    a = tracking.load_alerts(con, PAIR, TF)
    assert len(a) == 1 and a["resolved_at"].notna().all()


def test_strategy_conditions():
    rec = [tracking.RecentPerf("ma_cross", 20, -500, 0.5, -2.5),
           tracking.RecentPerf("macd_cross", 20, -100, 0.7, -1.0),
           tracking.RecentPerf("breakout", 5, -300, 0.2, -3.0)]   # 取引が少ないので警告しない
    logs = pd.DataFrame({"source": ["rsi_reversal"] * 7, "exit_time": pd.date_range("2026", periods=7, freq="h"),
                         "pips": [5, -1, -2, -3, -4, -5, -6]})
    conds = {(c.source, c.kind): c.level for c in tracking.strategy_conditions(rec, logs, "直近")}
    assert conds == {("ma_cross", "recent_bad"): "critical", ("macd_cross", "recent_bad"): "warning",
                     ("rsi_reversal", "losing_streak"): "warning"}


def test_model_conditions():
    reg = pd.DataFrame({"model": ["gbm", "gbm"], "horizon": [1, 1], "ok": [1, 0], "acc": [0.55, 0.53]})
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 200)
    y = (p < 0.5).astype(float)  # 予測が完全に外れている
    preds = pd.DataFrame({"bar_time": pd.date_range("2026", periods=200, freq="h"), "p": p, "y": y})
    kinds = {c.kind: c.level for c in tracking.model_conditions(reg, preds)}
    assert kinds == {"lost_edge": "warning", "live_acc": "critical"}
    assert tracking.model_conditions(reg.iloc[:1], preds.iloc[:10]) == []


def test_data_condition_ignores_weekend():
    bar = pd.Timedelta(hours=1)
    friday_close = pd.Timestamp("2026-10-02 20:00")
    assert tracking.data_condition(friday_close, pd.Timestamp("2026-10-03 12:00"), bar, 6) == []   # 土曜
    assert tracking.data_condition(friday_close, pd.Timestamp("2026-10-01 12:00") , bar, 6) == []  # 新しい
    assert len(tracking.data_condition(pd.Timestamp("2026-10-06 01:00"), pd.Timestamp("2026-10-06 12:00"), bar, 6)) == 1
