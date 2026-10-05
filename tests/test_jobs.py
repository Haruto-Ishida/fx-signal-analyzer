import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_path_ohlc
from fxtool import data, holdout, jobs, ml_eval, strategies, tracking, trials
from fxtool.config import IndicatorParams

PAIR, TF = "USD/JPY", "1h"
FULL = make_path_ohlc(3200, seed=11)  # 2020-01-01 00:00 から1時間ごと
START = 2400


class World:
    """時計を進めると、その時点までのデータが「取得できる」ようになる偽の環境。"""

    def __init__(self, tmp_path, monkeypatch, start_bars: int):
        self.now = FULL.index[start_bars] + pd.Timedelta(minutes=30)  # start_bars 本目が形成中
        monkeypatch.setattr(data, "now_utc", lambda: self.now)
        monkeypatch.setattr(data, "download", lambda ticker, tf, period: FULL[FULL.index <= self.now])
        monkeypatch.setattr(jobs, "MODELS_DIR", tmp_path / "data" / "models")
        monkeypatch.setattr(jobs, "ROOT", tmp_path)
        # 検証の細かさはテストでは小さく（仕組みの確認が目的なので）
        monkeypatch.setattr(jobs, "VERDICT_RANDOM_RUNS", 10)
        monkeypatch.setattr(jobs, "VERDICT_RW_PATHS", 2)
        monkeypatch.setattr(jobs, "META_RANDOM_RUNS", 20)
        self.con = data.connect(tmp_path / "t.db")

    def advance(self, hours: int):
        self.now += pd.Timedelta(hours=hours)

    def cycle(self, **kw):
        r = jobs.run_cycle(self.con, PAIR, TF, now=self.now, force=True, **kw)
        assert r.error is None, r.error
        return r


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch, start_bars=START)


def meta_registry(con):
    reg = tracking.load_registry(con, PAIR, TF)
    return reg[reg["model"] == "meta"].reset_index(drop=True)


def test_first_cycle(world):
    r = world.cycle()
    assert r.retrained and r.verdicts_updated and r.data_added == START + 1
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    assert (sl["signal_time"] == FULL.index[START - 1]).all()  # 過去にさかのぼって記録しない
    assert tracking.load_predictions(world.con, PAIR, TF).empty  # 旧方式の予測はもう記録しない

    # 最終テスト期間：直近20%で固定。判定（検証）はそれより前だけ、本番モデルは決着した最新のラベルまで
    b = holdout.get(PAIR, TF)
    assert b.start == FULL.index[int(START * 0.8)]
    reg = meta_registry(world.con)
    assert len(reg) == 1 and reg["kind"].iat[0] in ml_eval.LADDER
    assert reg["eval_to"].iat[0] < b.start and reg["eval_from"].iat[0] < reg["eval_to"].iat[0]
    assert reg["trained_through"].iat[0] > b.start  # 最終テスト期間も本番モデルの学習には使ってよい
    assert reg["trained_bar"].iat[0] == FULL.index[START - 1]
    assert reg["n_trials"].iat[0] == trials.count(world.con, PAIR, TF)
    v = tracking.load_verdicts(world.con, PAIR, TF)
    assert (v["bar_time"] < b.start).all() and (v["n_trials"] == trials.count(world.con, PAIR, TF)).all()
    # 試した組み合わせ：推定の過去分 + 戦略の候補37 + メタラベリングの3モデル
    assert trials.count(world.con, PAIR, TF, origin="job") == 37 + 3

    # 重みは本番の記録から。決済済みがまだないので全部 1
    w = tracking.latest_weights(world.con, PAIR, TF)
    assert len(w) == len(strategies.STRATEGIES) and (w["weight"] == 1).all() and (w["basis"] == "live").all()

    world.advance(5)
    r2 = world.cycle()
    assert not r2.verdicts_updated and not r2.retrained  # 24本経つまで・試した数が増えるまでは再計算しない


def test_signals_are_judged_only_by_models_trained_before_them(world):
    world.cycle()
    for _ in range(4):
        world.advance(30)
        world.cycle()
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    reg = meta_registry(world.con).set_index("version")
    assert len(reg) >= 4
    judged = sl[sl["meta_version"].notna()]
    assert len(judged) > 0
    trained_bar = reg.loc[judged["meta_version"], "trained_bar"].to_numpy()
    assert (judged["signal_time"].to_numpy() >= trained_bar).all()
    # 記録したシグナルは、すべて判定されている（取りこぼしなし）
    assert sl["meta_p"].notna().all()
    assert set(judged["meta_decision"]) <= {"乗る", "見送り"}
    th = reg["threshold"].iat[0]
    assert ((judged["meta_p"] >= th) == (judged["meta_decision"] == "乗る")).all()


def test_new_signals_are_judged_by_the_old_model_before_retraining(world):
    world.cycle()
    v1 = jobs.latest_meta(world.con, PAIR, TF)["version"]
    world.advance(30)
    assert world.cycle().retrained
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    v2 = jobs.latest_meta(world.con, PAIR, TF)
    before = sl[sl["signal_time"] < pd.Timestamp(v2["trained_bar"])]
    assert len(before) > 0 and (before["meta_version"] == v1).all()


def test_legacy_or_unjudged_model_is_retrained_immediately(world):
    world.cycle()
    world.con.execute("UPDATE model_registry SET eval_to=NULL")
    world.con.commit()
    world.advance(1)
    assert world.cycle().retrained


def test_more_trials_trigger_rededuction(world):
    """試した組み合わせが増えたら、戦略の判定と機械学習の条件を、新しい数で割り引き直す。"""
    world.cycle()
    cfg = trials.ml_config("rf", 48, 0.62, label="triple_barrier_meta", features="v2")
    trials.record(world.con, [{"family": "ml", "name": "rf", "purpose": "ml_eval_v2", "config": cfg}],
                  pair=PAIR, tf=TF, origin="dashboard", now=world.now)
    world.advance(1)
    r = world.cycle()
    assert r.verdicts_updated and r.retrained
    n = trials.count(world.con, PAIR, TF)
    assert (tracking.load_verdicts(world.con, PAIR, TF)["n_trials"] == n).all()
    assert meta_registry(world.con)["n_trials"].iat[-1] == n


def test_strategy_signals_logged_exactly_once(world):
    world.cycle()
    world.advance(200)
    world.cycle()
    world.cycle()  # 同じ時刻にもう一度：重複しない
    sl = tracking.load_signal_log(world.con, PAIR, TF)

    from fxtool import indicators
    ind = indicators.add_all(FULL.iloc[:START + 200], IndicatorParams())
    expected = 0
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, IndicatorParams()).to_numpy()
        expected += int(np.count_nonzero(sig[START - 1:START + 200]))
    assert len(sl) == expected and expected > 0
    assert sl["weight"].notna().all() and (sl["suppressed"] == 0).all()


def test_outcomes_are_filled_correctly(world):
    world.cycle()
    world.advance(300)
    world.cycle()
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    done = sl[sl["status"].isin(tracking.DONE)]
    assert len(done) > 0
    cost = jobs.default_costs(PAIR).round_trip
    want = done["direction"] * (done["exit_price"] - done["entry"]) / 0.01 - cost
    np.testing.assert_allclose(done["pips"], want)
    # 決済済みが出てくると、次の処理から、重みは本番の記録（各戦略の直近50回まで）で計算される
    world.advance(1)
    world.cycle()
    w = tracking.latest_weights(world.con, PAIR, TF).set_index("source")
    per = done.groupby("source").size()
    for src, n in per.items():
        assert w.loc[src, "recent_n"] == min(n, 50)


def test_max_loss_halts_new_signals(world):
    world.cycle()
    world.advance(100)
    # 今日（日本時間）すでに損切りが3回あった、という本番の記録を入れる（3R の損失）
    today = world.now - pd.Timedelta(minutes=10)
    for k in range(3):
        world.con.execute(
            """INSERT INTO signal_log (pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars, logged_at,
               status, exit_time, exit_price, pips, entry_basis)
               VALUES (?, ?, 'ma_cross', ?, 1, 150, 149.7, 150.6, 0.2, 48, ?, '損切り', ?, 149.7, -30.0, 'next_open')""",
            (PAIR, TF, f"2019-01-0{k + 1} 00:00:00", "2019-01-01 00:00:00", today.strftime("%Y-%m-%d %H:%M:%S")))
    world.con.commit()
    r = world.cycle()
    assert r.halted and "今日" in r.halted
    new = tracking.load_signal_log(world.con, PAIR, TF)
    new = new[new["signal_time"] > FULL.index[START - 1]]
    assert len(new) > 0 and (new["suppressed"] == 1).all()
    alerts = tracking.load_alerts(world.con, PAIR, TF, active_only=True)
    assert "max_loss" in set(alerts["kind"])


def test_throttle_and_job_runs(world):
    world.cycle()
    r = jobs.run_cycle(world.con, PAIR, TF, now=world.now)  # force なし・直後
    assert r.skipped
    runs = tracking.load_job_runs(world.con)
    assert len(runs) == 1 and runs["error"].isna().all()


def test_legacy_prediction_labels_are_still_filled(tmp_path):
    con = data.connect(tmp_path / "p.db")
    from fxtool import indicators
    ind = indicators.add_all(FULL.iloc[:300], IndicatorParams())
    t = FULL.index[100].strftime("%Y-%m-%d %H:%M:%S")
    con.execute("INSERT INTO prediction_log VALUES (?, ?, 'gbm', 1, ?, 0.6, 'シグナルなし', 'v0', ?, NULL)", (PAIR, TF, t, t))
    assert jobs.fill_labels(con, PAIR, TF, ind) == 1
    y = con.execute("SELECT y FROM prediction_log").fetchone()[0]
    assert y == float(FULL["close"].iat[101] > FULL["close"].iat[100])


# ---------------- 重み・最大損失・警告 ----------------

def test_weights():
    assert tracking.weight_from_z(0) == 1
    assert tracking.weight_from_z(10) == 1.75 and tracking.weight_from_z(-10) == 0.25
    assert tracking.z_score(np.array([1.0, 2.0])) == 0  # 取引が少ないときは中立
    assert tracking.weighted_score({"a": "買い", "b": "売り"}, {"a": 1.5, "b": 0.5}) == pytest.approx(0.5)
    assert tracking.weighted_score({"a": "様子見"}, {"a": 1.0}) == 0


def test_live_performance_weights():
    rng = np.random.default_rng(0)
    closed = pd.concat([
        pd.DataFrame({"source": "ma_cross", "exit_time": pd.date_range("2026-01-01", periods=60, freq="D"),
                      "pips": rng.normal(15, 10, 60)}),     # 本番で明らかに勝っている
        pd.DataFrame({"source": "rsi_reversal", "exit_time": pd.date_range("2026-01-01", periods=3, freq="D"),
                      "pips": [-50.0, -40.0, -60.0]}),       # 回数が少ないので重み1
    ])
    perf = {r.source: r for r in tracking.live_performance(closed, 50, 5)}
    assert perf["ma_cross"].n == 50  # 直近50回だけ
    assert perf["ma_cross"].weight > perf["rsi_reversal"].weight
    assert perf["rsi_reversal"].z == 0
    assert np.mean([r.weight for r in perf.values()]) == pytest.approx(1.0)


def test_loss_halt_uses_japan_time_day_and_week():
    def closed(times, pips):
        return pd.DataFrame({"exit_time": pd.to_datetime(times), "pips": pips, "atr": 0.2})  # 1R = 30 pips

    now = pd.Timestamp("2026-10-07 03:00")  # 水曜 日本時間12時
    h = tracking.loss_halt(closed(["2026-10-07 01:00", "2026-10-06 16:00"], [-30, -30]), now, 1.5, 0.01, 2, 5)
    assert h.active and h.day_r == pytest.approx(-2)  # 前日 UTC 16時 = 日本時間の今日 1時
    h2 = tracking.loss_halt(closed(["2026-10-06 14:00"], [-60]), now, 1.5, 0.01, 2, 5)
    assert not h2.active and h2.day_r == 0 and h2.week_r == pytest.approx(-2)  # 日本時間の昨日
    week = closed(["2026-10-05 01:00", "2026-10-06 01:00", "2026-10-06 02:00"], [-60, -60, -45])
    h3 = tracking.loss_halt(week, now, 1.5, 0.01, 2, 5)
    assert h3.active and "今週" in h3.reason and h3.week_r == pytest.approx(-5.5)
    last_week = closed(["2026-10-04 10:00"], [-300])  # 日曜 = 先週
    assert not tracking.loss_halt(last_week, now, 1.5, 0.01, 2, 5).active


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


def test_meta_conditions():
    reg = pd.DataFrame({"ok": [1, 0]})
    sig = pd.DataFrame({
        "status": ["利確"] * 40,
        "meta_decision": ["乗る"] * 20 + ["見送り"] * 20,
        "pips": [-10.0] * 20 + [10.0] * 20,  # 見送ったほうが成績が良い
    })
    kinds = {c.kind for c in tracking.meta_conditions(reg, sig)}
    assert kinds == {"lost_edge", "filter_hurts"}
    assert tracking.meta_conditions(reg.iloc[:1], sig.iloc[:10]) == []


def test_monthly():
    closed = pd.DataFrame({"source": ["a", "a", "b"], "pips": [1.0, 2.0, 5.0],
                           "exit_time": pd.to_datetime(["2026-01-31 16:00", "2026-01-31 14:00", "2026-02-02 00:00"])})
    m = tracking.monthly(closed)
    assert m.loc["2026-02", "a"] == 1.0 and m.loc["2026-01", "a"] == 2.0  # UTC 16時 = 日本時間の翌日
    assert m.loc["2026-02", "b"] == 5.0


def test_data_condition_ignores_weekend():
    bar = pd.Timedelta(hours=1)
    friday_close = pd.Timestamp("2026-10-02 20:00")
    assert tracking.data_condition(friday_close, pd.Timestamp("2026-10-03 12:00"), bar, 6) == []   # 土曜
    assert tracking.data_condition(friday_close, pd.Timestamp("2026-10-01 12:00"), bar, 6) == []  # 新しい
    assert len(tracking.data_condition(pd.Timestamp("2026-10-06 01:00"), pd.Timestamp("2026-10-06 12:00"), bar, 6)) == 1
