"""Step 8：バージョン管理・巻き戻し・チャレンジャー（シャドーモードと承認制）。"""
import numpy as np
import pandas as pd
import pytest

from fxtool import jobs, mlops, tracking, trials
from test_jobs import PAIR, START, TF, World


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(mlops, "MODELS_DIR", tmp_path / "data" / "models")
    monkeypatch.setattr(mlops, "ROOT", tmp_path)
    return World(tmp_path, monkeypatch, start_bars=START)


def test_rollback_pins_old_version_and_stops_retraining(world):
    world.cycle()
    v1 = jobs.latest_meta(world.con, PAIR, TF)["version"]
    world.advance(30)
    assert world.cycle().retrained
    v2 = jobs.latest_meta(world.con, PAIR, TF)["version"]
    assert v2 != v1 and len(mlops.versions(world.con, PAIR, TF)) == 2
    mlops.rollback(world.con, PAIR, TF, v1, world.now)
    assert jobs.latest_meta(world.con, PAIR, TF)["version"] == v1
    world.advance(30)
    r = world.cycle()
    assert not r.retrained                                         # 固定中は自動で再学習しない
    sl = tracking.load_signal_log(world.con, PAIR, TF)
    new = sl[sl["signal_time"] > world.now - pd.Timedelta(hours=30)]
    assert len(new) and (new["meta_version"].dropna() == v1).all()  # 新しいシグナルは戻したバージョンで判定
    alerts = tracking.load_alerts(world.con, PAIR, TF, active_only=True)
    assert "pinned" in set(alerts["kind"])
    mlops.unpin(world.con, PAIR, TF)
    world.advance(1)
    assert world.cycle().retrained                                 # 解除したら再開
    with pytest.raises(ValueError):
        mlops.rollback(world.con, PAIR, TF, "no_such_version", world.now)


def test_challenger_runs_in_shadow_without_touching_production(world):
    world.cycle()
    n0 = trials.count(world.con, PAIR, TF)
    cid = mlops.register(world.con, PAIR, TF, "fracdiff", world.now)
    assert trials.count(world.con, PAIR, TF) == n0 + 1            # チャレンジャーの設定が1つ増える
    with pytest.raises(ValueError):
        mlops.register(world.con, PAIR, TF, "fracdiff", world.now)  # 同じものを二重に始めない
    with pytest.raises(ValueError):
        mlops.register(world.con, PAIR, TF, "base", world.now)      # 本番と同じ設定
    champ = jobs.latest_meta(world.con, PAIR, TF)["version"]
    for _ in range(3):
        world.advance(30)
        world.cycle()
    j = pd.read_sql_query("SELECT * FROM shadow_judgments WHERE challenger_id=?", world.con, params=(cid,))
    assert len(j) > 0 and set(j["decision"]) <= {"乗る", "見送り"}
    sl = tracking.load_signal_log(world.con, PAIR, TF).set_index("id")
    # 判定したのは、シャドーモードを始めた後のシグナルだけ
    created = pd.Timestamp(mlops.challengers(world.con, PAIR, TF)["created_at"].iat[0])
    assert (sl.loc[j["signal_id"], "signal_time"] >= created).all()
    # 本番の判定（meta_version）はチャンピオンのモデルのまま（チャレンジャーのモデルは使わない）
    assert not sl["meta_version"].dropna().str.contains("challenger").any()
    assert (mlops.versions(world.con, PAIR, TF)["features"].fillna("v2") == "v2").all()
    assert mlops.current_spec(world.con, PAIR, TF).name == "base"
    # 4週間たっていないので、提案中にはならない
    assert mlops.challengers(world.con, PAIR, TF)["status"].iat[0] == "shadow"
    cmp = mlops.compare(world.con, cid, world.now)
    assert not cmp.ready and not cmp.checks[0][1]


def test_approval_switches_spec_and_next_cycle_trains_new_features(world):
    world.cycle()
    cid = mlops.register(world.con, PAIR, TF, "fracdiff", world.now)
    with pytest.raises(ValueError):
        mlops.approve(world.con, PAIR, TF, cid, world.now)          # 提案中でなければ承認できない
    world.con.execute("UPDATE challengers SET status='ready' WHERE id=?", (cid,))
    mlops.approve(world.con, PAIR, TF, cid, world.now)
    assert mlops.current_spec(world.con, PAIR, TF).name == "fracdiff"
    world.advance(1)
    r = world.cycle()
    assert r.retrained
    reg = jobs.latest_meta(world.con, PAIR, TF)
    assert reg["features"] == "v2+fd" and "_fd_" in reg["version"]
    # 新しい特徴量の検証は、新しい設定として試行に記録される
    rows = world.con.execute("SELECT config FROM trials WHERE config LIKE '%v2+fd%'").fetchall()
    assert len(rows) >= 2
    assert mlops.last_swap(world.con, PAIR, TF) == world.now - pd.Timedelta(hours=1)


def test_compare_conditions():
    """比べ方：同じシグナルに対する「乗る／見送り」。差の検定・ドローダウン・期間・件数。"""
    import sqlite3
    from fxtool import data
    con = data.connect(__import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "c.db")
    now = pd.Timestamp("2026-11-30")
    con.execute("""INSERT INTO challengers (id, pair, tf, spec, kind, status, created_at)
                   VALUES (1, 'USD/JPY', '1h', 'fracdiff', 'logistic', 'shadow', '2026-10-01 00:00:00')""")
    rng = np.random.default_rng(0)
    for k in range(80):
        win = k % 2 == 0
        pips = 30.0 if win else -15.0
        champ = "乗る" if k % 4 < 2 else "見送り"          # 勝ち負けと関係なく半分に乗る
        chal = "乗る" if win else "見送り"                  # 勝つものだけに乗る（良いチャレンジャー）
        t = (pd.Timestamp("2026-10-02") + pd.Timedelta(hours=6 * k)).strftime("%Y-%m-%d %H:%M:%S")
        con.execute("""INSERT INTO signal_log (id, pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars,
                       logged_at, status, exit_time, pips, meta_decision)
                       VALUES (?, 'USD/JPY', '1h', 'ma_cross', ?, 1, 150, 149.85, 150.3, 0.1, 48, ?, '利確', ?, ?, ?)""",
                    (k + 1, t, t, t, pips, champ))
        con.execute("INSERT INTO shadow_judgments VALUES (1, ?, 0.5, ?, 'v', ?)", (k + 1, chal, t))
    cmp = mlops.compare(con, 1, now, dev_ok=True, last_swap=None)
    assert cmp.n == 80 and cmp.taken_chal == 40 and cmp.taken_champ == 40
    assert cmp.sr_chal > cmp.sr_champ and cmp.p_value < 0.05 and cmp.dd_chal == 0.0
    assert cmp.ready
    assert not mlops.compare(con, 1, pd.Timestamp("2026-10-20"), dev_ok=True).ready      # 28日未満
    assert not mlops.compare(con, 1, now, dev_ok=False).ready                           # 開発用データで不合格
    assert not mlops.compare(con, 1, now, dev_ok=True, last_swap=now - pd.Timedelta(days=10)).ready  # 月1回まで
    assert mlops.update_status(con, "USD/JPY", "1h", 1, cmp, now)
    assert con.execute("SELECT status FROM challengers WHERE id=1").fetchone()[0] == "ready"


def test_bootstrap_p_value():
    rng = np.random.default_rng(1)
    assert mlops.bootstrap_p(rng.normal(1.0, 1.0, 200)) < 0.01
    assert mlops.bootstrap_p(rng.normal(-1.0, 1.0, 200)) > 0.99
    assert mlops.bootstrap_p(np.array([])) == 1.0


def test_schedule_gap(tmp_path):
    from fxtool import data
    con = data.connect(tmp_path / "g.db")
    now = pd.Timestamp("2026-10-07 03:00")
    assert jobs.schedule_gap(con, now) is None                     # 定期実行の記録がない
    con.execute("INSERT INTO job_runs (pair, tf, origin, started_at) VALUES ('USD/JPY', '1h', 'schedule', '2026-10-07 01:05:00')")
    assert jobs.schedule_gap(con, now) is None                     # 2時間前なら正常
    con.execute("INSERT INTO job_runs (pair, tf, origin, started_at) VALUES ('USD/JPY', '1h', 'dashboard', '2026-10-07 02:59:00')")
    assert jobs.schedule_gap(con, now + pd.Timedelta(hours=2)) == pd.Timedelta(hours=3, minutes=55)  # 画面の処理は数えない


def test_strategy_challenger_is_logged_in_shadow_only(world):
    """戦略のチャレンジャー：出ていたはずのシグナルを shadow=1 で記録・答え合わせするだけで、
    本番の記録・ペーパー口座・機械学習の判定・通知には使わない。試行数は増えない。"""
    from fxtool import ga, notify, paper
    world.cycle()
    n0 = trials.count(world.con, PAIR, TF)
    genes = ga.genes_of((0, 0, 0))   # SMA10 が SMA50 を抜ける（テスト用に、シグナルがよく出るもの）
    cid = mlops.register_strategy(world.con, PAIR, TF, genes, world.now)
    with pytest.raises(ValueError):
        mlops.register_strategy(world.con, PAIR, TF, genes, world.now)   # 二重には始めない
    assert trials.count(world.con, PAIR, TF) == n0
    for _ in range(6):
        world.advance(48)
        world.cycle()
    key = ga.key_of((0, 0, 0))
    allrows = tracking.load_signal_log(world.con, PAIR, TF, include_shadow=True)
    sh = allrows[allrows["source"] == key]
    assert len(sh) > 0 and (sh["shadow"] == 1).all()
    created = pd.Timestamp(mlops.challengers(world.con, PAIR, TF)["created_at"].iat[0])
    assert (sh["signal_time"] >= created - pd.Timedelta(hours=2)).all()     # 過去にさかのぼって作らない
    assert sh["status"].isin(tracking.DONE).any()                           # 本番と同じく答え合わせされる
    assert sh["meta_p"].isna().all()                                        # 機械学習の判定には使わない
    prod = tracking.load_signal_log(world.con, PAIR, TF)
    assert key not in set(prod["source"])                                   # 本番の記録には出ない
    assert key not in set(paper.eligible(world.con)["source"])              # ペーパー口座では取引しない
    assert key not in set(notify.pending_signals(world.con, pd.Timestamp("2000-01-01"))["source"])  # 通知しない
    cmp = mlops.compare_strategy(world.con, cid, world.now)
    assert cmp.n == int(sh["status"].isin(tracking.DONE).sum()) and not cmp.ready   # 28日たっていない
    assert "戦略：" in mlops.strategy_label(mlops.challengers(world.con, PAIR, TF).iloc[0])
