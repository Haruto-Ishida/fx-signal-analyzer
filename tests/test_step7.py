"""Step 7：キルスイッチ・経済指標フィルター・ペーパー口座。"""
import pandas as pd
import pytest

from fxtool import account, events, killswitch, paper
from fxtool.settings import EventFilter, KillSwitch, Settings

KS = KillSwitch()          # 5連敗・1日10回・1日3%・1週間6%
JST = pd.Timedelta(hours=9)


def tr(rows):
    """rows：(決済時刻 UTC, 損益 円)。"""
    return pd.DataFrame({"exit_time": pd.to_datetime([r[0] for r in rows]), "pnl_jpy": [r[1] for r in rows],
                         "units": [1000] * len(rows)})


def daily(rows):
    return pd.DataFrame({"balance": [r[1] for r in rows]}, index=pd.to_datetime([r[0] for r in rows]))


NOW = pd.Timestamp("2026-10-07 03:00")  # 日本時間 10/7（水）12:00


def test_daily_loss_trips_until_end_of_day_and_can_be_released():
    t = tr([("2026-10-07 01:00", -1500), ("2026-10-07 02:00", -1600)])     # 今日 −3,100円
    d = daily([("2026-10-06", 100_000), ("2026-10-07", 96_900)])
    s = killswitch.evaluate(t, d, 100_000, 0, NOW, KS)
    assert s.active and [x.kind for x in s.trips] == ["daily_loss"]
    assert s.day_loss_pct == pytest.approx(3.1)
    assert s.trips[0].until == pd.Timestamp("2026-10-07 15:00")             # 日本時間 10/8 0:00
    assert not killswitch.evaluate(t, d, 100_000, 0, pd.Timestamp("2026-10-07 15:00"), KS).active  # 翌日は自動で解除
    r = killswitch.evaluate(t, d, 100_000, 0, NOW, KS, {s.trips[0].key})
    assert not r.active and r.released[0].kind == "daily_loss"
    # 2.9% なら止まらない
    assert not killswitch.evaluate(tr([("2026-10-07 01:00", -2900)]), d, 100_000, 0, NOW, KS).active


def test_daily_loss_uses_balance_at_start_of_day():
    d = daily([("2026-10-05", 100_000), ("2026-10-06", 50_000)])  # 昨日の終わりに 5万円
    s = killswitch.evaluate(tr([("2026-10-07 01:00", -1600)]), d, 100_000, 0, NOW, KS)
    assert s.day_loss_pct == pytest.approx(3.2) and s.active


def test_weekly_loss_counts_from_monday_jst():
    # 日本時間の月曜 10/5 0:00 = UTC 10/4 15:00。その前の損失は数えない
    t = tr([("2026-10-04 14:00", -5000), ("2026-10-05 02:00", -2000), ("2026-10-06 02:00", -2000),
            ("2026-10-06 05:00", -2100)])
    d = daily([("2026-10-04", 95_000), ("2026-10-05", 93_000), ("2026-10-06", 88_900)])
    s = killswitch.evaluate(t, d, 100_000, 0, NOW, KS)
    assert s.week_loss_pct == pytest.approx(6100 / 95_000 * 100)
    assert [x.kind for x in s.trips] == ["weekly_loss"]
    assert s.trips[0].until == pd.Timestamp("2026-10-11 15:00")             # 日本時間 10/12（月）0:00


def test_consecutive_losses_trip_for_24h_then_reset():
    base = pd.Timestamp("2026-10-06 00:00")
    rows = [(base + pd.Timedelta(hours=i), -100) for i in range(5)]
    t = tr(rows)
    s = killswitch.evaluate(t, None, 100_000, 0, base + pd.Timedelta(hours=5), KS)
    assert [x.kind for x in s.trips] == ["consecutive"] and s.streak == 0
    assert s.trips[0].until == base + pd.Timedelta(hours=28)
    assert not killswitch.evaluate(t, None, 100_000, 0, base + pd.Timedelta(hours=28), KS).active
    # 4連敗のあとに勝ちがあれば数え直し
    t2 = tr(rows[:4] + [(base + pd.Timedelta(hours=4), 50)] + [(base + pd.Timedelta(hours=5 + i), -100) for i in range(4)])
    s2 = killswitch.evaluate(t2, None, 100_000, 0, base + pd.Timedelta(hours=10), KS)
    assert not s2.active and s2.streak == 4
    # 止まっている間に決済した負けは数えない（止まる前に入っていた取引）
    t3 = tr(rows + [(base + pd.Timedelta(hours=6 + i), -100) for i in range(5)])
    s3 = killswitch.evaluate(t3, None, 100_000, 0, base + pd.Timedelta(hours=29), KS)
    assert not s3.active and s3.streak == 0


def test_frequency_trips_at_the_limit():
    empty = pd.DataFrame(columns=["exit_time", "pnl_jpy", "units"])
    assert not killswitch.evaluate(empty, None, 100_000, 9, NOW, KS).active
    s = killswitch.evaluate(empty, None, 100_000, 10, NOW, KS)
    assert [x.kind for x in s.trips] == ["frequency"]


def test_settings_change_the_limits():
    t = tr([("2026-10-07 01:00", -1500)])
    assert killswitch.evaluate(t, None, 100_000, 0, NOW, KillSwitch(daily_loss_pct=1.0)).active
    assert not killswitch.evaluate(t, None, 100_000, 0, NOW, KS).active


# ---------------- 経済指標フィルター ----------------

def test_blackout_around_nfp_only_for_usd_pairs():
    nfp = events.et_to_utc(pd.Timestamp("2026-11-06"), 8, 30)  # 11月の第1金曜（米国は夏時間の終わった後）
    ef = EventFilter()
    assert events.blackout("USDJPY=X", nfp - pd.Timedelta(minutes=60), ef).name == "米雇用統計"
    assert events.blackout("USDJPY=X", nfp + pd.Timedelta(minutes=60), ef) is not None
    assert events.blackout("USDJPY=X", nfp - pd.Timedelta(minutes=61), ef) is None
    assert events.blackout("USDJPY=X", nfp + pd.Timedelta(minutes=61), ef) is None
    assert events.blackout("EURJPY=X", nfp, ef) is None                    # 米ドルを含まないペアは止めない
    assert events.blackout("USDJPY=X", nfp - pd.Timedelta(minutes=90), EventFilter(minutes_before=120)) is not None


def test_blackout_boj_uses_wider_window():
    sched = pd.DataFrame({"bank": ["BOJ"], "start_date": pd.to_datetime(["2026-10-29"]),
                          "end_date": pd.to_datetime(["2026-10-30"]), "source": [""], "checked_at": [""]})
    noon = events.jst_to_utc(pd.Timestamp("2026-10-30"), 12, 0)
    assert events.blackout("USDJPY=X", noon + pd.Timedelta(hours=2, minutes=59), schedule=sched) is not None
    assert events.blackout("USDJPY=X", noon + pd.Timedelta(hours=3, minutes=1), schedule=sched) is None
    assert events.blackout("EURUSD=X", noon, schedule=sched) is None


# ---------------- ペーパー口座・記録 ----------------

@pytest.fixture
def con(tmp_path):
    from fxtool import data
    c = data.connect(tmp_path / "p.db")
    yield c
    c.close()


def add_signal(con, pair, tf, source, t, d, entry, sl, pips, status="利確", exit_time=None, suppressed=0, meta=None,
               basis="next_open"):
    con.execute("""INSERT INTO signal_log (pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars,
                   logged_at, status, exit_time, pips, entry_basis, suppressed, meta_decision)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.1, 48, ?, ?, ?, ?, ?, ?, ?)""",
                (pair, tf, source, t, d, entry, sl, entry + 0.3 * d, t, status, exit_time, pips, basis, suppressed, meta))
    con.commit()


def test_paper_takes_only_eligible_signals_and_combines_pairs(con):
    add_signal(con, "USD/JPY", "1h", "ma_cross", "2026-10-05 01:00:00", 1, 150.0, 149.85, 30.0, exit_time="2026-10-05 05:00:00")
    add_signal(con, "EUR/USD", "1h", "ma_cross", "2026-10-05 02:00:00", -1, 1.10, 1.1015, -15.0, "損切り", "2026-10-05 06:00:00")
    add_signal(con, "USD/JPY", "1h", "rsi_reversal", "2026-10-05 03:00:00", 1, 150.0, 149.85, 30.0,
               exit_time="2026-10-05 07:00:00", suppressed=1)                      # 停止中 → 取引しない
    add_signal(con, "USD/JPY", "1h", "breakout", "2026-10-05 03:00:00", 1, 150.0, 149.85, 30.0,
               exit_time="2026-10-05 07:00:00", meta="見送り")                      # 機械学習が見送り → 取引しない
    add_signal(con, "USD/JPY", "1h", "ml_direction_h4", "2026-10-05 03:00:00", 1, 150.0, 149.85, 30.0,
               exit_time="2026-10-05 07:00:00")                                    # 機械学習の予測そのもの → 取引しない
    add_signal(con, "USD/JPY", "1d", "macd_cross", "2026-10-05 00:00:00", 1, 150.0, 149.0, 0.0, "進行中")  # 未決済
    t = paper.trades(con)
    assert sorted(t["pair"]) == ["EUR/USD", "USD/JPY"]
    assert (t["entry_time"] == pd.to_datetime(["2026-10-05 02:00", "2026-10-05 03:00"])).all()
    from dataclasses import replace
    wide = Settings(sizing=replace(Settings().sizing, max_leverage=25.0))
    acc = paper.build(con, wide, usdjpy=150.0)
    # USD/JPY：1% リスク＝1,000円、損切り幅 0.15円 → 6,666通貨 → 1,000通貨単位で 6,000通貨。+30pips = +1,800円
    # EUR/USD：残高 100,000（USD/JPY はまだ決済前）で 1,000円 ÷ (0.0015 × 150) = 4,444 → 4,000通貨。−15pips = −900円
    assert list(acc.trades.sort_values("entry_time")["units"]) == [6000, 4000]
    assert acc.metrics["final"] == pytest.approx(100_000 + 6_000 * 0.30 - 4_000 * 0.0015 * 150)
    # 初期設定（実効レバレッジ5倍まで＝50万円分）では、USD/JPY は 3,000通貨（45万円分）に減らし、
    # EUR/USD は残り5万円分では1,000通貨（16.5万円分）も持てないので見送る
    acc5 = paper.build(con, Settings(), usdjpy=150.0)
    t5 = acc5.trades.sort_values("entry_time")
    assert list(t5["units"]) == [3000, 0] and t5["note"].iloc[1].startswith("見送り")
    assert acc5.metrics["final"] == pytest.approx(100_000 + 3_000 * 0.30)
    # USD/JPY のレートがわからなければ、円のペアだけで計算する
    assert paper.build(con, Settings(), usdjpy=None).metrics["n"] == 1
    assert paper.signals_today(con, pd.Timestamp("2026-10-05 10:00")) == 3


def test_log_strategy_signals_marks_blackout_and_quota(con, monkeypatch):
    from fxtool import jobs, strategies, tracking
    idx = pd.date_range("2026-11-06 10:00", periods=6, freq="h")  # 13:30 UTC が米雇用統計
    ind = pd.DataFrame({"atr": [0.1] * 6}, index=idx)
    sig = pd.Series([1, 0, 0, 1, 0, 1], index=idx)
    monkeypatch.setattr(strategies, "STRATEGIES", strategies.STRATEGIES[:1])
    monkeypatch.setattr(strategies, "signals", lambda ind_, s, p: sig)
    tracking.set_state(con, "USD/JPY", "1h", "strategy_last_bar", "2026-11-06 09:00:00")
    # その日（日本時間 11/6）は、ほかのペアですでに 9 回出している → あと1回
    for k in range(9):
        add_signal(con, "EUR/JPY", "1h", "ma_cross", f"2026-11-06 0{k}:00:00", 1, 160.0, 159.8, 0.0, "進行中")
    n = jobs.log_strategy_signals(con, "USD/JPY", "1h", ind, {}, idx[-1], max_per_day=10, event_filter=EventFilter())
    assert n == 3
    rows = con.execute("""SELECT signal_time, suppressed, suppress_reason FROM signal_log WHERE pair='USD/JPY'
                          ORDER BY signal_time""").fetchall()
    # 10:00 の足（確定 11:00）は通常 → 10回目
    assert rows[0][:2] == ("2026-11-06 10:00:00", 0)
    # 13:00 の足（確定 14:00 ＝ 発表の30分後）は経済指標で停止
    assert rows[1][1] == 1 and rows[1][2].startswith("経済指標の前後（米雇用統計")
    # 15:00 の足（日本時間 11/7 0:00 ＝ 次の日）は、その日の1回目なので止めない
    assert rows[2][:2] == ("2026-11-06 15:00:00", 0)
    # 同じ日にもう1回あれば止まる。その日の停止を手動で解除すれば止まらない
    idx2 = pd.date_range("2026-11-06 16:00", periods=2, freq="h")
    ind2 = pd.DataFrame({"atr": [0.1, 0.1]}, index=idx2)
    monkeypatch.setattr(strategies, "signals", lambda ind_, s, p: pd.Series([0, 1], index=idx2))
    for k in range(9):
        add_signal(con, "EUR/JPY", "1h", "ma_cross", f"2026-11-06 {16 + k % 1}:{k:02d}:00", 1, 160.0, 159.8, 0.0, "進行中")
    jobs.log_strategy_signals(con, "USD/JPY", "1h", ind2, {}, idx2[-1], max_per_day=10)
    last = con.execute("SELECT suppressed, suppress_reason FROM signal_log WHERE pair='USD/JPY' ORDER BY signal_time DESC").fetchone()
    assert last[0] == 1 and "回数の上限" in last[1]
    con.execute("DELETE FROM signal_log WHERE pair='USD/JPY' AND signal_time='2026-11-06 17:00:00'")
    tracking.set_state(con, "USD/JPY", "1h", "strategy_last_bar", "2026-11-06 16:00:00")
    killswitch.release(con, "frequency:2026-11-07", idx2[-1])
    jobs.log_strategy_signals(con, "USD/JPY", "1h", ind2, {}, idx2[-1], max_per_day=10)
    last = con.execute("SELECT suppressed FROM signal_log WHERE pair='USD/JPY' ORDER BY signal_time DESC").fetchone()
    assert last[0] == 0


def test_loss_halt_reason_is_recorded(con, monkeypatch):
    from fxtool import jobs, strategies, tracking
    idx = pd.date_range("2026-10-06 00:00", periods=2, freq="h")
    ind = pd.DataFrame({"atr": [0.1, 0.1]}, index=idx)
    monkeypatch.setattr(strategies, "STRATEGIES", strategies.STRATEGIES[:1])
    monkeypatch.setattr(strategies, "signals", lambda ind_, s, p: pd.Series([0, -1], index=idx))
    jobs.log_strategy_signals(con, "USD/JPY", "1h", ind, {}, idx[-1], suppressed=True, reason="キルスイッチ：テスト")
    assert con.execute("SELECT suppressed, suppress_reason FROM signal_log").fetchone() == (1, "キルスイッチ：テスト")


def test_release_is_recorded(con):
    now = pd.Timestamp("2026-10-07 03:00")
    killswitch.release(con, "daily_loss:2026-10-07", now)
    assert killswitch.released_keys(con) == {"daily_loss:2026-10-07"}
