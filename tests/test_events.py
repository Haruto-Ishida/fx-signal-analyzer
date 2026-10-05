"""決まった時刻の予定（米雇用統計・FOMC・日銀会合など）のテスト。"""
import pandas as pd
import pytest

from fxtool import events

TS = pd.Timestamp


def test_us_and_uk_daylight_saving():
    # 2026年：米国の夏時間は 3/8〜11/1、英国の夏時間は 3/29〜10/25
    assert not events.us_dst(TS("2026-03-07")) and events.us_dst(TS("2026-03-08"))
    assert events.us_dst(TS("2026-10-31")) and not events.us_dst(TS("2026-11-01"))
    assert not events.uk_dst(TS("2026-03-28")) and events.uk_dst(TS("2026-03-29"))
    assert events.uk_dst(TS("2026-10-24")) and not events.uk_dst(TS("2026-10-25"))


def test_eastern_time_conversion():
    assert events.et_to_utc(TS("2026-07-01"), 8, 30) == TS("2026-07-01 12:30")  # 夏時間 UTC−4
    assert events.et_to_utc(TS("2026-12-01"), 8, 30) == TS("2026-12-01 13:30")  # 冬時間 UTC−5


def test_nonfarm_payrolls_first_friday_in_japan_time():
    assert events.first_friday(2026, 10) == TS("2026-10-02")
    assert events.first_friday(2026, 11) == TS("2026-11-06")
    evs = [e for e in events.recurring(TS("2026-10-01"), TS("2026-11-10")) if e.name == "米雇用統計"]
    assert [e.jst_label() for e in evs] == ["10/02（金）21:30", "11/06（金）22:30"]  # 夏時間の終わりで1時間ずれる


def test_meetings_from_schedule_file():
    sched = events.load_schedule()
    fomc = [e for e in events.meetings(sched) if "FOMC" in e.name]
    oct_fomc = [e for e in fomc if e.time.month == 10 and e.time.year == 2026][0]
    assert oct_fomc.time == TS("2026-10-28 18:00")  # 14:00 米東部（夏時間）
    assert oct_fomc.jst_label() == "10/29（木）03:00"
    boj = [e for e in events.meetings(sched) if "日銀" in e.name and e.time.year == 2026 and e.time.month == 10][0]
    assert boj.time == TS("2026-10-30 03:00") and not boj.time_fixed  # 日本時間の正午ごろ


def test_schedule_file_is_consistent():
    """日程ファイルの中身の確認：2日間の会合で、土日を含まない。"""
    s = events.load_schedule()
    assert set(s["bank"]) == {"FOMC", "BOJ"}
    assert ((s["end_date"] - s["start_date"]).dt.days == 1).all()
    assert (s["start_date"].dt.dayofweek < 5).all() and (s["end_date"].dt.dayofweek < 5).all()
    for bank, g in s.groupby("bank"):
        assert g["end_date"].is_monotonic_increasing and len(g) == 16  # 2026・2027年 各8回


def test_update_reminder():
    s = events.load_schedule()
    assert events.schedule_status(s, TS("2026-10-05")) == (False, "")
    need, msg = events.schedule_status(s, TS("2027-10-01"))  # 最後の会合（12月）まで90日を切る
    assert need and "central_bank_schedule.csv" in msg and "federalreserve.gov" in msg
    need2, msg2 = events.schedule_status(s[s["bank"] == "FOMC"], TS("2026-10-05"))
    assert need2 and "日銀" in msg2


def test_upcoming_only_relevant_currencies_and_near_state():
    sched = events.load_schedule()
    now = TS("2026-10-28 17:30")  # FOMC の30分前
    usdjpy = events.upcoming("USDJPY=X", now, 48, sched)
    eurgbp = events.upcoming("EURGBP=X", now, 48, sched)
    assert any("FOMC" in e.name for e in usdjpy) and not eurgbp
    fomc = [e for e in usdjpy if "FOMC" in e.name][0]
    assert events.near(fomc, now) == "直前"
    assert events.near(fomc, now + pd.Timedelta(minutes=45)) == "直後"
    assert events.near(fomc, now - pd.Timedelta(hours=3)) == ""
    assert any("日銀" in e.name for e in usdjpy)  # 10/30 の日銀会合も48時間以内
    assert not any(e.importance == "low" for e in usdjpy)
    assert any(e.name == "東京仲値" for e in events.upcoming("USDJPY=X", now, 48, sched, include_low=True))
