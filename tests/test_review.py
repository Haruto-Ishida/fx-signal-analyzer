"""2026-10-07：年1回の見直しの時期のお知らせ（画面と Discord）。"""
import json

import pandas as pd
import pytest

from fxtool import holdout, notify, review
from fxtool.settings import Review


@pytest.fixture
def hfile(tmp_path):
    p = tmp_path / "holdout.json"
    p.write_text(json.dumps({"boundaries": {"USD/JPY|1h": {
        "start": "2026-03-16 07:00:00", "fixed_at": "2026-10-05 10:00:00", "n_bars_at_fix": 17000, "touched": ""}}}),
        encoding="utf-8")
    return p


def test_status_ok_soon_overdue(hfile):
    r = Review()
    s = review.status("USD/JPY", "1h", pd.Timestamp("2027-08-01"), r, hfile)
    assert s.state == "ok" and s.due == pd.Timestamp("2027-10-05 10:00:00")
    s = review.status("USD/JPY", "1h", pd.Timestamp("2027-09-05 12:00"), r, hfile)
    assert s.state == "soon" and s.days_left == 30 and "あと 30 日" in s.message()
    s = review.status("USD/JPY", "1h", pd.Timestamp("2027-10-10"), r, hfile)
    assert s.state == "overdue" and "5 日過ぎています" in s.message()
    assert review.status("EUR/USD", "1h", pd.Timestamp("2027-10-10"), r, hfile) is None
    s = review.status("USD/JPY", "1h", pd.Timestamp("2027-04-01"), Review(interval_months=6, notice_days=10), hfile)
    assert s.state == "soon" and s.days_left == 4          # 6か月後（2027/04/05）の4日前


def test_review_reminder_is_sent_once_per_state(tmp_path, hfile, monkeypatch):
    from fxtool import data
    monkeypatch.setattr(holdout, "HOLDOUT_FILE", hfile)
    con = data.connect(tmp_path / "r.db")
    sent = []
    on = notify.Config(True, "https://example.invalid/webhook")
    soon = pd.Timestamp("2027-09-20 00:00")
    notify.notify_cycle(con, soon, on, lambda u, t: sent.append(t))
    notify.notify_cycle(con, soon + pd.Timedelta(hours=1), on, lambda u, t: sent.append(t))
    reviews = [t for t in sent if "見直し" in t]
    assert len(reviews) == 1 and "あと" in reviews[0]          # 近づいたときに1回だけ
    notify.notify_cycle(con, pd.Timestamp("2027-10-06 00:00"), on, lambda u, t: sent.append(t))
    reviews = [t for t in sent if "見直し" in t]
    assert len(reviews) == 2 and "過ぎています" in reviews[1]   # 過ぎたときにもう1回
    for t in reviews:
        notify.check_safe(t)
