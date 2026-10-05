"""経済指標・金融政策などの予定（研究ノート 6：重要な発表の前後や週明けの窓開けには注意を出す）。

外部の経済指標カレンダーは使わず、決まった発表時刻だけで判断する（ユーザーの決定）。
- 米雇用統計：原則として毎月第1金曜 8:30（米東部時間）。祝日などで日付がずれることがある。
- 米国の経済指標が多い時刻：平日の 8:30・10:00（米東部時間）。
- 東京仲値：平日 9:55（日本時間）。円の売買が集中しやすい。
- FOMC：会合の最終日 14:00（米東部時間）に声明、14:30 から議長の会見。
- 日銀の金融政策決定会合：最終日の昼ごろ（時刻は決まっていない）に結果、15:30 ごろから総裁の会見。
FOMC と日銀の日程は、年に1回 central_bank_schedule.csv にまとめて登録する（公式サイトで確認すること）。
米国の夏時間（3月第2日曜〜11月第1日曜）と英国の夏時間は自動で計算する。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ROOT

CALENDAR_FILE = ROOT / "central_bank_schedule.csv"
JST = pd.Timedelta(hours=9)
UPDATE_MARGIN_DAYS = 90  # 登録されている最後の会合がこれより近づいたら、日程ファイルの更新を知らせる
OFFICIAL = {
    "FOMC": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "BOJ": "https://www.boj.or.jp/mopo/mpmsche_minu/index.htm",
}
BANK_NAMES = {"FOMC": "FOMC（米国の金融政策）", "BOJ": "日銀の金融政策決定会合"}


@dataclass(frozen=True)
class Event:
    name: str
    time: pd.Timestamp          # UTC。時刻が決まっていない場合はおおよその時刻
    currencies: frozenset[str]  # 影響しやすい通貨
    importance: str             # "high" / "medium" / "low"
    time_fixed: bool = True     # False なら時刻はおおよそ
    note: str = ""

    def jst_label(self) -> str:
        t = self.time + JST
        wd = "月火水木金土日"[t.dayofweek]
        return f"{t:%m/%d}（{wd}）{'' if self.time_fixed else 'ごろ '}{t:%H:%M}"


# ---------------- 夏時間 ----------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """その月の第 n 回目の曜日（weekday: 月=0 … 日=6）。n=-1 なら最後。"""
    if n > 0:
        first = pd.Timestamp(year, month, 1)
        return first + pd.Timedelta(days=(weekday - first.dayofweek) % 7 + 7 * (n - 1))
    last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.dayofweek - weekday) % 7)


def us_dst(day: pd.Timestamp) -> bool:
    """米国の夏時間か（3月第2日曜〜11月第1日曜。日付で判断）。"""
    d = pd.Timestamp(day).normalize()
    return _nth_weekday(d.year, 3, 6, 2) <= d < _nth_weekday(d.year, 11, 6, 1)


def uk_dst(day: pd.Timestamp) -> bool:
    """英国の夏時間か（3月最終日曜〜10月最終日曜）。"""
    d = pd.Timestamp(day).normalize()
    return _nth_weekday(d.year, 3, 6, -1) <= d < _nth_weekday(d.year, 10, 6, -1)


def et_to_utc(day: pd.Timestamp, hour: int, minute: int = 0) -> pd.Timestamp:
    """米東部時間の時刻を UTC に直す（夏時間は UTC−4、冬時間は UTC−5）。"""
    d = pd.Timestamp(day).normalize()
    return d + pd.Timedelta(hours=hour + (4 if us_dst(d) else 5), minutes=minute)


def jst_to_utc(day: pd.Timestamp, hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(day).normalize() + pd.Timedelta(hours=hour, minutes=minute) - JST


# ---------------- 決まった時刻の予定 ----------------

def first_friday(year: int, month: int) -> pd.Timestamp:
    return _nth_weekday(year, month, 4, 1)


def recurring(start: pd.Timestamp, end: pd.Timestamp) -> list[Event]:
    """[start, end)（UTC）にある、決まった時刻の予定。"""
    out = []
    days = pd.date_range((start - pd.Timedelta(days=2)).normalize(), end.normalize() + pd.Timedelta(days=1), freq="D")
    for d in days:
        if d.dayofweek >= 5:
            continue
        if d == first_friday(d.year, d.month):
            out.append(Event("米雇用統計", et_to_utc(d, 8, 30), frozenset({"USD"}), "high",
                             note="原則として毎月第1金曜（祝日などでずれることがあります）"))
        out.append(Event("米国の経済指標が多い時刻", et_to_utc(d, 8, 30), frozenset({"USD"}), "low"))
        out.append(Event("米国の経済指標が多い時刻", et_to_utc(d, 10, 0), frozenset({"USD"}), "low"))
        jd = d  # 日本の平日（祝日は考慮しない）
        out.append(Event("東京仲値", jst_to_utc(jd, 9, 55), frozenset({"JPY"}), "low",
                         note="円の売買が集中しやすい時刻"))
    return sorted((e for e in out if start <= e.time < end), key=lambda e: e.time)


# ---------------- FOMC・日銀 ----------------

def load_schedule(path: Path | None = None) -> pd.DataFrame:
    path = path or CALENDAR_FILE
    if not path.exists():
        return pd.DataFrame(columns=["bank", "start_date", "end_date", "source", "checked_at"])
    return pd.read_csv(path, parse_dates=["start_date", "end_date"])


def meetings(schedule: pd.DataFrame) -> list[Event]:
    out = []
    for r in schedule.itertuples():
        if r.bank == "FOMC":
            out.append(Event("FOMC（米国の金融政策）の結果発表", et_to_utc(r.end_date, 14, 0), frozenset({"USD"}), "high",
                             note="14:30（米東部時間）から議長の会見"))
        elif r.bank == "BOJ":
            out.append(Event("日銀の金融政策決定会合の結果発表", jst_to_utc(r.end_date, 12, 0), frozenset({"JPY"}), "high",
                             time_fixed=False, note="時刻は決まっていません（昼ごろが多い）。15:30ごろから総裁の会見"))
    return out


def schedule_status(schedule: pd.DataFrame, today: pd.Timestamp) -> tuple[bool, str]:
    """日程ファイルの更新が必要か。登録されている最後の会合が UPDATE_MARGIN_DAYS 日以内に近づいたら知らせる。"""
    today = pd.Timestamp(today).normalize()
    msgs = []
    for bank, name in BANK_NAMES.items():
        g = schedule[schedule["bank"] == bank]
        if g.empty:
            msgs.append(f"{name}の日程が登録されていません")
            continue
        last = g["end_date"].max()
        if last < today + pd.Timedelta(days=UPDATE_MARGIN_DAYS):
            msgs.append(f"{name}の日程は {last:%Y/%m/%d} までしか登録されていません")
    if not msgs:
        return False, ""
    return True, ("、".join(msgs) + "。公式サイト（FOMC：" + OFFICIAL["FOMC"] + "、日銀：" + OFFICIAL["BOJ"]
                  + "）で翌年の日程を確認し、central_bank_schedule.csv に追加してください。")


# ---------------- まとめ ----------------

def pair_currencies(ticker: str) -> frozenset[str]:
    t = ticker.replace("=X", "").upper()
    return frozenset({t[:3], t[3:6]})


def upcoming(ticker: str, now: pd.Timestamp, hours: float = 48, schedule: pd.DataFrame | None = None,
             include_low: bool = False) -> list[Event]:
    """その通貨ペアに関係する、now の1時間前から hours 時間後までの予定（UTC）。"""
    start, end = now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=hours)
    sched = load_schedule() if schedule is None else schedule
    evs = recurring(start, end) + [e for e in meetings(sched) if start <= e.time < end]
    cur = pair_currencies(ticker)
    evs = [e for e in evs if e.currencies & cur and (include_low or e.importance != "low")]
    return sorted(evs, key=lambda e: e.time)


def blackout(ticker: str, t: pd.Timestamp, ef=None, schedule: pd.DataFrame | None = None) -> Event | None:
    """経済指標フィルター（要件定義書 8章の2）：時刻 t（UTC）が、その通貨ペアに関係する重要な予定の
    前後（settings.toml の [event_filter]。時刻が決まっていない日銀会合は前後 unfixed_window_minutes）なら、その予定。"""
    from .settings import EventFilter
    ef = ef or EventFilter()
    span = pd.Timedelta(minutes=max(ef.minutes_before, ef.minutes_after, ef.unfixed_window_minutes))
    sched = load_schedule() if schedule is None else schedule
    evs = [e for e in recurring(t - span, t + span + pd.Timedelta(minutes=1)) if e.importance == "high"]
    evs += [e for e in meetings(sched) if t - span <= e.time <= t + span]
    cur = pair_currencies(ticker)
    for e in sorted(evs, key=lambda e: e.time):
        if not e.currencies & cur:
            continue
        if e.time_fixed:
            lo, hi = e.time - pd.Timedelta(minutes=ef.minutes_before), e.time + pd.Timedelta(minutes=ef.minutes_after)
        else:
            w = pd.Timedelta(minutes=ef.unfixed_window_minutes)
            lo, hi = e.time - w, e.time + w
        if lo <= t <= hi:
            return e
    return None


def near(e: Event, now: pd.Timestamp) -> str:
    """発表の前後の状況（"直前" / "直後" / ""）。時刻が決まっていない予定は前後3時間を広めにとる。"""
    before, after = (pd.Timedelta(hours=3), pd.Timedelta(hours=3)) if not e.time_fixed else \
        (pd.Timedelta(minutes=60), pd.Timedelta(minutes=60))
    if e.time - before <= now < e.time:
        return "直前"
    if e.time <= now < e.time + after:
        return "直後"
    return ""
