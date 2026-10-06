"""論文ウォッチ（要件定義書 11章）：集めるのは自動、採用はユーザーの判断。

- arXiv の公式 API（https://export.arxiv.org/api/query）から、キーワードと分野（settings.toml の [papers]）に合う
  新しい論文の情報（題名・著者・日付・概要・リンク）を集め、ローカルの DB に保存する。重複（同じ論文の別の版も）は除く。
- 送るのは検索の条件だけで、個人情報や口座の情報は送らない。arXiv の利用ルールに合わせて、連続して問い合わせるときは3秒以上あける。
- 自動収集は、settings.toml の enabled = true にしたときだけ、週末の定期処理で週1回行う。画面のボタンでも集められる。
- 論文の内容を自動で売買のルールにすることはしない。印（未読／既読／評価したい／見送り）を付け、
  「評価したい」の論文はユーザーから依頼があったときに評価する。
"""
from __future__ import annotations

import re
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import pandas as pd

from .settings import Papers
from .tracking import JST, ts_str

API = "https://export.arxiv.org/api/query"
USER_AGENT = "fx-signal-tool/1.0 (personal research; paper watch)"
MIN_INTERVAL_SEC = 3.0
STATUSES = ("未読", "既読", "評価したい", "見送り")

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,       -- 版の番号（v2 など）を除いた ID
    title TEXT NOT NULL, authors TEXT NOT NULL, published TEXT NOT NULL, updated TEXT,
    summary TEXT NOT NULL, link TEXT NOT NULL, categories TEXT,
    fetched_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT '未読', note TEXT
);
CREATE TABLE IF NOT EXISTS paper_fetches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fetched_at TEXT NOT NULL, query TEXT NOT NULL,
    n_found INTEGER, n_new INTEGER, origin TEXT NOT NULL, error TEXT
);
"""
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: str
    published: str
    updated: str
    summary: str
    link: str
    categories: str


def ensure(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


def build_query(p: Papers) -> str:
    """(キーワードのどれか) AND (分野のどれか)。"""
    kw = " OR ".join(f'all:"{k}"' if " " in k else f"all:{k}" for k in p.keywords)
    cat = " OR ".join(f"cat:{c}" for c in p.categories)
    return f"({kw}) AND ({cat})" if cat else f"({kw})"


def build_url(p: Papers, start: int = 0) -> str:
    params = {"search_query": build_query(p), "start": start, "max_results": p.max_results,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    return API + "?" + urllib.parse.urlencode(params)


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        raw_id = _clean(e.findtext("a:id", "", NS))
        m = re.search(r"abs/(.+?)(v\d+)?$", raw_id)
        if not m:
            continue
        link = next((l.get("href") for l in e.findall("a:link", NS) if l.get("rel") == "alternate"), raw_id)
        out.append(Paper(
            arxiv_id=m.group(1), title=_clean(e.findtext("a:title", "", NS)),
            authors=", ".join(_clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)),
            published=_clean(e.findtext("a:published", "", NS)), updated=_clean(e.findtext("a:updated", "", NS)),
            summary=_clean(e.findtext("a:summary", "", NS)), link=link,
            categories=", ".join(c.get("term", "") for c in e.findall("a:category", NS)),
        ))
    return out


def save(con: sqlite3.Connection, papers: list[Paper], now: pd.Timestamp) -> int:
    """新しい論文だけを保存する（すでにある論文は、印やメモを残したまま、題名などを新しい版に更新する）。戻り値：新しい件数。"""
    ensure(con)
    have = {r[0] for r in con.execute("SELECT arxiv_id FROM papers")}
    new = 0
    for p in {p.arxiv_id: p for p in papers}.values():
        if p.arxiv_id in have:
            con.execute("UPDATE papers SET title=?, authors=?, updated=?, summary=?, link=?, categories=? WHERE arxiv_id=?",
                        (p.title, p.authors, p.updated, p.summary, p.link, p.categories, p.arxiv_id))
        else:
            con.execute("""INSERT INTO papers (arxiv_id, title, authors, published, updated, summary, link, categories,
                           fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (p.arxiv_id, p.title, p.authors, p.published, p.updated, p.summary, p.link, p.categories,
                         ts_str(now)))
            new += 1
    con.commit()
    return new


_last_call = [0.0]


def download(url: str, timeout: float = 90.0) -> str:
    """arXiv に問い合わせる（外部への接続）。前回から3秒以上あける。
    条件が多い検索は arXiv 側で時間がかかるので、待ち時間は長め（90秒）。混んでいると 429（問い合わせが多すぎる）が返るので、
    そのときは時間をおいてからやり直す（すぐに何度もやり直さない）。"""
    wait = MIN_INTERVAL_SEC - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")
    finally:
        _last_call[0] = time.monotonic()


def fetch(con: sqlite3.Connection, p: Papers, now: pd.Timestamp, origin: str = "dashboard", downloader=download) -> tuple[int, int]:
    """集めて保存する。戻り値：（見つかった件数, 新しい件数）。失敗しても記録を残して例外を出す。"""
    ensure(con)
    url = build_url(p)
    try:
        found = parse_atom(downloader(url))
        n_new = save(con, found, now)
    except Exception as e:
        con.execute("INSERT INTO paper_fetches (fetched_at, query, origin, error) VALUES (?, ?, ?, ?)",
                    (ts_str(now), build_query(p), origin, f"{type(e).__name__}: {e}"))
        con.commit()
        raise
    con.execute("INSERT INTO paper_fetches (fetched_at, query, n_found, n_new, origin) VALUES (?, ?, ?, ?, ?)",
                (ts_str(now), build_query(p), len(found), n_new, origin))
    con.commit()
    return len(found), n_new


def weekly_due(con: sqlite3.Connection, p: Papers, now: pd.Timestamp) -> bool:
    """有効にしていて、日本時間の週末で、前回の収集（成功）から6日以上たっているとき。"""
    if not p.enabled or (now + JST).weekday() < 5:
        return False
    ensure(con)
    last = con.execute("SELECT MAX(fetched_at) FROM paper_fetches WHERE error IS NULL").fetchone()[0]
    return last is None or now - pd.Timestamp(last) >= pd.Timedelta(days=6)


def load(con: sqlite3.Connection, keyword: str = "", statuses: tuple[str, ...] = STATUSES) -> pd.DataFrame:
    ensure(con)
    df = pd.read_sql_query(
        f"SELECT * FROM papers WHERE status IN ({','.join('?' * len(statuses))}) ORDER BY published DESC",
        con, params=statuses)
    if keyword:
        k = keyword.lower()
        df = df[df["title"].str.lower().str.contains(k, regex=False) | df["summary"].str.lower().str.contains(k, regex=False)]
    return df.reset_index(drop=True)


def mark(con: sqlite3.Connection, arxiv_id: str, status: str, note: str | None = None) -> None:
    if status not in STATUSES:
        raise ValueError(status)
    ensure(con)
    con.execute("UPDATE papers SET status=?, note=COALESCE(?, note) WHERE arxiv_id=?", (status, note, arxiv_id))
    con.commit()


def unread_count(con: sqlite3.Connection) -> int:
    ensure(con)
    return int(con.execute("SELECT COUNT(*) FROM papers WHERE status='未読'").fetchone()[0])
