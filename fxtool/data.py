"""価格データの取得（Yahoo Finance）とローカル保存（SQLite）。

- 時刻はすべて UTC（タイムゾーンなし）で保存する。
- 2回目以降は、保存済みの最後の足から現在までをカバーする最短の期間だけ取得する。
- 同じ足を取り直した場合は上書きする（形成中だった足が確定値に置き換わる）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import holdout
from .config import DB_PATH, PERIOD_DAYS, TIMEFRAMES, Timeframe

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlc (
    pair TEXT NOT NULL,
    tf   TEXT NOT NULL,
    ts   TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    PRIMARY KEY (pair, tf, ts)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    pair TEXT NOT NULL,
    tf   TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    period TEXT NOT NULL,
    rows INTEGER NOT NULL,
    PRIMARY KEY (pair, tf)
);
"""

TS_FMT = "%Y-%m-%d %H:%M:%S"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    from .tracking import SCHEMA as TRACKING_SCHEMA
    from .tracking import migrate

    path.parent.mkdir(parents=True, exist_ok=True)
    # 定期実行と画面が同時に書き込むことがあるので、ロック待ちを長めにする
    con = sqlite3.connect(path, timeout=30)
    con.executescript(SCHEMA + TRACKING_SCHEMA)
    migrate(con)
    return con


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance の生データを open/high/low/close・UTC の形にそろえ、明らかな異常値を除く。"""
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].rename(columns=str.lower).dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "ts"
    ok = (df > 0).all(axis=1) & (df["high"] >= df["low"])
    df = df[ok]
    return df[~df.index.duplicated(keep="last")].sort_index()


def download(ticker: str, tf: Timeframe, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, interval=tf.interval, period=period, progress=False, auto_adjust=False)
    return clean(raw)


def save(con: sqlite3.Connection, pair: str, tf_key: str, df: pd.DataFrame) -> int:
    """保存して、新しく増えた足の本数を返す。"""
    before = count(con, pair, tf_key)
    rows = [
        (pair, tf_key, ts.strftime(TS_FMT), float(r.open), float(r.high), float(r.low), float(r.close))
        for ts, r in df.iterrows()
    ]
    con.executemany("INSERT OR REPLACE INTO ohlc VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    return count(con, pair, tf_key) - before


def load(con: sqlite3.Connection, pair: str, tf_key: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close FROM ohlc WHERE pair = ? AND tf = ? ORDER BY ts",
        con, params=(pair, tf_key), parse_dates=["ts"], index_col="ts",
    )
    return df


def latest_close(con: sqlite3.Connection, pair: str) -> float | None:
    """保存済みの最新の終値（時間足を問わない）。"""
    row = con.execute("SELECT close FROM ohlc WHERE pair = ? ORDER BY ts DESC LIMIT 1", (pair,)).fetchone()
    return float(row[0]) if row else None


def count(con: sqlite3.Connection, pair: str, tf_key: str) -> int:
    return con.execute("SELECT COUNT(*) FROM ohlc WHERE pair = ? AND tf = ?", (pair, tf_key)).fetchone()[0]


def last_fetch(con: sqlite3.Connection, pair: str, tf_key: str) -> pd.Timestamp | None:
    row = con.execute(
        "SELECT fetched_at FROM fetch_log WHERE pair = ? AND tf = ?", (pair, tf_key)
    ).fetchone()
    return pd.Timestamp(row[0]) if row else None


def choose_period(tf: Timeframe, last_ts: pd.Timestamp | None, now: pd.Timestamp) -> str:
    """保存済みの最後の足から現在までを（余裕を持って）カバーする最短の期間を選ぶ。"""
    if last_ts is None:
        return tf.periods[-1]
    gap_days = (now - last_ts) / pd.Timedelta(days=1) + 2
    for p in tf.periods:
        if PERIOD_DAYS.get(p, float("inf")) >= gap_days:
            return p
    return tf.periods[-1]


@dataclass
class UpdateResult:
    period: str | None
    fetched: int
    added: int
    error: str | None = None


def update(con: sqlite3.Connection, pair: str, ticker: str, tf_key: str, force: bool = False) -> UpdateResult:
    """必要なら取得して保存する。最終取得から stale_after 経っていなければ何もしない（force 時を除く）。"""
    tf = TIMEFRAMES[tf_key]
    now = now_utc()
    fetched_at = last_fetch(con, pair, tf_key)
    if not force and fetched_at is not None and now - fetched_at < tf.stale_after:
        return UpdateResult(None, 0, 0)

    last = con.execute("SELECT MAX(ts) FROM ohlc WHERE pair = ? AND tf = ?", (pair, tf_key)).fetchone()[0]
    period = choose_period(tf, pd.Timestamp(last) if last else None, now)
    try:
        df = download(ticker, tf, period)
    except Exception as e:  # ネットワークエラーなど。保存済みデータで動き続ける
        return UpdateResult(period, 0, 0, f"{type(e).__name__}: {e}")
    if df.empty:
        return UpdateResult(period, 0, 0, "データを取得できませんでした（市場の休場中、または接続の問題）")

    added = save(con, pair, tf_key, df)
    con.execute(
        "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?, ?)",
        (pair, tf_key, now.strftime(TS_FMT), period, len(df)),
    )
    con.commit()
    return UpdateResult(period, len(df), added)


def confirmed(df: pd.DataFrame, tf: Timeframe, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """確定済みの足だけを返す（足の開始時刻 + 足の長さ <= 現在時刻）。"""
    now = now_utc() if now is None else now
    return df[df.index + tf.bar <= now]


# ---------------- 最終テスト期間（holdout.py） ----------------

def boundary(con: sqlite3.Connection, pair: str, tf_key: str, now: pd.Timestamp | None = None,
             path: Path | None = None) -> holdout.Boundary:
    """最終テスト期間の境界。まだ決めていなければ、いま保存済みの確定足の直近20%で固定する。"""
    b = holdout.get(pair, tf_key, path)
    if b is not None:
        return b
    now = now_utc() if now is None else now
    conf = confirmed(load(con, pair, tf_key), TIMEFRAMES[tf_key], now)
    touched = holdout.TOUCHED_USDJPY if pair == "USD/JPY" else holdout.TOUCHED_UNKNOWN
    return holdout.fix(pair, tf_key, conf, now, touched, path)


def load_dev(con: sqlite3.Connection, pair: str, tf_key: str, now: pd.Timestamp | None = None,
             path: Path | None = None) -> pd.DataFrame:
    """開発用データ：確定足のうち、最終テスト期間より前の足だけ。戦略・設定・モデルを選ぶ処理は必ずこれを使う。"""
    b = boundary(con, pair, tf_key, now, path)
    return holdout.dev_part(confirmed(load(con, pair, tf_key), TIMEFRAMES[tf_key], now), b)


# ---------------- データの品質 ----------------

def quality(df: pd.DataFrame, tf: Timeframe) -> dict:
    """タイムゾーン・週末・欠損まわりの確認用の集計（時刻は UTC）。"""
    idx = df.index
    s = idx.to_series()
    out = {
        "bars": len(df),
        "duplicates": int(idx.duplicated().sum()),
        "saturday": int((idx.dayofweek == 5).sum()),
        "flat": int(((df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])).sum()),
    }
    if tf.key == "1h":
        out["sunday_hours"] = sorted(set(idx[idx.dayofweek == 6].hour))
        prev = s.shift()
        weekend = (prev.dt.dayofweek == 4) & s.dt.dayofweek.isin([6, 0])
        out["gaps"] = int(((s.diff() > tf.bar) & ~weekend).sum())  # 週末以外の空き（祝日の休場・欠損）
    else:
        out["sunday"] = int((idx.dayofweek == 6).sum())
        out["not_midnight"] = int((idx.hour != 0).sum())
    return out
