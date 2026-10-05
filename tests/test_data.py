import pandas as pd

from conftest import make_ohlc
from fxtool import data
from fxtool.config import TIMEFRAMES


def yf_like(df: pd.DataFrame, tz: str | None = "Europe/London") -> pd.DataFrame:
    """yfinance が返す形（MultiIndex 列・タイムゾーン付き）を再現する。"""
    out = df.rename(columns=str.title).copy()
    out["Adj Close"] = out["Close"]
    out["Volume"] = 0
    out.columns = pd.MultiIndex.from_product([out.columns, ["USDJPY=X"]])
    if tz:
        out.index = out.index.tz_localize("UTC").tz_convert(tz)
    return out


def test_clean_converts_to_utc_and_drops_bad_rows():
    df = make_ohlc([150, 151, 152, 153])
    df.iloc[1, df.columns.get_loc("high")] = 140  # 高値 < 安値
    df.iloc[2, df.columns.get_loc("close")] = 0
    out = data.clean(yf_like(df))
    assert list(out.columns) == ["open", "high", "low", "close"]
    assert out.index.tz is None
    assert list(out.index) == [df.index[0], df.index[3]]  # UTC に戻っている


def test_save_load_roundtrip_and_upsert(tmp_path):
    con = data.connect(tmp_path / "t.db")
    df = make_ohlc([150, 151, 152])
    assert data.save(con, "USD/JPY", "1h", df) == 3
    assert data.save(con, "USD/JPY", "1h", df) == 0  # 同じ足は増えない
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] = 999  # 形成中の足が確定値で上書きされるケース
    data.save(con, "USD/JPY", "1h", df2)
    loaded = data.load(con, "USD/JPY", "1h")
    pd.testing.assert_frame_equal(loaded, df2, check_freq=False, check_names=False)
    assert data.load(con, "USD/JPY", "1d").empty


def test_choose_period():
    tf = TIMEFRAMES["1h"]
    now = pd.Timestamp("2026-10-05 12:00")
    assert data.choose_period(tf, None, now) == "730d"
    assert data.choose_period(tf, now - pd.Timedelta(hours=3), now) == "5d"
    assert data.choose_period(tf, now - pd.Timedelta(days=20), now) == "1mo"
    assert data.choose_period(tf, now - pd.Timedelta(days=1000), now) == "730d"
    assert data.choose_period(TIMEFRAMES["1d"], now - pd.Timedelta(days=3000), now) == "max"


def test_confirmed_excludes_forming_bar():
    df = make_ohlc([1, 2, 3], start="2026-10-05 09:00")
    now = pd.Timestamp("2026-10-05 11:30")
    assert list(data.confirmed(df, TIMEFRAMES["1h"], now).index.hour) == [9, 10]


def test_update_incremental_and_throttled(tmp_path, monkeypatch):
    con = data.connect(tmp_path / "t.db")
    calls = []
    first = make_ohlc([150, 151, 152], start="2026-10-05 00:00")
    second = make_ohlc([152.5, 153, 154], start="2026-10-05 02:00")  # 最後の足が重複 + 2本追加

    def fake_download(ticker, tf, period):
        calls.append(period)
        return first if len(calls) == 1 else second

    monkeypatch.setattr(data, "download", fake_download)
    monkeypatch.setattr(data, "now_utc", lambda: pd.Timestamp("2026-10-05 05:00"))

    r1 = data.update(con, "USD/JPY", "USDJPY=X", "1h")
    assert (r1.period, r1.added) == ("730d", 3)
    assert data.update(con, "USD/JPY", "USDJPY=X", "1h").period is None  # 10分以内は取りに行かない
    r2 = data.update(con, "USD/JPY", "USDJPY=X", "1h", force=True)
    assert (r2.period, r2.added) == ("5d", 2)
    assert calls == ["730d", "5d"]
    assert data.load(con, "USD/JPY", "1h")["close"].tolist() == [150, 151, 152.5, 153, 154]


def test_update_keeps_working_offline(tmp_path, monkeypatch):
    con = data.connect(tmp_path / "t.db")

    def boom(*a):
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "download", boom)
    r = data.update(con, "USD/JPY", "USDJPY=X", "1h")
    assert r.error and "offline" in r.error and r.added == 0
