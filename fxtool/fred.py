"""FRED（米国セントルイス連銀）の公式 API からデータを取る（G10 の仮説の検証用）。

- API キーは .env の FRED_API_KEY（画面・ログ・エラーの文に出さない）。
- 取ったデータは data/external/fred/{系列のID}.csv に保存する（git に入れない・配らない。OECD のデータは著作物）。
- end を必ず指定する。開発用の検証では g10.DEV_CUTOFF（2020-10-14）より後は取らない。
- FRED の利用ルールに合わせて、問い合わせの間を0.6秒以上あける。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .config import ROOT

API = "https://api.stlouisfed.org/fred/series/observations"
DIR = ROOT / "data" / "external" / "fred"
_last = [0.0]


def _key() -> str:
    from .notify import load_env
    k = load_env().get("FRED_API_KEY", "")
    if not k:
        raise RuntimeError(".env に FRED_API_KEY がありません")
    return k


def observations(series_id: str, start: str, end: str) -> pd.Series:
    """start〜end の値（欠けている日は NaN）。"""
    key = _key()
    wait = 0.6 - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = API + "?" + urllib.parse.urlencode({"series_id": series_id, "api_key": key, "file_type": "json",
                                               "observation_start": start, "observation_end": end})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fx-signal-tool/1.0"}), timeout=60) as r:
            obs = json.loads(r.read().decode())["observations"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"FRED の {series_id} を取れませんでした（HTTP {e.code}）") from None
    finally:
        _last[0] = time.monotonic()
    s = pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else float("nan"))
                   for o in obs}, name=series_id, dtype=float)
    if len(s) and s.index.max() > pd.Timestamp(end):
        raise RuntimeError(f"{series_id}：指定した終わり（{end}）より後のデータが返ってきました")
    return s


def fetch_to_csv(series_id: str, start: str, end: str) -> pd.Series:
    s = observations(series_id, start, end)
    DIR.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(DIR / f"{series_id}.csv", index_label="date")
    return s


def load_csv(series_id: str, end: pd.Timestamp | None = None) -> pd.Series:
    s = pd.read_csv(DIR / f"{series_id}.csv", index_col="date", parse_dates=["date"])[series_id].astype(float)
    return s[s.index <= end] if end is not None else s
