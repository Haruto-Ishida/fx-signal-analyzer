"""FRED の取得（実際には接続しない）。"""
import io
import json

import pandas as pd
import pytest

from fxtool import fred


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_observations_parse_and_hide_key(monkeypatch):
    monkeypatch.setattr(fred, "_key", lambda: "0123456789abcdef0123456789abcdef")
    seen = {}

    def fake(req, timeout):
        seen["url"] = req.full_url
        body = {"observations": [{"date": "2020-01-02", "value": "1.12"}, {"date": "2020-01-03", "value": "."}]}
        return Resp(json.dumps(body).encode())

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake)
    s = fred.observations("DEXUSEU", "2020-01-01", "2020-10-14")
    assert s.iloc[0] == 1.12 and pd.isna(s.iloc[1])
    assert "observation_end=2020-10-14" in seen["url"]


def test_error_message_has_no_key(monkeypatch):
    monkeypatch.setattr(fred, "_key", lambda: "0123456789abcdef0123456789abcdef")

    def boom(req, timeout):
        raise fred.urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(fred.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as e:
        fred.observations("X", "2020-01-01", "2020-10-14")
    assert "0123456789abcdef" not in str(e.value)


def test_refuses_data_after_end(monkeypatch):
    monkeypatch.setattr(fred, "_key", lambda: "k")
    monkeypatch.setattr(fred.urllib.request, "urlopen",
                        lambda req, timeout: Resp(json.dumps({"observations": [{"date": "2021-01-04", "value": "1"}]}).encode()))
    with pytest.raises(RuntimeError):
        fred.observations("X", "2020-01-01", "2020-10-14")
