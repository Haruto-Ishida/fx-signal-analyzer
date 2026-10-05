"""Step 9：論文ウォッチ（実際の arXiv への接続はしない。サンプルの応答で確認）。"""
import urllib.parse

import pandas as pd
import pytest

from fxtool import papers, settings
from fxtool.settings import Papers

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2609.01234v2</id>
    <updated>2026-09-20T10:00:00Z</updated>
    <published>2026-09-15T09:00:00Z</published>
    <title>Forecasting   Exchange Rates
      with Transformers</title>
    <summary>  We study FX forecasting.
      Costs are considered. </summary>
    <author><name>A. Author</name></author>
    <author><name>B. Author</name></author>
    <link href="http://arxiv.org/abs/2609.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2609.01234v2" rel="related" type="application/pdf"/>
    <category term="q-fin.ST" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2609.05678v1</id>
    <updated>2026-09-18T10:00:00Z</updated>
    <published>2026-09-18T10:00:00Z</published>
    <title>Currency Trading with Reinforcement Learning</title>
    <summary>Abstract two.</summary>
    <author><name>C. Author</name></author>
    <link href="http://arxiv.org/abs/2609.05678v1" rel="alternate" type="text/html"/>
    <category term="q-fin.TR" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

NOW = pd.Timestamp("2026-10-10 03:00")  # 日本時間の土曜


@pytest.fixture
def con(tmp_path):
    from fxtool import data
    c = data.connect(tmp_path / "pp.db")
    yield c
    c.close()


def test_query_and_url():
    p = Papers(keywords=("forex", "exchange rate"), categories=("q-fin.TR",), max_results=20)
    assert papers.build_query(p) == '(all:forex OR all:"exchange rate") AND (cat:q-fin.TR)'
    url = papers.build_url(p)
    assert url.startswith("https://export.arxiv.org/api/query?")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["max_results"] == ["20"] and q["sortBy"] == ["submittedDate"]
    assert q["search_query"] == [papers.build_query(p)]


def test_parse_atom_cleans_and_strips_version():
    ps = papers.parse_atom(SAMPLE)
    assert [p.arxiv_id for p in ps] == ["2609.01234", "2609.05678"]
    a = ps[0]
    assert a.title == "Forecasting Exchange Rates with Transformers"
    assert a.summary == "We study FX forecasting. Costs are considered."
    assert a.authors == "A. Author, B. Author" and a.categories == "q-fin.ST, cs.LG"
    assert a.link == "http://arxiv.org/abs/2609.01234v2"


def test_fetch_saves_dedupes_and_keeps_marks(con):
    calls = []

    def fake(url):
        calls.append(url)
        return SAMPLE

    assert papers.fetch(con, Papers(), NOW, downloader=fake) == (2, 2)
    papers.mark(con, "2609.01234", "評価したい", note="あとで")
    newer = SAMPLE.replace("2609.01234v2", "2609.01234v3").replace("with Transformers", "with Transformers (rev)")
    assert papers.fetch(con, Papers(), NOW, downloader=lambda u: newer) == (2, 0)   # 重複は増えない
    df = papers.load(con)
    row = df[df["arxiv_id"] == "2609.01234"].iloc[0]
    assert row["status"] == "評価したい" and row["note"] == "あとで" and row["title"].endswith("(rev)")
    assert len(papers.load(con, keyword="reinforcement")) == 1
    assert len(papers.load(con, statuses=("評価したい",))) == 1
    assert papers.unread_count(con) == 1
    with pytest.raises(ValueError):
        papers.mark(con, "2609.01234", "よくわからない")


def test_fetch_error_is_recorded(con):
    def broken(url):
        raise OSError("network down")
    with pytest.raises(OSError):
        papers.fetch(con, Papers(), NOW, downloader=broken)
    assert con.execute("SELECT error FROM paper_fetches").fetchone()[0].startswith("OSError")


def test_weekly_due_requires_enabled_weekend_and_interval(con):
    on = Papers(enabled=True)
    assert not papers.weekly_due(con, Papers(), NOW)                  # 初期値は停止中
    assert papers.weekly_due(con, on, NOW)
    assert not papers.weekly_due(con, on, pd.Timestamp("2026-10-08 03:00"))  # 木曜
    papers.fetch(con, on, NOW, downloader=lambda u: SAMPLE)
    assert not papers.weekly_due(con, on, NOW + pd.Timedelta(days=1))
    assert papers.weekly_due(con, on, NOW + pd.Timedelta(days=7))


def test_settings_lists(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[papers]\nkeywords = ["fx", " carry trade "]\ncategories = "q-fin.TR"\nmax_results = 999\n', encoding="utf-8")
    s = settings.load(p)
    assert s.papers.keywords == ("fx", "carry trade")
    assert s.papers.categories == Papers().categories and s.papers.max_results == 200
    assert len(s.warnings) == 2
    assert settings.load().papers.enabled is False


def test_no_network_by_default(monkeypatch, con):
    """テストや画面の表示だけでは、外部に接続しない。"""
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("外部に接続しようとした")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    papers.load(con)
    papers.weekly_due(con, Papers(), NOW)
