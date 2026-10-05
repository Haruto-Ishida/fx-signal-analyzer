"""Step 11：用語集と、断定的な表現を使っていないことの確認。"""
from pathlib import Path

from fxtool import glossary

ROOT = Path(__file__).resolve().parent.parent


def test_required_terms_from_the_spec():
    names = " ".join(glossary.GLOSSARY)
    for w in ("GA", "ケリー基準", "PBO", "DSR", "シャドーモード", "時間加重リターン"):
        assert w in names, w


def test_each_term_has_four_parts_and_example():
    for t in glossary.GLOSSARY.values():
        assert all(len(x) >= 10 for x in (t.what, t.read, t.use, t.caution)), t.name
        assert t.short and t.group in glossary.GROUPS
        assert any(ch.isdigit() for ch in t.read), f"{t.name}：数字の例がない"


def test_search():
    assert [t.name for t in glossary.search("ケリー")] == ["ケリー基準・分数ケリー"]
    assert all(t.group == "守り" for t in glossary.search(group="守り"))
    assert len(glossary.search()) == len(glossary.GLOSSARY)


def test_no_assertive_promises_anywhere():
    """画面に出る文章（画面のコード・解説・通知・用語集の本文）に、断定的な表現がない。"""
    texts = {t.name: " ".join([t.short, t.what, t.read, t.use, t.caution]) for t in glossary.GLOSSARY.values()}
    files = list((ROOT / "dashboard").glob("*.py")) + [ROOT / "app.py", ROOT / "fxtool" / "commentary.py",
                                                         ROOT / "fxtool" / "notify.py"]
    texts.update({f.name: f.read_text(encoding="utf-8") for f in files})
    for name, text in texts.items():
        for bad in glossary.FORBIDDEN:
            assert bad not in text, f"{name} に「{bad}」"
