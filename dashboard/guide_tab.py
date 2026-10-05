"""タブ「使い方・用語集」（要件定義書 13章）。"""
from __future__ import annotations

import streamlit as st

from fxtool import glossary

from .common import Context


def render(ctx: Context) -> None:
    st.markdown(
        "### このツールでできること\n"
        "1. **📡 現在のシグナル**：5つの戦略の「買い・売り・様子見」と、相場の解説・これからの予定。\n"
        "2. **🧪 バックテスト**：過去のデータで、戦略の成績をコスト込みで検証（ランダム売買などと比べる）。\n"
        "3. **🤖 機械学習予測**：機械学習でシグナルに「乗るか、見送るか」を判断。本番モデルの管理（巻き戻し・チャレンジャー）。\n"
        "4. **📒 記録と成績**：本番の記録（シグナルとその結果）と、成績悪化の警告。\n"
        "5. **💰 ペーパー口座**：仮想の10万円での口座全体の成績と、キルスイッチ（自動停止）。\n"
        "6. **📚 有名手法**・**🧬 GA探索**：有名な手法や、自動で探した手法を、同じ基準で正直に検証。\n"
        "7. **📰 論文ウォッチ**：新しい論文を集めて一覧に。\n\n"
        "**このツールは表示と記録だけを行い、注文は一切しません。**シグナルは参考情報で、将来の利益を保証するものではありません。"
    )
    st.markdown("### 用語集")
    st.caption("「何を表すか」「どう読むか（数字の例）」「このツールでの使い方」「注意点」の順に説明しています。")
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("用語をさがす", key="glossary_q", placeholder="例：DSR、ケリー、シャドー")
    group = c2.selectbox("分類", ["すべて", *glossary.GROUPS], key="glossary_group")
    terms = glossary.search(q, None if group == "すべて" else group)
    if not terms:
        st.info("見つかりませんでした。")
    for t in terms:
        with st.expander(f"{t.name}：{t.short}"):
            st.markdown(f"**何を表すか**：{t.what}\n\n**どう読むか**：{t.read}\n\n**このツールでの使い方**：{t.use}\n\n"
                        f"**注意点**：{t.caution}")
