"""タブ「論文ウォッチ」（要件定義書 11-2）。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fxtool import data, papers

from .common import Context


def render(ctx: Context) -> None:
    p = ctx.settings.papers
    st.markdown(
        "FX・為替の予測・金融の時系列データの機械学習についての**新しい論文**を、arXiv（論文の公開サイト）から集めて一覧にします。"
        "**集めるのは自動、使うかどうかはあなたが決めます。**気になる論文に「評価したい」の印を付けて依頼してもらえれば、"
        "研究ノートの原則（コストを考えているか、未来のデータが混ざっていないか、比べる基準があるか など）に照らして評価します。"
    )
    con = data.connect()
    try:
        papers.ensure(con)
        last = con.execute("SELECT fetched_at, n_found, n_new, error FROM paper_fetches ORDER BY id DESC LIMIT 1").fetchone()
        c1, c2 = st.columns([3, 1])
        keyword = c1.text_input("キーワードで絞り込み（題名・概要）", key="paper_kw")
        show = c2.multiselect("表示する印", list(papers.STATUSES), default=["未読", "評価したい"], key="paper_status")
        df = papers.load(con, keyword, tuple(show) or papers.STATUSES)
    finally:
        con.close()
    st.caption(
        f"検索の条件：{papers.build_query(p)}（最大 {p.max_results} 件。`settings.toml` の `[papers]` で変更できます）。"
        f"週末の自動収集は **{'有効' if p.enabled else '停止中'}**。"
        + (f"前回の収集：{last[0]} UTC" + (f"（エラー：{last[3]}）" if last[3] else f"（{last[1]}件中、新しい論文 {last[2]}件）")
           if last else "まだ集めていません。")
    )
    if st.button("📥 いま集める（arXiv に接続します）", key="paper_fetch",
                 help="送るのは検索の条件だけです。個人情報や口座の情報は送りません。"):
        con = data.connect()
        try:
            with st.spinner("arXiv から集めています…"):
                n, new = papers.fetch(con, p, data.now_utc())
            st.success(f"{n} 件見つかり、新しい論文は {new} 件でした。")
        except Exception as e:
            st.error(f"集められませんでした：{e}")
        finally:
            con.close()
    if df.empty:
        st.info("表示する論文がありません。")
        return
    for r in df.head(50).itertuples():
        with st.expander(f"[{r.status}] {r.title}（{r.published[:10]}）"):
            st.markdown(f"**著者**：{r.authors}  \n**分野**：{r.categories}  \n**リンク**：{r.link}")
            st.markdown(f"**概要（原文）**：{r.summary}")
            new = st.radio("印", list(papers.STATUSES), index=list(papers.STATUSES).index(r.status), horizontal=True,
                           key=f"paper_mark_{r.arxiv_id}")
            if new != r.status:
                con = data.connect()
                try:
                    papers.mark(con, r.arxiv_id, new)
                finally:
                    con.close()
                st.rerun()
