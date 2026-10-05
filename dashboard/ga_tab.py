"""タブ「GA探索」（要件定義書 6章）。遺伝的アルゴリズムで見つけた戦略の候補（シグナルには使わない）。"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from fxtool import data, ga, ga_jobs, trials

from .common import Context, fmt_pips


def render(ctx: Context) -> None:
    s = ctx.settings
    tb = s.trial_budget
    st.markdown(
        "**遺伝的アルゴリズム（GA）**は、戦略の部品（売買のきっかけ1つ＋絞り込みの条件 最大2つ）の組み合わせを、"
        "生き物の進化のように「良いものどうしを組み合わせる」「一部を変える」をくり返して探す方法です。"
        f"部品の組み合わせは全部で **{len(set(ga.canonical([t, a, b]) for t in range(len(ga.TRIGGERS)) for a in range(len(ga.FILTERS)) for b in range(len(ga.FILTERS)))):,} 通り**あり、"
        f"1回の探索で調べるのは **{tb.ga_full_search} 通りまで**です。"
    )
    st.warning(
        "⚠️ GA で見つかった戦略は「たくさん試した中で一番良かったもの」なので、**偶然よく見えているだけの可能性が高い**です。"
        "ここに出る候補はシグナルには使いません。ウォークフォワード・PBO・DSR・ランダムとの比較・年別の確認に合格し、"
        "さらにあなたが承認したものだけを採用の候補にします。"
    )
    con = data.connect()
    try:
        runs = ga_jobs.load_runs(con, ctx.pair, ctx.tf_key)
        n_now = trials.count(con, ctx.pair, ctx.tf_key)
        cands = ga_jobs.load_candidates(con, int(runs["id"].iloc[0])) if len(runs) else pd.DataFrame()
    finally:
        con.close()

    per_year = ga_jobs.projected_trials_per_year(s)
    st.caption(
        f"🧮 試した組み合わせ（{ctx.pair}・{ctx.tf.label}の累計）：{n_now:,} 通り。"
        f"本格的な探索を1回すると最大 {tb.ga_full_search} 通り増えます。"
        f"週末の自動の再探索は **{'有効' if tb.ga_weekly_enabled else '停止中'}**"
        f"（有効にすると1回 {tb.ga_weekly_search} 通りまで、1年で最大 {52 * tb.ga_weekly_search:,} 通り）。"
        f"いまの設定での1年分の見込み：最大 {per_year:,} 通り（本格的な探索1回を含む）。"
    )
    if st.button(f"🧬 開発用データで GA 探索をする（{ctx.pair}・{ctx.tf.label}・最大 {tb.ga_full_search} 通り）",
                 key="ga_run"):
        con = data.connect()
        try:
            with st.spinner("探索しています…（数十秒〜数分）"):
                ga_jobs.run(con, ctx.pair, ctx.tf_key, "full")
        finally:
            con.close()
        st.rerun()

    if runs.empty:
        st.info("まだ探索していません。")
        return
    r = runs.iloc[0]
    st.markdown(f"### 前回の探索の上位候補（{r['started_at']} UTC・{'本格的な探索' if r['kind'] == 'full' else '周辺の再探索'}・"
                f"{int(r['n_evaluated'])} 通り・シード値 {int(r['seed'])}）")
    st.caption("適応度：開発用データを時間順に5つに分け、区間ごとのシャープレシオ（年率）の平均から、ばらつきと条件の数を差し引いたもの。"
               "どの時期でも安定して良いものほど高くなります。")
    st.dataframe(pd.DataFrame({
        "順位": cands["rank"], "ルール": cands["rule"], "適応度": cands["fitness"].round(2),
        "取引回数": cands["trades"], "損益(pips・コスト後)": cands["total_pips"].map(fmt_pips),
        "区間ごとのシャープレシオ": cands["segment_sharpe"].map(lambda x: " / ".join(f"{v:+.1f}" for v in json.loads(x))),
    }), hide_index=True, width="stretch")
