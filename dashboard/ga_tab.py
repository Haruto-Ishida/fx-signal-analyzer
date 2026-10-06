"""タブ「GA探索」（要件定義書 6章）。遺伝的アルゴリズムで見つけた戦略の候補（シグナルには使わない）。"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import data, ga, ga_jobs, trials

from .common import Context, fmt_pips


def render_convergence(trace: list[float], cap: int) -> None:
    """本格的な探索で、何通り目で最良の候補が見つかり、それ以降は改善しなかったかのグラフ。"""
    conv = ga_jobs.convergence(trace)
    if conv is None:
        return
    st.markdown("#### 何通り目で最良の候補が見つかったか（最初の本格的な探索）")
    x = list(range(1, conv.n + 1))
    y = [max(v, -3.0) for v in trace]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="それぞれの組み合わせの適応度",
                             marker=dict(size=4, color="#9e9e9e")))
    fig.add_trace(go.Scatter(x=x, y=conv.best_so_far, mode="lines", name="それまでの最良", line=dict(color="#2962ff", width=3)))
    fig.add_vline(x=conv.best_at, line_dash="dash", line_color="#e53935",
                  annotation_text=f"最良が見つかった：{conv.best_at}通り目", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="評価した順番（何通り目）",
                      yaxis_title="適応度（−3 未満は −3 で表示）", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        f"- 全部で **{conv.n} 通り**評価し、最良の候補が見つかったのは **{conv.best_at} 通り目**。"
        f"それ以降の {conv.n - conv.best_at} 通りでは改善しませんでした。\n"
        f"- 最良との差が5%以内の候補は **{conv.near_at} 通り目**までに見つかっていました。\n"
        f"- 再探索の1回の数の目安（自動計算）：**{ga_jobs.suggest_budget(conv, cap)} 通り**"
        f"（最良が見つかるまでの数の半分。周辺だけを探すので、ゼロからより少なくて済む。上限 {cap}）。"
    )


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

    st.caption(
        f"🧮 試した組み合わせ（{ctx.pair}・{ctx.tf.label}の累計）：{n_now:,} 通り。"
        f"本格的な探索を1回すると最大 {tb.ga_full_search} 通り増えます。"
        f"定期的な再探索は **{'有効' if tb.ga_weekly_enabled else '停止中'}**"
        f"（{tb.ga_interval_weeks}週間に1回・日本時間の土曜・1回 {tb.ga_weekly_search} 通りまで。"
        "これまでの上位候補の周辺のうち、まだ試していない組み合わせだけを調べます）。"
    )
    with st.expander("📈 累計の試行数の見込みと、合格の基準の厳しさ"):
        has_full = len(runs) and (runs["kind"] == "full").any()
        rows = []
        for y in (1, 4):
            p = ga_jobs.projection(n_now, s, y, full_runs=0 if has_full else 1)
            rows.append({"期間": f"{y}年後", "再探索の回数": p["re_search_runs"],
                         "単純な見込み（通り）": f"{p['naive']:,}", "基準の厳しさ（単純）": f"{p['strict_naive']:.2f}倍",
                         "実際の上限（通り）": f"{p['capped']:,}", "基準の厳しさ（上限）": f"{p['strict_capped']:.2f}倍"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"「単純な見込み」は、毎回上限まで新しい組み合わせを試した場合。ただし GA の組み合わせは全部で "
            f"{ga_jobs.search_space_size():,} 通りしかなく、同じ組み合わせは二度数えないので、「実際の上限」を超えることはありません。"
            f"「基準の厳しさ」は、偶然でも出てしまうシャープレシオの水準（DSR の基準）が、いま（{n_now:,} 通り）の何倍になるかです。"
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
    full = runs[runs["kind"] == "full"]
    if len(full) and isinstance(full["trace"].iloc[-1], str):
        render_convergence(json.loads(full["trace"].iloc[-1]), tb.ga_weekly_search)
    st.dataframe(pd.DataFrame({
        "順位": cands["rank"], "ルール": cands["rule"], "適応度": cands["fitness"].round(2),
        "取引回数": cands["trades"], "損益(pips・コスト後)": cands["total_pips"].map(fmt_pips),
        "区間ごとのシャープレシオ": cands["segment_sharpe"].map(lambda x: " / ".join(f"{v:+.1f}" for v in json.loads(x))),
        "検証の判定": cands["verdict"].fillna("未検証"),
    }), hide_index=True, width="stretch")

    st.markdown(f"### 上位{ga_jobs.VALIDATE_TOP}個の検証")
    st.caption(
        "ほかの戦略と同じ合格基準（要件定義書 6-3）で調べます：ウォークフォワードの検証区間でコスト後プラス・取引100回以上・"
        "ランダム売買とランダムウォークに勝つ・黒字の区間が6割以上・極端に悪い年がない・DSR 95%以上・PBO 50%未満。"
        "探索のときと同じ設定なので、試行数は増えません。"
    )
    if st.button(f"🔍 上位{ga_jobs.VALIDATE_TOP}個を検証する", key="ga_validate"):
        con = data.connect()
        try:
            with st.spinner("検証しています…（1〜3分）"):
                ga_jobs.validate(con, ctx.pair, ctx.tf_key, int(r["id"]))
        finally:
            con.close()
        st.rerun()
    done = cands[cands["verdict"].notna()]
    if len(done):
        st.markdown(f"**{int(r['n_evaluated'])} 通り試して、その中で上位だったもの**の検証結果"
                    f"（DSR は累計 {int(done['n_trials'].max()):,} 通りで割り引き）：")
        for c in done.itertuples():
            yearly = json.loads(c.yearly) if c.yearly else {}
            with st.expander(f"{c.rank}位：{c.verdict} — {c.rule}"):
                st.markdown(c.reason)
                st.markdown(
                    f"- 検証区間の損益：{fmt_pips(c.wf_total)}（{c.wf_n} 回）\n"
                    f"- ランダム売買を上回った割合：{c.wf_beat:.0%}"
                    + ("" if pd.isna(c.rw_beat) else f"、ランダムウォークを上回った割合：{c.rw_beat:.0%}")
                    + f"\n- DSR：{'—' if pd.isna(c.dsr) else f'{c.dsr:.0%}'}、PBO：{'—' if pd.isna(c.pbo) else f'{c.pbo:.0%}'}"
                    + ("\n- 年ごとの損益（検証区間）：" + "、".join(f"{y}年 {v:+,.0f}" for y, v in yearly.items()) if yearly else "")
                )
