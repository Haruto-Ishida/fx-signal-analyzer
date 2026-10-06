"""タブ「組み合わせ探索」（要件定義書 6章）。戦略の部品の組み合わせを全通り調べた候補（シグナルには使わない）。
2026-10-07 に GA の探索から全通りの評価に切り替えた。GA の探索の結果があれば、そのグラフも表示する。"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import data, ga_jobs, trials

from .common import Context, fmt_pips


def render_distribution(trace: list[float]) -> None:
    """全通りの適応度の分布（ほとんどがどのあたりにあるか・プラスはいくつあるか）。"""
    x = [v for v in trace if v > -10]   # −10 は取引が少なすぎて評価しなかった組み合わせ
    n_skip = len(trace) - len(x)
    fig = go.Figure(go.Histogram(x=[max(v, -3.0) for v in x], nbinsx=60, marker_color="#9e9e9e"))
    fig.add_vline(x=0, line_dash="dash", line_color="#e53935", annotation_text="0（プラスとマイナスの境目）")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="適応度（−3 未満は −3 にまとめて表示）",
                      yaxis_title="組み合わせの数")
    st.plotly_chart(fig, width="stretch")
    pos = sum(v > 0 for v in x)
    st.caption(f"評価した {len(x):,} 通りのうち、適応度がプラスは {pos:,} 通り（{pos / max(len(x), 1):.0%}）。"
               f"取引が少なすぎて評価しなかった組み合わせ：{n_skip:,} 通り。")


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
    space = ga_jobs.search_space_size()
    st.markdown(
        "戦略の部品（売買のきっかけ1つ＋絞り込みの条件 最大2つ）の組み合わせを、**全部で "
        f"{space:,} 通り、1つ残らず**開発用データで調べます。"
        "以前は遺伝的アルゴリズム（GA：良いものどうしを組み合わせて探す方法）で一部だけを探していましたが、"
        "組み合わせが少なく全部調べられるので、見逃しがなく、運（乱数）にも左右されない「全通り」に切り替えました。"
        "GA の仕組みは、将来部品を増やして全部は調べられなくなったときのために残しています。"
    )
    st.warning(
        f"⚠️ ここで一番良かった戦略は「{space:,} 通り試した中で一番良かったもの」なので、**偶然よく見えているだけの可能性が高い**です。"
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
    done = bool(len(runs) and (runs["kind"] == "exhaustive").any())

    st.caption(f"🧮 試した組み合わせ（{ctx.pair}・{ctx.tf.label}の累計）：{n_now:,} 通り。"
               + ("全通りの評価は済んでいます（開発用データが変わるまで、やり直しても同じ結果です）。" if done
                  else f"全通りを調べると {space:,} 通り増えます。"))
    with st.expander("📈 累計の試行数の見込みと、合格の基準の厳しさ（全通りの評価＋年1回の見直し）"):
        rows = [{"時期": "全通りを調べた直後" if y == 0 else f"{y}年後（見直し{y}回）", "累計（通り）": f"{p['trials']:,}",
                 "基準の厳しさ": f"{p['strict']:.2f}倍"}
                for y in (0, 1, 4) for p in [ga_jobs.review_projection(n_now, y, done)]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "年1回の見直しで開発用データを新しくすると、同じ組み合わせでも「新しいデータでの試行」として数えるので、"
            f"見直しのたびに約 {space + 40:,} 通り増えます。「基準の厳しさ」は、偶然でも出てしまうシャープレシオの水準"
            f"（DSR の基準）が、いま（{n_now:,} 通り）の何倍になるかです。"
        )
    if not done and st.button(f"🔎 開発用データで全通りを調べる（{ctx.pair}・{ctx.tf.label}・{space:,} 通り）", key="exh_run"):
        con = data.connect()
        try:
            with st.spinner("調べています…（数十秒〜数分）"):
                ga_jobs.run(con, ctx.pair, ctx.tf_key, "exhaustive")
        finally:
            con.close()
        st.rerun()

    if runs.empty:
        st.info("まだ調べていません。")
        return
    r = runs.iloc[0]
    kind_label = {"exhaustive": "全通りの評価", "full": "GA の探索", "neighborhood": "周辺の再探索"}.get(r["kind"], r["kind"])
    st.markdown(f"### 上位20個（{r['started_at']} UTC・{kind_label}・{int(r['n_evaluated']):,} 通り）")
    st.caption("適応度：開発用データを時間順に5つに分け、区間ごとのシャープレシオ（年率）の平均から、ばらつきと条件の数を差し引いたもの。"
               "どの時期でも安定して良いものほど高くなります（0 より大きければ、平均してプラス）。")
    if r["kind"] == "exhaustive" and isinstance(r["trace"], str):
        render_distribution(json.loads(r["trace"]))
        if pd.notna(r.get("pbo")):
            st.caption(f"「{int(r['n_evaluated']):,} 通りの中から一番を選ぶ」こと自体の過学習の確率（PBO）：**{r['pbo']:.0%}**"
                       "（50% 以上なら、過去で一番だった組み合わせが、この先は平均以下になりやすい）。"
                       "下の検証の DSR・PBO は、この選び方の割り引きを使っています。")
    full = runs[runs["kind"] == "full"]
    if r["kind"] != "exhaustive" and len(full) and isinstance(full["trace"].iloc[-1], str):
        render_convergence(json.loads(full["trace"].iloc[-1]), s.trial_budget.ga_weekly_search)
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
