"""タブ「記録と成績」（フェーズ4）。"""
from __future__ import annotations

import math
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import backtest as bt
from fxtool import data, strategies, tracking
from fxtool.config import ML_HORIZON, ML_THRESHOLD, TRACKING
from fxtool.jobs import RISK

from .common import JST, NEUTRAL_COLOR, STRATEGY_COLORS, Context, fmt_pct, fmt_pf, fmt_pips

TASK_NAME = "FXSignalTool_Update"
SOURCE_NAMES = {s.key: s.name for s in strategies.STRATEGIES} | {
    f"ml_{k}": f"機械学習（{v}）" for k, v in tracking.MODEL_NAMES.items()
}
SOURCE_COLORS = dict(zip(SOURCE_NAMES, STRATEGY_COLORS + ["#5d4037", "#455a64"]))


@st.cache_data(ttl=300, show_spinner=False)
def task_registered() -> bool | None:
    try:
        r = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return None


def jst(ts) -> str:
    return "—" if pd.isna(ts) else (pd.Timestamp(ts) + JST).strftime("%Y/%m/%d %H:%M")


def render(ctx: Context) -> None:
    con = data.connect()
    try:
        sl = tracking.load_signal_log(con, ctx.pair, ctx.tf_key)
        preds = tracking.load_predictions(con, ctx.pair, ctx.tf_key)
        reg = tracking.load_registry(con, ctx.pair, ctx.tf_key)
        alerts = tracking.load_alerts(con, ctx.pair, ctx.tf_key)
        wh = tracking.load_weight_history(con, ctx.pair, ctx.tf_key)
        runs = tracking.load_job_runs(con, 200)
    finally:
        con.close()
    settings = TRACKING[ctx.tf_key]

    st.markdown(
        "このツールが**実際に出したシグナルと予測**を記録し、その後どうなったかを自動で答え合わせしています。"
        "過去にさかのぼって作った記録ではないので、件数は使い続けるほど増えていきます。"
    )

    # ---------------- 自動更新の状態 ----------------
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        reg_state = task_registered()
        c1.metric("定期実行（タスクスケジューラ）",
                  {True: "登録済み", False: "未登録", None: "確認できません"}[reg_state])
        sched = runs[runs["origin"] == "schedule"]
        c2.metric("最後の定期実行", jst(sched["started_at"].iloc[0]) if len(sched) else "—")
        mine = runs[(runs["pair"] == ctx.pair) & (runs["tf"] == ctx.tf_key)]
        c3.metric("最後の処理（この通貨ペア・時間足）", jst(mine["started_at"].iloc[0]) if len(mine) else "—")
        if reg_state is False:
            st.caption("画面を開いている間は、表示のたびに記録・答え合わせをしています（1分に1回まで）。"
                       "画面を閉じていても記録を続けるには、フォルダの「自動更新を登録.bat」を実行してください（1時間ごと）。")
        errs = mine[mine["error"].notna()].head(3)
        for _, r in errs.iterrows():
            st.caption(f"⚠️ {jst(r['started_at'])} の処理でエラー：{r['error']}")
        with st.expander("処理の履歴"):
            st.dataframe(pd.DataFrame({
                "開始": runs["started_at"].map(jst), "通貨ペア": runs["pair"], "時間足": runs["tf"],
                "きっかけ": runs["origin"].map({"schedule": "定期実行", "dashboard": "画面", "manual": "手動"}),
                "内容": runs["summary"], "エラー": runs["error"].fillna(""),
            }), hide_index=True, width="stretch")

    # ---------------- 警告 ----------------
    st.markdown("### ① 警告")
    active = alerts[alerts["resolved_at"].isna()]
    if active.empty:
        st.success("現在、成績悪化などの警告はありません。")
    for _, a in active.iterrows():
        (st.error if a["level"] == "critical" else st.warning)(f"{a['message']}（{jst(a['created_at'])} から）")
    past = alerts[alerts["resolved_at"].notna()]
    if not past.empty:
        with st.expander(f"解決済みの警告（{len(past)}件）"):
            st.dataframe(pd.DataFrame({
                "発生": past["created_at"].map(jst), "解決": past["resolved_at"].map(jst),
                "レベル": past["level"].map({"critical": "重大", "warning": "注意"}), "内容": past["message"],
            }), hide_index=True, width="stretch")

    # ---------------- 重み ----------------
    period = "直近約1か月" if ctx.tf_key == "1h" else "直近約1年"
    st.markdown(f"### ② 戦略の重み（{period}＝{settings.recent_bars}本の成績から計算）")
    if wh.empty:
        st.info("まだ計算されていません。")
    else:
        latest = wh[wh["bar_time"] == wh["bar_time"].max()]
        st.dataframe(pd.DataFrame({
            "戦略": latest["source"].map(SOURCE_NAMES),
            "重み": latest["weight"].map(lambda v: f"{v:.2f}"),
            "直近の取引回数": latest["recent_n"],
            "直近の損益(pips)": latest["recent_pips"].map(fmt_pips),
            "直近のPF": latest["recent_pf"].map(lambda v: fmt_pf(math.inf if pd.isna(v) else v)),
            "z値": latest["z"].map(lambda v: f"{v:+.2f}"),
        }), hide_index=True, width="stretch")
        st.caption(
            "重み = 1 + 0.25 × z値（0.25〜1.75 に制限し、平均が1になるようそろえたもの）。z値は「1取引あたりの平均損益が0から"
            "標準誤差いくつ分離れているか」で、取引が5回未満なら0（重み1）とします。"
            "直近の成績を追いかけすぎないよう、意図的に控えめにしています。この重みは「現在のシグナル」タブの重み付き総合スコアに使います。"
        )
        if wh["bar_time"].nunique() > 1:
            fig = go.Figure()
            for src, g in wh.groupby("source"):
                fig.add_trace(go.Scatter(x=ctx.to_display_time(pd.DatetimeIndex(g["bar_time"])), y=g["weight"],
                                         name=SOURCE_NAMES.get(src, src), line=dict(color=SOURCE_COLORS.get(src))))
            fig.add_hline(y=1, line=dict(color=NEUTRAL_COLOR, dash="dash", width=1))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="重み",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, width="stretch")

    # ---------------- 記録したシグナル ----------------
    st.markdown("### ③ 記録したシグナルの成績")
    if sl.empty:
        st.info("まだシグナルは記録されていません。新しい足が確定してシグナルが出ると、ここに記録されます。")
    else:
        rows = []
        for src in SOURCE_NAMES:
            g = sl[sl["source"] == src]
            if g.empty:
                continue
            done = g[g["status"] != "進行中"].sort_values("exit_time")
            m = bt.metrics(done["pips"])
            rows.append({
                "戦略": SOURCE_NAMES[src], "記録件数": len(g), "進行中": int((g["status"] == "進行中").sum()),
                "決済済み": m["n"], "勝率": fmt_pct(m["win_rate"]), "損益合計(pips)": fmt_pips(m["total"]),
                "平均(pips/回)": fmt_pips(m["avg"]), "PF": fmt_pf(m["pf"]), "最大DD(pips)": fmt_pips(m["max_dd"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"エントリーはシグナルが出た足の終値、損切り・利確は ATR×{RISK.sl_atr:g}・ATR×{RISK.tp_atr:g}"
            "（左の設定ではなく初期値で固定）で記録し、"
            f"{settings.log_max_bars}本経っても決着しなければその足の終値で「期限」決済とします（機械学習は{ML_HORIZON}本）。"
            "損益は取引コストを差し引いた値です。決済済みが20回程度になるまでは、成績はほぼ偶然で決まると考えてください。"
        )
        done = sl[sl["status"] != "進行中"].sort_values("exit_time")
        if len(done) >= 2:
            fig = go.Figure()
            for src, g in done.groupby("source"):
                fig.add_trace(go.Scatter(
                    x=ctx.to_display_time(pd.DatetimeIndex(g["exit_time"])), y=g["pips"].cumsum(), mode="lines+markers",
                    name=SOURCE_NAMES.get(src, src), line=dict(color=SOURCE_COLORS.get(src), shape="hv"),
                ))
            fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOR, width=1))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="累積損益（pips）",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, width="stretch")
        with st.expander(f"記録の一覧（{len(sl)}件）"):
            t = sl.iloc[::-1]
            st.dataframe(pd.DataFrame({
                "シグナル": ctx.labels(pd.DatetimeIndex(t["signal_time"])),
                "戦略": t["source"].map(SOURCE_NAMES),
                "方向": t["direction"].map({1: "買い", -1: "売り"}),
                "エントリー": t["entry"].map(ctx.fmt_price),
                "損切り": t["sl"].map(ctx.fmt_price),
                "利確": t["tp"].map(ctx.fmt_price),
                "重み": t["weight"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
                "確率": t["prob"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
                "状況": t["status"],
                "決済": [ "—" if pd.isna(x) else ctx.label(pd.Timestamp(x)) for x in t["exit_time"]],
                "損益(pips)": t["pips"].round(1),
            }), hide_index=True, width="stretch")

    # ---------------- 機械学習 ----------------
    st.markdown(f"### ④ 機械学習モデル（{ML_HORIZON}本後の方向、確信度 {ML_THRESHOLD:.0%}）")
    if reg.empty:
        st.info("まだ学習されていません。")
        return
    st.caption(f"{settings.retrain_every}本ごとに、新しいデータを含めて自動で再学習します。")
    r = reg.iloc[::-1]
    st.dataframe(pd.DataFrame({
        "学習日時": r["trained_at"].map(jst),
        "モデル": r["model"].map(tracking.MODEL_NAMES),
        "学習データの最後": ctx.labels(pd.DatetimeIndex(r["trained_through"])),
        "検証の正解率": r["acc"].map(lambda v: f"{v:.1%}"),
        "基準（多い方）": r["majority_acc"].map(lambda v: f"{v:.1%}"),
        "検証の売買損益(pips)": r["trade_pips"].map(fmt_pips),
        "ランダムを上回った割合": r["beat"].map(fmt_pct),
        "シグナルを出す条件": r["ok"].map({1: "✅ 満たす", 0: "❌ 満たさない"}),
    }), hide_index=True, width="stretch")

    rows = []
    fig = go.Figure()
    for kind, name in tracking.MODEL_NAMES.items():
        g = preds[(preds["model"] == kind) & (preds["horizon"] == ML_HORIZON)]
        done = g.dropna(subset=["y"]).sort_values("bar_time")
        n = len(done)
        hit = ((done["p"] >= 0.5) == (done["y"] == 1)).astype(float)
        acc = float(hit.mean()) if n else math.nan
        rng = "—"
        if n >= 10:
            se = math.sqrt(acc * (1 - acc) / (n / ML_HORIZON))
            rng = f"{acc - 1.96 * se:.1%} 〜 {acc + 1.96 * se:.1%}"
        rows.append({"モデル": name, "予測した件数": len(g), "答え合わせ済み": n,
                     "実際の正解率": "—" if not n else f"{acc:.1%}", "95%の範囲": rng,
                     "シグナルを出した件数": int(g["decision"].isin(["買い", "売り"]).sum())})
        if n >= 20:
            w = min(50, n)
            fig.add_trace(go.Scatter(x=ctx.to_display_time(pd.DatetimeIndex(done["bar_time"])),
                                     y=hit.rolling(w).mean(), name=f"{name}（直近{w}件の正解率）",
                                     line=dict(color=SOURCE_COLORS[f"ml_{kind}"])))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("「実際の正解率」は、学習に使っていない新しい足について記録した予測の答え合わせです。"
               "数十件では誤差が大きいので、95%の範囲もあわせて見てください。")
    if fig.data:
        fig.add_hline(y=0.5, line=dict(color=NEUTRAL_COLOR, dash="dash", width=1))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")
    with st.expander(f"予測の記録（{len(preds)}件）"):
        t = preds.iloc[::-1]
        st.dataframe(pd.DataFrame({
            "足": ctx.labels(pd.DatetimeIndex(t["bar_time"])),
            "モデル": t["model"].map(tracking.MODEL_NAMES),
            "上がる確率": t["p"].map(lambda v: f"{v:.1%}"),
            "判断": t["decision"],
            "結果": t["y"].map({1.0: "上がった", 0.0: "上がらなかった"}).fillna("答え合わせ待ち"),
            "当たり": np.where(t["y"].isna(), "", np.where((t["p"] >= 0.5) == (t["y"] == 1), "○", "×")),
        }), hide_index=True, width="stretch")
