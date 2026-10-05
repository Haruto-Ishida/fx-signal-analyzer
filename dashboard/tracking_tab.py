"""タブ「記録と成績」（フェーズ4。研究ノートを反映した改修版）。"""
from __future__ import annotations

import math
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import backtest as bt
from fxtool import data, ml_eval, strategies, tracking
from fxtool.config import (
    DAILY_MAX_LOSS_R, LIVE_WEIGHT_MIN_TRADES, LIVE_WEIGHT_WINDOW, ML_HORIZON, TRACKING, WEEKLY_MAX_LOSS_R,
)
from fxtool.jobs import RISK

from .common import JST, NEUTRAL_COLOR, STRATEGY_COLORS, Context, fmt_pct, fmt_pf, fmt_pips

TASK_NAME = "FXSignalTool_Update"
DONE = list(tracking.DONE)
SOURCE_NAMES = {s.key: s.name for s in strategies.STRATEGIES} | {
    f"ml_{k}": f"機械学習・旧方式（{v}）" for k, v in tracking.MODEL_NAMES.items()
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


def summary_row(name: str, g: pd.DataFrame) -> dict:
    done = g[g["status"].isin(DONE)].sort_values("exit_time")
    m = bt.metrics(done["pips"])
    return {
        "対象": name, "記録件数": len(g),
        "エントリー待ち": int((g["status"] == "待機").sum()), "進行中": int((g["status"] == "進行中").sum()),
        "決済済み": f"{m['n']}" + ("（参考値）" if 0 < m["n"] < bt.MIN_TRADES_RELIABLE else ""),
        "勝率": fmt_pct(m["win_rate"]), "損益合計(pips)": fmt_pips(m["total"]),
        "平均(pips/回)": fmt_pips(m["avg"]), "PF": fmt_pf(m["pf"]), "最大DD(pips)": fmt_pips(m["max_dd"]),
    }


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
        "このツールが**実際に出したシグナル**を記録し、その後どうなったかを自動で答え合わせしています。"
        "過去にさかのぼって作った記録ではないので、件数は使い続けるほど増えていきます。"
        "**この本番の記録が、最も信頼できる最終テスト**です（開発中に見ていない、本当に未来のデータだからです）。"
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

    # ---------------- 最大損失 ----------------
    st.markdown("### ② 最大損失（1日・1週間）")
    h = ctx.halt or tracking.Halt(False, "", 0.0, 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("今日の損益（日本時間）", f"{h.day_r:+.1f}R", help=f"上限 −{DAILY_MAX_LOSS_R:g}R")
    c2.metric("今週の損益（月曜から）", f"{h.week_r:+.1f}R", help=f"上限 −{WEEKLY_MAX_LOSS_R:g}R")
    c3.metric("新しいシグナル", "停止中" if h.active else "出しています")
    if h.active:
        st.error(h.reason)
    n_supp = int(sl["suppressed"].sum()) if "suppressed" in sl else 0
    st.caption(
        f"1回の損切り幅を 1R（口座残高の約1%）として、本番の記録（全戦略）で決済した損益を合計しています。"
        f"今日 −{DAILY_MAX_LOSS_R:g}R・今週 −{WEEKLY_MAX_LOSS_R:g}R に達したら、その期間は新しいシグナルを出さず、"
        f"「停止中」として記録だけします（これまでに停止中に記録したシグナル：{n_supp}件）。"
        "このほか、ペーパー口座の損失（%）・連敗・取引の回数によるキルスイッチと、経済指標の前後の停止もあります"
        "（「💰 ペーパー口座」タブ）。上の「停止中」は、それらをすべてまとめた状態です。"
    )

    # ---------------- 重み ----------------
    st.markdown(f"### ③ 戦略の重み（本番の記録の直近{LIVE_WEIGHT_WINDOW}回から計算）")
    live = wh[wh["basis"] == "live"] if "basis" in wh else wh.iloc[0:0]
    if live.empty:
        st.info("まだ計算されていません。")
    else:
        latest = live[live["bar_time"] == live["bar_time"].max()]
        st.dataframe(pd.DataFrame({
            "戦略": latest["source"].map(SOURCE_NAMES),
            "重み": latest["weight"].map(lambda v: f"{v:.2f}"),
            "決済済み（直近）": latest["recent_n"],
            "損益(pips)": latest["recent_pips"].map(fmt_pips),
            "PF": latest["recent_pf"].map(lambda v: fmt_pf(math.inf if pd.isna(v) else v)),
            "z値": latest["z"].map(lambda v: f"{v:+.2f}"),
        }), hide_index=True, width="stretch")
        st.caption(
            "重み = 1 + 0.25 × z値（0.25〜1.75 に制限し、平均が1になるようそろえたもの）。z値は「1回あたりの平均損益が0から"
            f"標準誤差いくつ分離れているか」で、決済済みが{LIVE_WEIGHT_MIN_TRADES}回未満の戦略は0（重み1）とします。"
            "直近の成績を追いかけすぎないよう、意図的に控えめにしています。"
        )
        if live["bar_time"].nunique() > 1:
            fig = go.Figure()
            for src, g in live.groupby("source"):
                fig.add_trace(go.Scatter(x=ctx.to_display_time(pd.DatetimeIndex(g["bar_time"])), y=g["weight"],
                                         name=SOURCE_NAMES.get(src, src), line=dict(color=SOURCE_COLORS.get(src))))
            fig.add_hline(y=1, line=dict(color=NEUTRAL_COLOR, dash="dash", width=1))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="重み",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, width="stretch")
    old = wh[wh["basis"] != "live"] if "basis" in wh else wh
    if not old.empty:
        st.caption(f"※ 以前の重み（{len(old)}件）は、直近のバックテスト＝最終テスト期間から計算していたため使っていません。")

    # ---------------- 記録したシグナル ----------------
    st.markdown("### ④ 記録したシグナルの成績")
    if sl.empty:
        st.info("まだシグナルは記録されていません。新しい足が確定してシグナルが出ると、ここに記録されます。")
    else:
        rows = [summary_row(SOURCE_NAMES.get(src, src), sl[sl["source"] == src]) for src in SOURCE_NAMES
                if (sl["source"] == src).any()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "エントリーはシグナルの次の足の始値（次の足が始まるまでは「待機」）、"
            f"損切り・利確は ATR×{RISK.sl_atr:g}・ATR×{RISK.tp_atr:g}（左の設定ではなく初期値で固定）、"
            f"{settings.log_max_bars}本で決着しなければ時間切れ（トリプルバリア）。損益は取引コストを差し引いた値です。"
            f"研究ノートにならい、決済済みが{bt.MIN_TRADES_RELIABLE}回未満の成績は「参考値」です。"
            "研究ノート反映前の記録（エントリーの基準が「シグナルの足の終値」）も含まれています。"
        )
        done = sl[sl["status"].isin(DONE)].sort_values("exit_time")
        if len(done) >= 2:
            fig = go.Figure()
            for src, g in done.groupby("source"):
                fig.add_trace(go.Scatter(
                    x=ctx.to_display_time(pd.DatetimeIndex(g["exit_time"])), y=g["pips"].cumsum(), mode="lines+markers",
                    name=SOURCE_NAMES.get(src, src), line=dict(color=SOURCE_COLORS.get(src), shape="hv"),
                ))
            fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOR, width=1))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="累積損益（pips）",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, width="stretch")
        mon = tracking.monthly(done)
        if not mon.empty:
            st.markdown("**月ごとの損益（pips・日本時間で決済した月）**")
            mon = mon.rename(columns=SOURCE_NAMES).sort_index(ascending=False)
            mon["合計"] = mon.sum(axis=1)
            st.dataframe(mon.map(lambda v: "" if pd.isna(v) else f"{v:+,.1f}"), width="stretch")
            st.caption("全期間の合計だけでなく、月ごとに崩れていないか（直近で悪化していないか）を確認してください。")
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
                "機械学習": [("—" if pd.isna(p) else f"{d}（{p:.0%}）") for p, d in zip(t["meta_p"], t["meta_decision"])],
                "停止中": t["suppressed"].map({1: "停止中", 0: ""}),
                "状況": t["status"].replace({"期限": "時間切れ", "待機": "エントリー待ち"}),
                "エントリーの基準": t["entry_basis"].map({"next_open": "次の足の始値", "signal_close": "シグナルの足の終値（旧）"}),
                "決済": ["—" if pd.isna(x) else ctx.label(pd.Timestamp(x)) for x in t["exit_time"]],
                "損益(pips)": t["pips"].round(1),
            }), hide_index=True, width="stretch")

    render_meta(ctx, sl, reg, settings)
    render_legacy(ctx, preds, reg)


def render_meta(ctx: Context, sl: pd.DataFrame, reg: pd.DataFrame, settings) -> None:
    st.markdown("### ⑤ 機械学習（メタラベリング：シグナルに乗るか見送るか）")
    meta = reg[reg["model"] == "meta"]
    if meta.empty:
        st.info("まだ学習されていません。")
        return
    st.caption(f"{settings.retrain_every}本ごと（と、試した組み合わせの数が増えたとき）に自動で再学習します。"
               "条件の判定とモデルの選択は開発用データだけで行い、本番モデルは選んだ設定のまま、決着したすべてのラベルで学習します。")
    r = meta.iloc[::-1]

    def period(a, b) -> str:
        return f"{ctx.label(pd.Timestamp(a))} 〜 {ctx.label(pd.Timestamp(b))}"

    st.dataframe(pd.DataFrame({
        "学習日時": r["trained_at"].map(jst),
        "採用モデル": r["kind"].map(lambda k: ml_eval.LADDER[k][0] if k in ml_eval.LADDER else k),
        "条件の判定に使った期間（開発用）": [period(a, b) for a, b in zip(r["eval_from"], r["eval_to"])],
        "本番モデルの学習期間（決着したラベル）": [period(a, b) for a, b in zip(r["train_from"], r["trained_through"])],
        "学習件数": r["n_train"],
        "検証の売買損益(pips)": r["trade_pips"].map(fmt_pips),
        "ランダムを上回った割合": r["beat"].map(fmt_pct),
        "試した数": r["n_trials"].map(lambda v: "—" if pd.isna(v) else f"{int(v)}"),
        "シグナルを出す条件": r["ok"].map({1: "✅ 満たす", 0: "❌ 満たさない（参考）"}),
    }), hide_index=True, width="stretch")

    judged = sl[sl["meta_decision"].isin(["乗る", "見送り"])]
    done = judged[judged["status"].isin(DONE)]
    rows = [summary_row("乗ると判定したシグナル", judged[judged["meta_decision"] == "乗る"]),
            summary_row("見送ると判定したシグナル", judged[judged["meta_decision"] == "見送り"]),
            summary_row("判定したシグナル全体", judged)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    taken = done[done["meta_decision"] == "乗る"]["pips"]
    skipped = done[done["meta_decision"] == "見送り"]["pips"]
    if len(taken) and len(skipped):
        diff = taken.mean() - skipped.mean()
        st.caption(f"本番の記録では、乗ったシグナルの1回あたり損益が、見送ったシグナルより {diff:+.1f} pips "
                   f"{'良い' if diff > 0 else '悪い'}結果です（決済済み {len(done)} 件。件数が少ないうちは偶然の影響が大きい）。")
    else:
        st.caption("「乗る」「見送り」の両方に決済済みのシグナルがたまると、見送りの効果を比べられます。")


def render_legacy(ctx: Context, preds: pd.DataFrame, reg: pd.DataFrame) -> None:
    old = reg[reg["model"] != "meta"]
    if old.empty and preds.empty:
        return
    with st.expander(f"旧方式（{ML_HORIZON}本後の方向予測）の記録：研究ノート反映前のもので、今は記録していません"):
        if not old.empty:
            st.dataframe(pd.DataFrame({
                "学習日時": old["trained_at"].map(jst),
                "モデル": old["model"].map(tracking.MODEL_NAMES),
                "検証の正解率": old["acc"].map(lambda v: f"{v:.1%}"),
                "シグナルを出す条件": old["ok"].map({1: "✅", 0: "❌"}),
            }), hide_index=True, width="stretch")
        if not preds.empty:
            t = preds.iloc[::-1]
            st.dataframe(pd.DataFrame({
                "足": ctx.labels(pd.DatetimeIndex(t["bar_time"])),
                "モデル": t["model"].map(tracking.MODEL_NAMES),
                "上がる確率": t["p"].map(lambda v: f"{v:.1%}"),
                "判断": t["decision"],
                "結果": t["y"].map({1.0: "上がった", 0.0: "上がらなかった"}).fillna("答え合わせ待ち"),
                "当たり": np.where(t["y"].isna(), "", np.where((t["p"] >= 0.5) == (t["y"] == 1), "○", "×")),
            }), hide_index=True, width="stretch")
