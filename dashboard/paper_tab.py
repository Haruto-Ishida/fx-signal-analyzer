"""タブ「ペーパー口座」（要件定義書 8章・ユーザーの決定 4・5）：仮想のお金での口座全体のシミュレーションと、キルスイッチ。"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import data, killswitch, paper

from .common import JST, Context


def _yen(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+,.0f}円"


def render(ctx: Context) -> None:
    s = ctx.settings
    acc, kill = ctx.paper, ctx.kill
    st.markdown(
        "**ペーパートレード**：このツールが出したシグナルを、**仮想のお金**ですべて取引したとみなした口座です（実際の注文はしません）。"
        f"初期残高は **{s.paper.initial_balance:,.0f}円**（`settings.toml` の `[paper]` で変更できます）。"
        "すべての通貨ペア・時間足をまとめた1つの口座として計算します。"
    )
    st.caption(
        f"取引するのは、停止中でなく、機械学習が「見送り」と判定しなかった戦略のシグナルです。"
        f"1回の損失の目安は残高の {s.sizing.fixed_risk_pct:g}%（固定）、実効レバレッジは {s.sizing.max_leverage:g} 倍までです。"
        "シグナルごとの記録（「📒 記録と成績」タブ）も、戦略の評価用にそのまま残しています。"
    )

    # ---------------- キルスイッチ ----------------
    st.markdown("### ① キルスイッチ（自動停止）")
    ks = s.kill_switch
    if kill is None:
        st.info("キルスイッチの状態を計算できませんでした。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("今日の損失", f"{max(kill.day_loss_pct, 0):.1f}%", help=f"上限 {ks.daily_loss_pct:g}%（その日の始まりの残高に対して）")
        c2.metric("今週の損失", f"{max(kill.week_loss_pct, 0):.1f}%", help=f"上限 {ks.weekly_loss_pct:g}%（月曜の始まりの残高に対して）")
        c3.metric("いまの連敗", f"{kill.streak}回", help=f"上限 {ks.max_consecutive_losses}回")
        cap = s.signal_cap(ctx.pair)
        c4.metric(f"今日のシグナル（{ctx.pair}）", f"{kill.signals_today.get(ctx.pair, 0)}回",
                  help=f"上限 {cap}回（過去のデータで「100日に1日くらいしか起きない回数」。通貨ペアごと）")
        if kill.active:
            for t in kill.trips:
                st.error(f"⛔ {t.reason}（自動で解除：日本時間 {(t.until + JST):%m/%d %H:%M}）")
                if st.button("この停止を手動で解除する", key=f"release_{t.key}",
                             help="解除しても、条件にもう一度達すれば（別の期間として）また止まります。解除した記録は残ります。"):
                    con = data.connect()
                    try:
                        killswitch.release(con, t.key, data.now_utc())
                    finally:
                        con.close()
                    st.rerun()
        else:
            st.success("✅ キルスイッチは作動していません。")
        for t in kill.released:
            st.caption(f"手動で解除済み：{t.reason}")
        st.caption(
            "どれか1つに達したら、その期間は新しいシグナルを出さず「停止中」として記録だけします："
            f"1日の損失 {ks.daily_loss_pct:g}%（その日の終わりまで）・1週間の損失 {ks.weekly_loss_pct:g}%（その週の終わりまで）・"
            f"{ks.max_consecutive_losses}連敗（最後の負けから24時間）・1日のシグナルの回数の上限（通貨ペアごと。その日の終わりまで、"
            "その通貨ペアだけ）。日付は日本時間です。数値は `settings.toml` の `[kill_switch]`・`[signal_caps]` で変更できます。"
            "このほか、重要な経済指標の前後"
            f"（発表の{s.event_filter.minutes_before}分前〜{s.event_filter.minutes_after}分後。時刻が決まっていない日銀会合は前後"
            f"{s.event_filter.unfixed_window_minutes}分）も新しいシグナルを止めます。"
        )
        with st.expander("通貨ペアごとの「1日のシグナルの回数」の上限"):
            from fxtool.config import PAIRS
            st.dataframe(pd.DataFrame({"通貨ペア": list(PAIRS), "上限（回）": [s.signal_cap(p) for p in PAIRS],
                                       "今日の回数": [kill.signals_today.get(p, 0) for p in PAIRS]}),
                         hide_index=True, width="stretch")
            st.caption("過去の開発用データ（1時間足と日足の5つの戦略の合計）で、「1日の回数がこの値以上になる日が100日に1日以下」"
                       "になる最小の回数です（2026-10-07 に計算）。")

    # ---------------- 口座 ----------------
    st.markdown("### ② 口座の成績")
    if acc is None or acc.trades.empty:
        st.info("まだ決済した取引がありません。本番の記録がたまると、ここに残高の推移が出ます。")
        return
    m = acc.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("残高", f"{m['final']:,.0f}円", _yen(m["final"] - m["initial"] - m["deposits"]))
    c2.metric("運用成績（時間加重リターン）", f"{m['twr']:+.1%}", help="入金の影響を除いた、運用だけの増え方です。")
    c3.metric("最大ドローダウン", f"{m['max_dd']:.1%}", help="残高（入金の影響を除く）が、それまでの最高からどれだけ下がったか。")
    c4.metric("取引した回数", f"{m['n']}回", f"見送り {m['n_skipped']}回", delta_color="off")
    d = acc.daily
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=s.paper.initial_balance + d["deposits_cum"], name="元本（初期残高＋入金）",
                             line=dict(color="#9e9e9e", dash="dot")))
    fig.add_trace(go.Scatter(x=d.index, y=d["balance"], name="残高", line=dict(color="#2962ff")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="円", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    t = acc.trades.sort_values("exit_time", ascending=False).head(30)
    st.markdown("**最近の取引（最大30件）**")
    st.dataframe(pd.DataFrame({
        "決済（日本時間）": (pd.DatetimeIndex(t["exit_time"]) + JST).strftime("%m/%d %H:%M"),
        "通貨ペア": t.get("pair", ""), "時間足": t.get("tf", ""), "戦略": t.get("source", ""),
        "方向": t["dir"].map({1: "買い", -1: "売り"}), "数量": t["units"].map(lambda u: f"{u:,.0f}" if u else "見送り"),
        "本来の数量": t["requested_units"].map(lambda u: f"{u:,.0f}"), "扱い": t["sizing_status"],
        "損益(pips)": t["pips"].map(lambda x: f"{x:+.1f}"), "損益(円)": t["pnl_jpy"].map(_yen), "メモ": t["note"],
    }), hide_index=True, width="stretch")

    # ---------------- 見送り・数量を減らした取引 ----------------
    st.markdown("### ③ 見送り・数量を減らした取引の記録")
    con = data.connect()
    try:
        log = paper.load_log(con)
    finally:
        con.close()
    adj = log[log["sizing_status"] != "通常"] if len(log) else log
    c1, c2, c3 = st.columns(3)
    c1.metric("記録した取引", f"{len(log)}件")
    c2.metric("数量を減らした", f"{int((log['sizing_status'] == '数量を減らした').sum()) if len(log) else 0}件")
    c3.metric("見送り", f"{int((log['sizing_status'] == '見送り').sum()) if len(log) else 0}件")
    st.caption(
        "1回の損失の目安（残高の1%）どおりに取ると実効レバレッジの上限（5倍）を超える取引は、数量を減らすか、"
        "1,000通貨も取れなければ見送ります。それぞれの取引を、決済したときの数量で記録しています"
        "（あとで設定を変えても、記録は書き換えません）。"
    )
    if len(adj):
        st.dataframe(pd.DataFrame({
            "エントリー（日本時間）": (pd.DatetimeIndex(adj["entry_time"]) + JST).strftime("%Y/%m/%d %H:%M"),
            "通貨ペア": adj["pair"], "時間足": adj["tf"], "戦略": adj["source"],
            "方向": adj["direction"].map({1: "買い", -1: "売り"}), "扱い": adj["sizing_status"],
            "本来の数量": adj["requested_units"].map(lambda u: f"{u:,.0f}"), "実際の数量": adj["units"].map(lambda u: f"{u:,.0f}"),
            "理由": adj["note"], "損益(pips)": adj["pips"].map(lambda x: f"{x:+.1f}"),
        }), hide_index=True, width="stretch")
    else:
        st.info("見送り・数量を減らした取引はまだありません。")
