"""タブ「機械学習予測」（フェーズ3）。"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import backtest as bt
from fxtool import data, ml, trials
from fxtool.config import DEFAULT_SLIPPAGE, DEFAULT_SPREAD, ML_HORIZON, ML_THRESHOLD, RiskParams

from .common import BUY_COLOR, NEUTRAL_COLOR, SELL_COLOR, Context, fmt_pct, fmt_pf, fmt_pips


@st.cache_data(show_spinner=False, max_entries=16)
def run(pair: str, ind: pd.DataFrame, tf_key: str, kind: str, h: int, threshold: float, risk: RiskParams,
        costs: bt.Costs, pip: float, k: int, holdout_start: pd.Timestamp) -> ml.ModelEval:
    """開発用データだけで検証し、試した設定を記録する（同じ設定はキャッシュされ、二重に記録しない）。"""
    ev = ml.evaluate_model(ind, tf_key, kind, h, threshold, risk, costs, pip, k=k, holdout_start=holdout_start)
    con = data.connect()
    try:
        trials.record(con, [ml.trial_row(ev, k, 0.5)], pair=pair, tf=tf_key, origin="dashboard", now=data.now_utc(),
                      data_from=ind.index[0], data_to=ind.index[-1], n_bars=len(ind))
    finally:
        con.close()
    ev.model = None  # 画面では使わない（キャッシュを軽くする）
    return ev


def horizon_label(tf_key: str, h: int) -> str:
    return f"{h}本後（{h}時間後）" if tf_key == "1h" else f"{h}本後（{h}営業日後）"


def costs_from_state(pair: str) -> bt.Costs:
    """バックテストタブで設定したコストを使う（未設定なら初期値）。"""
    return bt.Costs(st.session_state.get(f"spread_{pair}", DEFAULT_SPREAD[pair]),
                    st.session_state.get("slippage", DEFAULT_SLIPPAGE))


def render(ctx: Context) -> None:
    st.markdown(
        "テクニカル指標などを特徴量にして、**h本後の終値が今より上がっているか**を確率で予測します。"
        "まだ見ていない期間での成績を基準と比べ、**予測力が確認できない場合はシグナルを出しません。**"
    )
    with st.container(border=True):
        c = st.columns(4)
        kind = c[0].radio("モデル", list(ml.MODELS), format_func=lambda k: ml.MODELS[k][0])
        h = c[1].selectbox("予測する先", ml.HORIZONS[ctx.tf_key], format_func=lambda v: horizon_label(ctx.tf_key, v))
        threshold = c[2].slider("売買する確信度（確率がこれ以上で買い、1−これ以下で売り）", 0.51, 0.70, 0.55, 0.01)
        k = c[3].slider("検証の区間数", 3, 8, 5)
        costs = costs_from_state(ctx.pair)
        st.caption(
            f"取引コスト {costs.round_trip:.1f} pips/回（バックテストタブの設定と共通）、損切り ATR×{ctx.risk.sl_atr:g}・"
            f"利確 ATR×{ctx.risk.tp_atr:g}、保有は最大 {h} 本。"
        )

    dev = ctx.dev
    st.info(
        f"🔒 検証は開発用データ（{ctx.label(dev.index[0])} 〜 {ctx.label(dev.index[-1])}）だけで行います。"
        f"{ctx.label(ctx.boundary.start)} 以降の最終テスト期間は使いません。"
    )
    if not st.session_state.get("ml_on"):
        if st.button("▶ 学習して検証する", type="primary"):
            st.session_state["ml_on"] = True
            st.rerun()
        st.info("1時間足で数秒〜10秒ほどかかります。一度実行すると、以後は設定を変えるたびに自動で再計算します。"
                "設定を変えて実行するたびに、試した組み合わせとして記録されます。")
        return

    with st.spinner("学習・検証しています…"):
        evals = {kk: run(ctx.pair, dev, ctx.tf_key, kk, h, threshold, ctx.risk, costs, ctx.pip, k, ctx.boundary.start)
                 for kk in ml.MODELS}
    ev = evals[kind]

    # ---------------- 判定と現在の予測 ----------------
    if ev.ok:
        st.success(
            f"### ✅ {ev.name}：3つの条件をすべて満たしました\n"
            "まだ見ていない期間で基準を上回りました。ただし、複数の設定を試した中での結果なので、"
            "偶然の可能性は残ります。フェーズ4の記録で継続的に確かめる必要があります。"
        )
    else:
        failed = sum(not passed for _, passed, _ in ev.checks)
        st.warning(
            f"### ⚠️ {ev.name}：今回はシグナルを控えています\n"
            "**故障やエラーではありません。** 予測の精度が基準に届かなかったため、ツールの設計どおり、"
            f"無理にシグナルを出さないようにしています（下の3つの条件のうち {failed} つが未達）。"
            "為替の値動きは予測が非常に難しく、この結果になるのはよくあることです。"
        )
    for name, passed, detail in ev.checks:
        st.markdown(f"{'✅' if passed else '❌'} **{name}** — {detail}")

    # 現在の予測は、本番モデル（設定を固定し、最新データまで使って学習したもの）の記録から表示する
    prod = next((m for m in ctx.ml_status if m.model == kind), None)
    if h == ML_HORIZON and abs(threshold - ML_THRESHOLD) < 1e-9 and prod is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{horizon_label(ctx.tf_key, h)}に上がる確率（本番モデル）", f"{prod.p:.1%}")
        m2.metric("シグナル", prod.decision)
        m3.metric("予測の基準にした足", ctx.label(prod.bar_time))
        st.caption(
            "本番モデルは、この設定のまま最新データまで使って学習し直したものです（最終テスト期間も学習に含みますが、"
            "設定を選ぶ判断には使っていません）。本番モデルの実力は「記録と成績」タブの本番の記録で評価します。"
            + ("" if prod.ok else " 開発用データでの検証で条件を満たしていないため、確率は参考値です。売買の判断には使わないでください。")
        )
    else:
        st.caption(
            f"現在の予測は、本番で使っている設定（{horizon_label(ctx.tf_key, ML_HORIZON)}・確信度 {ML_THRESHOLD:.0%}）のときだけ表示します。"
            "ここで設定を変えて検証した結果は、試した組み合わせとして記録されますが、本番の予測には使いません。"
        )

    # ---------------- 正解率の比較 ----------------
    st.markdown(f"### ① 方向の正解率（検証期間 {ev.n:,} 件）")
    rows = []
    for e in evals.values():
        rows.append({
            "予測方法": e.name + ("（選択中）" if e.kind == kind else ""),
            "正解率": f"{e.acc:.1%}",
            "95%の範囲": f"{e.acc_ci[0]:.1%} 〜 {e.acc_ci[1]:.1%}",
            "対数損失": f"{e.logloss:.4f}",
            "AUC": "—" if math.isnan(e.auc) else f"{e.auc:.3f}",
        })
    rows += [
        {"予測方法": "常に「多い方」を予測（基準）", "正解率": f"{ev.majority_acc:.1%}", "95%の範囲": "",
         "対数損失": f"{ev.base_logloss:.4f}", "AUC": "0.500"},
        {"予測方法": "コイン投げ（基準）", "正解率": "50.0%", "95%の範囲": "", "対数損失": f"{math.log(2):.4f}", "AUC": "0.500"},
        {"予測方法": "直前と同じ方向（基準）", "正解率": f"{ev.momentum_acc:.1%}", "95%の範囲": "", "対数損失": "—", "AUC": "—"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"「多い方」は、各学習期間で上昇・下落のどちらが多かったかで決めます。95%の範囲は、予測先が{h}本先で正解ラベルが重なり合う分を考慮し、"
        f"件数を {ev.n_eff:,.0f} 件とみなして計算しています。対数損失は確率の当たり具合で、小さいほど良く、"
        "基準の行は「学習期間の上昇割合をそのまま確率として出し続けた場合」です。AUCは0.5なら当てずっぽうと同じです。"
    )

    # ---------------- 売買の成績 ----------------
    st.markdown("### ② 予測に従って売買した場合（コスト込み）")
    tm = ev.trade_metrics
    r = ev.random
    st.dataframe(pd.DataFrame([
        {"方法": f"{ev.name}（確信度 {threshold:.0%}）", "取引回数": f"{tm['n']:,}", "勝率": fmt_pct(tm["win_rate"]),
         "損益合計(pips)": fmt_pips(tm["total"]), "平均(pips/回)": fmt_pips(tm["avg"]), "PF": fmt_pf(tm["pf"]),
         "最大DD(pips)": fmt_pips(tm["max_dd"]), "ランダムを上回った割合": fmt_pct(ev.beat)},
        {"方法": "何もしない", "取引回数": "0", "勝率": "—", "損益合計(pips)": "±0.0", "平均(pips/回)": "—", "PF": "—",
         "最大DD(pips)": "±0.0", "ランダムを上回った割合": "—"},
        {"方法": "ランダム売買（同じ頻度・中央値）", "取引回数": f"{r['n'].median():,.0f}", "勝率": fmt_pct(r["win_rate"].median()),
         "損益合計(pips)": fmt_pips(r["total"].median()), "平均(pips/回)": fmt_pips(r["avg"].median()),
         "PF": fmt_pf(r["pf"].median()), "最大DD(pips)": fmt_pips(r["max_dd"].median()), "ランダムを上回った割合": "50%"},
    ]), hide_index=True, width="stretch")
    st.caption(
        "検証期間の各足で予測し、確信度を超えたときだけ次の足の始値でエントリー、"
        f"損切り・利確に届くか {h} 本経ったら決済します（フェーズ2と同じエンジン）。"
        "取引回数が多いほどコストが積み重なるため、少しの予測力ではコストに負けることがあります。"
    )

    # ---------------- グラフ ----------------
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**予測確率と実際の上昇割合**")
        cal = ml.calibration(ev.pred)
        fig = go.Figure()
        lo = min(cal["予測確率"].min(), cal["実際に上がった割合"].min(), 0.4)
        hi = max(cal["予測確率"].max(), cal["実際に上がった割合"].max(), 0.6)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="理想（予測どおり）",
                                 line=dict(color=NEUTRAL_COLOR, dash="dash")))
        fig.add_trace(go.Scatter(x=cal["予測確率"], y=cal["実際に上がった割合"], mode="lines+markers", name=ev.name,
                                 line=dict(color="#2962ff"), customdata=cal["件数"],
                                 hovertemplate="予測 %{x:.1%} → 実際 %{y:.1%}（%{customdata}件）<extra></extra>"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="予測した上昇確率",
                          yaxis_title="実際に上がった割合", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("点が斜めの線に沿って右上がりなら、確率に意味があります。横ばいなら、確率の高低と実際の結果は無関係です。")
    with g2:
        st.markdown("**区間ごとの正解率**")
        fig = go.Figure()
        xs = [f"区間{i + 1}<br>{ctx.label(dev.index[f.test[0]])[:10]}〜" for i, f in enumerate(ev.folds)]
        fig.add_trace(go.Bar(x=xs, y=[f.acc for f in ev.folds], name=ev.name, marker_color="#2962ff"))
        fig.add_trace(go.Bar(x=xs, y=[f.majority_acc for f in ev.folds], name="多い方を予測", marker_color=NEUTRAL_COLOR))
        fig.add_hline(y=0.5, line=dict(color=SELL_COLOR, dash="dot", width=1))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), barmode="group", yaxis_tickformat=".0%",
                          yaxis_range=[0.4, 0.65], legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("期間によって正解率が基準を上回ったり下回ったりする場合、安定した予測力とは言えません。")

    st.markdown("**予測に効いている特徴量**（最後の検証区間で、その特徴量をシャッフルしたときに対数損失がどれだけ悪化するか）")
    imp = ev.importance.head(12)[::-1]
    fig = go.Figure(go.Bar(
        x=imp.values, y=[ml.feature_label(n) for n in imp.index], orientation="h",
        marker_color=[BUY_COLOR if v > 0 else SELL_COLOR for v in imp.values],
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="重要度（大きいほど予測に効いている）")
    st.plotly_chart(fig, width="stretch")
    st.caption("0付近や赤（マイナス）の特徴量は、予測の役に立っていない（むしろ邪魔している）ことを示します。")

    with st.expander("📘 検証の方法と限界"):
        st.markdown(
            f"- **学習と検証の分け方**：データの後半50%を {k} 区間に分け、各区間の予測には**それより前のデータだけ**で学習したモデルを使います。"
            f"学習データの最後の {h} 本は答えが検証期間の値動きを含むので、学習から外しています（パージ）。\n"
            "- **現在の予測**：本番モデル（本番の設定のまま、答えがわかっている最新の足まで使って学習し直したもの）の予測です。"
            "どのデータでどのモデルを作ったかは「記録と成績」タブのモデルの履歴に残しています。\n"
            "- **特徴量**：" + "、".join(ml.feature_label(c) for c in ev.importance.index) + "。"
            "値幅に関するものは ATR で割って、時期が違っても比べられるようにしています。時間帯は UTC 基準です。\n"
            "- **ハイパーパラメータは固定**です。検証結果を見ながら調整すると、それ自体が過去への合わせ込みになるためです。\n"
            "- **試した組み合わせの数に注意**：モデル2種類 × 予測先3種類 × 確信度…と試すほど、どれかが偶然条件を満たす可能性は高まります。"
            "条件を満たした設定も、フェーズ4で実際の結果を記録して確かめる必要があります。"
        )
