"""タブ「機械学習予測」（フェーズ3。研究ノートを反映した改修版）。"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import backtest as bt
from fxtool import data, indicators, ml, ml_eval, trials
from fxtool.config import DEFAULT_SLIPPAGE, DEFAULT_SPREAD, ML_HORIZON, IndicatorParams, RiskParams

from .common import BUY_COLOR, NEUTRAL_COLOR, SELL_COLOR, Context, fmt_pct, fmt_pf, fmt_pips


@st.cache_data(show_spinner=False, max_entries=8)
def run(pair: str, tf_key: str, ohlc: pd.DataFrame, mode: str, params: IndicatorParams, risk: RiskParams,
        threshold: float, width: float, k: int, costs: bt.Costs, pip: float, max_bars: int,
        holdout_start: pd.Timestamp) -> ml_eval.Report:
    """開発用データだけで3つのモデルを検証し、試した設定を記録してから、記録済みの試行すべての数で DSR を割り引く。"""
    if len(ohlc) and ohlc.index[-1] >= holdout_start:
        from fxtool.holdout import FinalTestAccessError
        raise FinalTestAccessError("最終テスト期間のデータが機械学習の検証に渡されています")
    ind = indicators.add_all(ohlc, params)
    if mode == "meta":
        smp = ml_eval.build_meta(ind, tf_key, params, risk, max_bars, costs.round_trip, pip)
    else:
        smp = ml_eval.build_direction(ind, tf_key, width, max_bars, costs, pip)
    rep = ml_eval.evaluate_all(smp, threshold, k)
    con = data.connect()
    try:
        trials.record(con, rep.trials, pair=pair, tf=tf_key, origin="dashboard", now=data.now_utc(),
                      data_from=ohlc.index[0], data_to=ohlc.index[-1], n_bars=len(ohlc))
        rep = ml_eval.finalize(rep, smp, trials.count(con, pair, tf_key))
    finally:
        con.close()
    for r in rep.results.values():
        r.last_fold = None  # 画面では使わない（キャッシュを軽くする）
    return rep


def costs_from_state(pair: str) -> bt.Costs:
    """バックテストタブで設定したコストを使う（未設定なら初期値）。"""
    return bt.Costs(st.session_state.get(f"spread_{pair}", DEFAULT_SPREAD[pair]),
                    st.session_state.get("slippage", DEFAULT_SLIPPAGE))


def n_passed(r: ml_eval.Result) -> str:
    return f"{sum(c[1] for c in r.checks)}/{len(r.checks)}"


def render(ctx: Context) -> None:
    st.markdown(
        "研究ノートの方法で、機械学習に予測力があるかを確かめます。**予測力が確認できない場合はシグナルを出しません。**\n"
        "- **メタラベリング（おすすめ）**：方向は5つの戦略のシグナルが決め、機械学習は「そのシグナルに乗るか、見送るか」だけを判断します。\n"
        "- **方向予測**：上下同じ幅のバリアのうち、上に先に届く確率を予測し、確信度を超えたら買い・売りします。\n"
        "- 正解（ラベル）は**トリプルバリア**：利確・損切り・時間切れのうち、どれに最初に届いたかで決めます。"
    )
    dev = ctx.dev
    st.info(
        f"🔒 検証は開発用データ（{ctx.label(dev.index[0])} 〜 {ctx.label(dev.index[-1])}）だけで行います。"
        f"{ctx.label(ctx.boundary.start)} 以降の最終テスト期間は使いません。"
    )
    with st.container(border=True):
        c = st.columns(4)
        mode = c[0].radio("方法", list(ml_eval.MODES), format_func=lambda m: ml_eval.MODES[m])
        if mode == "meta":
            threshold = c[1].slider("乗る確信度（勝つ確率がこれ以上なら乗る）", 0.30, 0.70,
                                    ml_eval.DEFAULT_THRESHOLD["meta"], 0.01, key="ml_th_meta")
            width = 1.5
            c[2].caption(f"利確 ATR×{ctx.risk.tp_atr:g}・損切り ATR×{ctx.risk.sl_atr:g}（左の設定と共通）")
        else:
            threshold = c[1].slider("売買する確信度（上に先に届く確率がこれ以上で買い、1−これ以下で売り）", 0.51, 0.70,
                                    ml_eval.DEFAULT_THRESHOLD["direction"], 0.01, key="ml_th_dir")
            width = c[2].number_input("バリアの幅（ATR × ?、上下同じ）", 0.5, 4.0, 1.5, 0.1)
        k = c[3].slider("検証の区間数", 3, 8, 5)
        costs = costs_from_state(ctx.pair)
        st.caption(
            f"時間切れ {ctx.max_bars}本・取引コスト {costs.round_trip:.1f} pips/回（バックテストタブの設定と共通）。"
            "モデルは ロジスティック回帰 → ランダムフォレスト → 勾配ブースティング の順に比べます（ハイパーパラメータは固定）。"
        )

    if not st.session_state.get("ml_on"):
        if st.button("▶ 学習して検証する", type="primary"):
            st.session_state["ml_on"] = True
            st.rerun()
        st.info("1つの方法につき20〜40秒ほどかかります。一度実行すると、以後は設定を変えるたびに自動で再計算します。"
                "設定を変えて実行するたびに、試した組み合わせとして記録されます。")
        return

    with st.spinner("3つのモデルを学習・検証しています…（20〜40秒）"):
        rep = run(ctx.pair, ctx.tf_key, dev[["open", "high", "low", "close"]], mode, ctx.params, ctx.risk,
                  threshold, width, k, costs, ctx.pip, ctx.max_bars, ctx.boundary.start)
    ad = rep.results[rep.adopted]

    # ---------------- 判定 ----------------
    if ad.ok:
        st.success(
            f"### ✅ {ad.name}：条件をすべて満たしました\n"
            f"まだ見ていない期間で基準を上回り、{rep.n_trials}通り試した数で割り引いても偶然ではない確率が95%以上でした。"
            "それでも将来は保証されないため、本番の記録で確かめ続けます。"
        )
    else:
        failed = sum(not c[1] for c in ad.checks)
        st.warning(
            f"### ⚠️ 採用モデル（{ad.name}）：今回はシグナルを控えています\n"
            "**故障やエラーではありません。** 予測力が基準に届かなかったため、ツールの設計どおり、"
            f"無理にシグナルを出さないようにしています（{len(ad.checks)}つの条件のうち {failed} つが未達）。"
        )
    for name, passed, detail in ad.checks:
        st.markdown(f"{'✅' if passed else '❌'} **{name}** — {detail}")

    # ---------------- モデルの比較（段階的に） ----------------
    st.markdown(f"### ① モデルの比較（検証区間 {ad.n:,} サンプル、実質 {ad.n_eff:,.0f} 件）")
    rows = []
    for r in rep.results.values():
        tm = r.trade_metrics
        rows.append({
            "モデル": ("👉 " if r.kind == rep.adopted else "") + r.name,
            "正解率": f"{r.acc:.1%}",
            "対数損失（基準）": f"{r.logloss:.4f}（{r.base_logloss:.4f}）",
            "AUC": "—" if math.isnan(r.auc) else f"{r.auc:.3f}",
            "売買回数": f"{tm['n']:,}" + ("（参考値）" if 0 < tm["n"] < bt.MIN_TRADES_RELIABLE else ""),
            "損益合計(pips)": fmt_pips(tm["total"]),
            "1回あたり(pips)": fmt_pips(tm["avg"]),
            "ランダムを上回った割合": fmt_pct(r.beat),
            "DSR": fmt_pct(r.dsr.get("dsr", math.nan)),
            "条件": n_passed(r),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**採用の流れ**（複雑なモデルは、単純なモデルにコスト後の成績で勝ったときだけ採用）：  \n"
                + "  \n".join(f"- {n}" for n in rep.ladder_notes))
    st.caption(
        f"サンプルは {rep.n_samples:,} 件で、正解（{'コスト後に利益' if mode == 'meta' else '上に先に届いた'}）の割合は "
        f"{rep.base_win_rate:.1%} です。ラベルの期間が重なるサンプルは独立ではないので、重なりに応じて学習の重みを下げ、"
        "正解率の95%範囲も「実質の件数」で計算しています。"
    )

    # ---------------- 基準との比較 ----------------
    st.markdown("### ② 基準との比較（コスト込み）")
    tm = ad.trade_metrics
    rnd = pd.Series(ad.random_totals)
    if mode == "meta":
        am = ad.all_metrics
        rows = [
            {"方法": f"{ad.name}で見送り（確信度 {threshold:.0%}）", "取引回数": f"{tm['n']:,}", "勝率": fmt_pct(tm["win_rate"]),
             "損益合計(pips)": fmt_pips(tm["total"]), "1回あたり(pips)": fmt_pips(tm["avg"]), "PF": fmt_pf(tm["pf"])},
            {"方法": "全シグナルに乗る（見送りなし）", "取引回数": f"{am['n']:,}", "勝率": fmt_pct(am["win_rate"]),
             "損益合計(pips)": fmt_pips(am["total"]), "1回あたり(pips)": fmt_pips(am["avg"]), "PF": fmt_pf(am["pf"])},
            {"方法": "同じ数をランダムに見送る（中央値）", "取引回数": f"{tm['n']:,}", "勝率": "—",
             "損益合計(pips)": fmt_pips(rnd.median()), "1回あたり(pips)": "—", "PF": "—"},
            {"方法": "何もしない", "取引回数": "0", "勝率": "—", "損益合計(pips)": "±0.0", "1回あたり(pips)": "—", "PF": "—"},
        ]
        note = ("メタラベリングでは、各シグナルを独立した取引として数えています（同時に複数のポジションを持つことがあります）。"
                "比べる3つの方法は同じ扱いなので、比較は公平です。")
    else:
        rows = [
            {"方法": f"{ad.name}（確信度 {threshold:.0%}）", "取引回数": f"{tm['n']:,}", "勝率": fmt_pct(tm["win_rate"]),
             "損益合計(pips)": fmt_pips(tm["total"]), "1回あたり(pips)": fmt_pips(tm["avg"]), "PF": fmt_pf(tm["pf"])},
            {"方法": "同じ頻度のランダム売買（中央値）", "取引回数": "—", "勝率": "—",
             "損益合計(pips)": fmt_pips(rnd.median()), "1回あたり(pips)": "—", "PF": "—"},
            {"方法": "何もしない", "取引回数": "0", "勝率": "—", "損益合計(pips)": "±0.0", "1回あたり(pips)": "—", "PF": "—"},
        ]
        note = "方向予測の売買は、バックテストと同じエンジンで、ポジションを1つずつ持つ前提で測っています。"
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(note + f" ランダムな方法を {len(rnd)} 回試した中で、{ad.name}が上回った割合は {ad.beat:.0%} です。")

    # ---------------- Purged K-Fold ----------------
    pk = ad.purged
    if pk:
        st.markdown("### ③ 別の検証方法でも確かめる（Purged K-Fold）")
        st.dataframe(pd.DataFrame([
            {"検証方法": "ウォークフォワード（本番に近い）", "対数損失": f"{ad.logloss:.4f}", "基準": f"{ad.base_logloss:.4f}",
             "AUC": "—" if math.isnan(ad.auc) else f"{ad.auc:.3f}"},
            {"検証方法": f"Purged K-Fold（パージ＋エンバーゴ {pk['embargo']}本）", "対数損失": f"{pk['logloss']:.4f}",
             "基準": f"{pk['base_logloss']:.4f}", "AUC": "—" if math.isnan(pk["auc"]) else f"{pk['auc']:.3f}"},
        ]), hide_index=True, width="stretch")
        st.caption(
            "Purged K-Fold は、期間を分けてそれぞれを検証に使い、学習データからはラベルの期間が検証期間と重なるもの（パージ）と、"
            "検証期間の直後に始まるもの（エンバーゴ）を除く方法です。どちらの方法でも対数損失が基準より小さく、AUC が 0.5 を上回るなら、"
            "予測力がある可能性が高まります。片方だけ良い場合は、たまたまの可能性があります。"
        )

    # ---------------- グラフ ----------------
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**予測確率と実際の割合**")
        cal = ml.calibration(ad.pred)
        fig = go.Figure()
        lo = min(cal["予測確率"].min(), cal["実際に上がった割合"].min()) - 0.02
        hi = max(cal["予測確率"].max(), cal["実際に上がった割合"].max()) + 0.02
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="理想（予測どおり）",
                                 line=dict(color=NEUTRAL_COLOR, dash="dash")))
        fig.add_trace(go.Scatter(x=cal["予測確率"], y=cal["実際に上がった割合"], mode="lines+markers", name=ad.name,
                                 line=dict(color="#2962ff"), customdata=cal["件数"],
                                 hovertemplate="予測 %{x:.1%} → 実際 %{y:.1%}（%{customdata}件）<extra></extra>"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="予測した確率", yaxis_title="実際に" + ("勝った" if mode == "meta" else "上に先に届いた") + "割合",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")
    with g2:
        st.markdown("**区間ごとの正解率**")
        pr = ad.pred.assign(hit=(ad.pred["p"] >= 0.5) == (ad.pred["y"] == 1),
                            base_hit=(ad.pred["base_rate"] >= 0.5) == (ad.pred["y"] == 1))
        by = pr.groupby("fold")[["hit", "base_hit"]].mean()
        xs = [f"区間{f + 1}<br>{ctx.label(dev.index[ad.folds[f][0]])[:10]}〜" for f in by.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xs, y=by["hit"], name=ad.name, marker_color="#2962ff"))
        fig.add_trace(go.Bar(x=xs, y=by["base_hit"], name="多い方を予測", marker_color=NEUTRAL_COLOR))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), barmode="group", yaxis_tickformat=".0%",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")

    if ad.importance is not None:
        st.markdown(f"**予測に効いている特徴量**（{ad.name}、最後の検証区間でシャッフルしたときの対数損失の悪化）")
        imp = ad.importance.head(12)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[ml_eval.feature_label(n) for n in imp.index], orientation="h",
                               marker_color=[BUY_COLOR if v > 0 else SELL_COLOR for v in imp.values]))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="重要度（大きいほど予測に効いている）")
        st.plotly_chart(fig, width="stretch")

    # ---------------- 本番の予測 ----------------
    st.markdown("### 本番の予測について")
    mm = ctx.meta_model
    if mm:
        st.caption(
            f"本番では、**メタラベリング（確信度 {mm['threshold']:.0%}）**を使っています。モデルは開発用データの検証で選んだ"
            f"**{ml_eval.LADDER[mm['kind']][0]}**で、設定はそのまま、決着したすべてのラベル（{mm['n_train']:,}件）で学習し直しています"
            f"（最終学習 {(pd.Timestamp(mm['trained_at']) + pd.Timedelta(hours=9)):%Y/%m/%d %H:%M}）。"
            f"シグナルを出す条件は{'満たしています' if mm['ok'] else '満たしていないため、判定は参考値です'}。"
            "新しいシグナルが出るたびに「乗る／見送り」を記録し、その実力は「記録と成績」タブの本番の記録で評価します。"
            "この画面で設定を変えて検証した結果は、試した組み合わせとして記録されますが、本番には使いません。"
        )
    else:
        st.caption("本番の機械学習のモデルはまだ学習されていません（次の定期処理・画面の更新で学習されます）。")

    with st.expander("📘 検証の方法と限界"):
        st.markdown(
            f"- **ウォークフォワード**：開発用データの後半50%を {k} 区間に分け、各区間の予測には、それより前に**決着した**ラベルだけで学習したモデルを使います（パージ）。\n"
            "- **重複度の重み**：ラベルの期間が重なるサンプルは、重なり具合（独自性）に応じて学習の重みを下げています。\n"
            "- **モデルの順番**：ロジスティック回帰 → ランダムフォレスト → 勾配ブースティング。複雑なモデルは単純なモデルにコスト後の成績で勝ったときだけ採用します。"
            "ハイパーパラメータは固定です（検証結果を見ながら調整すると、それ自体が過去への合わせ込みになるため）。\n"
            f"- **試した数**：{rep.n_trials} 通り（戦略・機械学習・研究ノート反映前の推定分を含む）。DSR はこの数で割り引いています。\n"
            "- **未対応**：CPCV（複数の検証経路）と分数階差分は、研究ノートで「余裕があれば」とされているため、今回は入れていません。"
        )
