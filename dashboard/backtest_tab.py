"""タブ「バックテスト」（フェーズ2）。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxtool import backtest as bt
from fxtool import data, trials
from fxtool.config import DEFAULT_SLIPPAGE, DEFAULT_SPREAD, IndicatorParams, RiskParams

from .common import NEUTRAL_COLOR, STRATEGY_COLORS, Context, fmt_pct, fmt_pf, fmt_pips

VERDICT_ICONS = {"有望（要継続監視）": "🟢", "偶然の範囲": "🟡", "不安定": "🟡", "優位性なし": "🔴", "判断材料不足": "⚪"}


@st.cache_data(show_spinner=False, max_entries=6)
def run(pair: str, tf_key: str, ohlc: pd.DataFrame, params: IndicatorParams, risk: RiskParams, rules: bt.ExitRules,
        costs: bt.Costs, pip: float, runs: int, k: int, train_mult: int, holdout_start: pd.Timestamp) -> bt.Report:
    """開発用データだけで評価し、試した組み合わせを記録する（同じ設定はキャッシュされ、二重に記録しない）。"""
    rep = bt.evaluate_all(ohlc, params, risk, rules, costs, pip, runs, k, train_mult, holdout_start=holdout_start)
    con = data.connect()
    try:
        trials.record(con, rep.trials, pair=pair, tf=tf_key, origin="dashboard", now=data.now_utc(),
                      data_from=ohlc.index[0], data_to=ohlc.index[-1], n_bars=len(ohlc))
    finally:
        con.close()
    return rep


def holdout_notice(ctx: Context) -> None:
    """検証に使う期間と、使わない最終テスト期間を表示する。"""
    b = ctx.boundary
    dev = ctx.dev
    con = data.connect()
    try:
        n_trials = trials.count(con, ctx.pair, ctx.tf_key)
    finally:
        con.close()
    st.info(
        f"🔒 **検証に使う期間（開発用データ）**：{ctx.label(dev.index[0])} 〜 {ctx.label(dev.index[-1])}（{len(dev):,}本）。"
        f"**{ctx.label(b.start)} 以降は最終テスト期間**として、検証には一切使いません"
        f"（{b.fixed_at:%Y/%m/%d} に固定。それ以降の本番の記録を、最も信頼できる最終テストとして扱います）。  \n"
        f"⚠️ {b.touched}  \n"
        f"これまでに試した組み合わせ（この通貨ペア・時間足）：**{n_trials} 通り**"
        "（研究ノート反映前の推定分を含む。試すほど、どれかが偶然よく見える可能性が高まります）。"
    )


def metric_row(name: str, m: dict, beat: str = "—") -> dict:
    return {
        "戦略": name,
        "取引回数": f"{m['n']:,}",
        "勝率": fmt_pct(m["win_rate"]),
        "損益合計(pips)": fmt_pips(m["total"]),
        "平均(pips/回)": fmt_pips(m["avg"]),
        "PF": fmt_pf(m["pf"]),
        "最大DD(pips)": fmt_pips(m["max_dd"]),
        "ランダムを上回った割合": beat,
    }


def render(ctx: Context) -> None:
    st.markdown(
        "過去のデータで各戦略を売買したと仮定して成績を測ります。"
        "**取引コストを差し引いた成績**を、「何もしない」「ランダム売買」と比べます。"
    )
    with st.container(border=True):
        c = st.columns(4)
        spread = c[0].number_input("スプレッド（往復, pips）", 0.0, 20.0, DEFAULT_SPREAD[ctx.pair], 0.1,
                                   key=f"spread_{ctx.pair}")
        slip = c[1].number_input("スリッページ（片道, pips）", 0.0, 10.0, DEFAULT_SLIPPAGE, 0.1, key="slippage",
                                 help="注文した価格と実際に約定する価格のずれ。エントリーと決済の両方でかかるとみなします。")
        max_bars = c[2].number_input(
            "時間切れ（本。0 = 制限なし）", 0, 2000, ctx.max_bars, 1, key=f"max_bars_{ctx.tf_key}",
            help="トリプルバリアの「時間切れ」。エントリーからこの本数で利確・損切りに届かなければ、その足の終値で決済します。",
        )
        opposite = c[3].checkbox("反対シグナルでドテン", True,
                                 help="持っているポジションと逆のシグナルが出たら、決済して逆方向に入り直します。")
        d = st.columns(3)
        runs = d[0].selectbox("ランダム売買の試行回数", [100, 200, 500], index=1)
        k = d[1].slider("ウォークフォワードの区間数", 3, 8, 5)
        train_mult = d[2].slider("学習区間の長さ（検証区間の何倍か）", 1, 4, 2)
        st.caption(
            f"損切り ATR×{ctx.risk.sl_atr:g}・利確 ATR×{ctx.risk.tp_atr:g} は左の設定と共通です。"
            f"1回の取引のコスト：{spread + 2 * slip:.1f} pips（スプレッド + スリッページ×2）。スワップポイントは含みません。"
        )

    holdout_notice(ctx)
    if not st.session_state.get("bt_on"):
        if st.button("▶ バックテストを実行", type="primary"):
            st.session_state["bt_on"] = True
            st.rerun()
        st.info("計算には1時間足で10秒前後かかります。一度実行すると、以後は設定を変えるたびに自動で再計算します。"
                "設定を変えて実行するたびに、試した組み合わせとして記録されます。")
        return

    conf = ctx.dev  # 最終テスト期間より前だけ
    rules = bt.ExitRules(opposite_exit=opposite, max_bars=int(max_bars))
    costs = bt.Costs(spread, slip)
    with st.spinner("バックテスト中…（初回は10秒前後かかります）"):
        rep = run(ctx.pair, ctx.tf_key, conf[["open", "high", "low", "close"]], ctx.params, ctx.risk, rules, costs,
                  ctx.pip, runs, k, train_mult, ctx.boundary.start)

    n = len(conf)
    t_disp = ctx.to_display_time(conf.index)

    # ---------------- まとめ ----------------
    st.markdown("### まとめ（ウォークフォワードの検証区間での成績にもとづく判定）")
    for r in rep.reports:
        verdict, reason = r.verdict
        st.markdown(f"{VERDICT_ICONS[verdict]} **{r.strategy.name}**：{verdict} — {reason}")
    if not any(r.verdict[0].startswith("有望") for r in rep.reports):
        st.warning(
            "今回の条件では、まだ見ていない期間（検証区間）でランダム売買をはっきり上回った戦略はありませんでした。"
            "これらのシグナルだけに従って売買しても、長期的に利益が出るとは言えません。"
        )

    # ---------------- 開発用データ全体 ----------------
    st.markdown(f"### ① 開発用データ全体の成績（初期パラメータ）　{ctx.label(conf.index[rep.start])} 〜 {ctx.label(conf.index[-1])}")
    rows = [metric_row(r.strategy.name, r.full_metrics, fmt_pct(r.full_beat)) for r in rep.reports]
    rows.append(metric_row("何もしない", bt.metrics([])) | {"損益合計(pips)": "±0.0", "最大DD(pips)": "±0.0"})
    b = rep.baseline_random
    rows.append({
        "戦略": "ランダム売買（中央値）",
        "取引回数": f"{b['n'].median():,.0f}",
        "勝率": fmt_pct(b["win_rate"].median()),
        "損益合計(pips)": fmt_pips(b["total"].median()),
        "平均(pips/回)": fmt_pips(b["avg"].median()),
        "PF": fmt_pf(b["pf"].median()),
        "最大DD(pips)": fmt_pips(b["max_dd"].median()),
        "ランダムを上回った割合": "50%",
    })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "「ランダムを上回った割合」は、その戦略と同じ頻度でランダムに売買した試行（決済ルール・コストは同じ）のうち、"
        "損益合計で上回った割合です。95%以上なら偶然では出にくい成績と言えます。"
        "なお、初期パラメータは一般的によく使われる値ですが、過去のデータを見たうえでの結果なので、実力より良く見えている可能性があります。"
    )

    fig = go.Figure()
    q = rep.baseline_curves
    x = t_disp[rep.start:]
    fig.add_trace(go.Scatter(x=x, y=q[2][rep.start:], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[0][rep.start:], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(158,158,158,0.25)", name="ランダム売買の5〜95%の範囲", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[1][rep.start:], name="ランダム売買（中央値）",
                             line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dash")))
    for r, color in zip(rep.reports, STRATEGY_COLORS):
        curve = bt.equity_curve(r.full_trades["exit_i"], r.full_trades["pips"], n)
        fig.add_trace(go.Scatter(x=x, y=curve[rep.start:], name=r.strategy.name, line=dict(color=color, width=1.8)))
    fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOR, width=1))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="累積損益（pips）",
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

    # ---------------- ウォークフォワード ----------------
    st.markdown("### ② ウォークフォワード検証（過去への合わせ込みを除いた実力の目安）")
    st.markdown(
        f"データを時間順に {k} 回、「学習区間 → 検証区間」に分けます。学習区間でパラメータの候補から最も成績の良いものを選び、"
        "**まだ見ていない直後の検証区間**でそのまま使ったときの成績を測ります。"
        "学習区間に比べて検証区間の成績が大きく落ちる戦略は、過去のデータに合わせ込んでいるだけの可能性が高いです。"
    )
    rows = []
    for r in rep.reports:
        m = r.wf.test_metrics
        rows.append({
            "戦略": r.strategy.name,
            "評価": f"{VERDICT_ICONS[r.verdict[0]]} {r.verdict[0]}",
            "学習区間の平均(pips/回)": fmt_pips(r.wf.train_avg),
            "検証区間の平均(pips/回)": fmt_pips(m["avg"]),
            "検証 取引回数": f"{m['n']:,}",
            "検証 勝率": fmt_pct(m["win_rate"]),
            "検証 損益合計(pips)": fmt_pips(m["total"]),
            "検証 PF": fmt_pf(m["pf"]),
            "検証 最大DD(pips)": fmt_pips(m["max_dd"]),
            "黒字の区間": f"{r.wf.positive_folds}/{k}",
            "ランダムを上回った割合": fmt_pct(r.wf_beat),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("学習区間どうしは重なっているため、合計ではなく1取引あたりの平均で比べています。")

    folds = rep.reports[0].wf.folds
    fig = go.Figure()
    t0 = folds[0].test[0]
    for i, f in enumerate(folds):
        a, c_ = f.test
        fig.add_vrect(x0=t_disp[a], x1=t_disp[c_ - 1], fillcolor=NEUTRAL_COLOR, opacity=0.08 if i % 2 else 0.0,
                      line_width=0, annotation_text=f"区間{i + 1}", annotation_position="top left")
    for r, color in zip(rep.reports, STRATEGY_COLORS):
        t = r.wf.test_trades
        curve = bt.equity_curve(t["exit_i"], t["pips"], n)
        fig.add_trace(go.Scatter(x=t_disp[t0:], y=curve[t0:], name=r.strategy.name, line=dict(color=color, width=1.8)))
    fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOR, width=1))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="検証区間の累積損益（pips）",
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0))
    st.plotly_chart(fig, width="stretch")

    keys = [r.strategy.key for r in rep.reports]
    by_key = {r.strategy.key: r for r in rep.reports}
    name_of = lambda key: by_key[key].strategy.name  # noqa: E731

    with st.expander("区間ごとの詳細（選ばれたパラメータ）"):
        key = st.selectbox("戦略", keys, format_func=name_of, key="wf_detail")
        r = by_key[key]
        st.dataframe(pd.DataFrame([{
            "区間": i + 1,
            "学習期間": f"{ctx.label(conf.index[f.train[0]])} 〜 {ctx.label(conf.index[f.train[1] - 1])}",
            "検証期間": f"{ctx.label(conf.index[f.test[0]])} 〜 {ctx.label(conf.index[f.test[1] - 1])}",
            "選ばれたパラメータ": bt.describe_params(key, f.overrides),
            "学習 取引回数": f.train_metrics["n"],
            "学習 損益(pips)": fmt_pips(f.train_metrics["total"]),
            "検証 取引回数": len(f.test_trades),
            "検証 損益(pips)": fmt_pips(f.test_trades["pips"].sum()),
        } for i, f in enumerate(r.wf.folds)]), hide_index=True, width="stretch")
        st.caption("パラメータの候補：" + "、".join(bt.describe_params(key, ov) for ov in bt.PARAM_GRIDS[key]))

    with st.expander("取引の一覧（開発用データ全体・初期パラメータ）"):
        key = st.selectbox("戦略", keys, format_func=name_of, key="trade_list")
        t = by_key[key].full_trades
        if t.empty:
            st.write("取引はありませんでした。")
        else:
            st.dataframe(pd.DataFrame({
                "方向": t["dir"].map({1: "買い", -1: "売り"}),
                "シグナル": ctx.labels(conf.index[t["signal_i"]]),
                "エントリー": ctx.labels(conf.index[t["entry_i"]]),
                "エントリー価格": t["entry"].map(ctx.fmt_price),
                "損切り": t["sl"].map(ctx.fmt_price),
                "利確": t["tp"].map(ctx.fmt_price),
                "決済": ctx.labels(conf.index[t["exit_i"]]),
                "決済価格": t["exit"].map(ctx.fmt_price),
                "決済理由": t["reason"],
                "損益(pips)": t["pips"].round(1),
            }).iloc[::-1], hide_index=True, width="stretch")

    with st.expander("📘 指標の意味と、この検証の限界"):
        st.markdown(
            "- **勝率**：利益が出た取引の割合。損切りより利確の幅を広くしているので、勝率が50%を下回っても利益が出ることがあります。\n"
            "- **PF（プロフィットファクター）**：総利益 ÷ 総損失。1を超えればプラス。1.0〜1.1程度はコストや偶然で簡単に変わる水準です。\n"
            "- **最大DD（ドローダウン）**：累積損益のピークからの最大の落ち込み（決済ベース）。実際に耐える必要があった含み損の目安です。\n"
            "- **売買の前提**：シグナルの次の足の始値でエントリー。1本の足の中で損切りと利確の両方に届いた場合は損切りとみなします。"
            "足の中の細かい値動きはわからないため、損切り・利確はちょうどその価格で約定したと仮定しています（急変時のずれは考慮していません）。\n"
            "- **複数の戦略を同時に試している**ので、そのうち1つが偶然よく見えることは起こりえます。"
            "「有望」と出た戦略も、フェーズ4の継続的な記録で成績を確かめていく必要があります。"
        )
