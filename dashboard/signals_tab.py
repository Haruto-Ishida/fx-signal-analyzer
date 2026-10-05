"""タブ「現在のシグナル」（フェーズ1）。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fxtool import commentary, data, sizing, strategies, tracking
from fxtool.config import MIN_UNITS

from .common import BUY_COLOR, NEUTRAL_COLOR, SELL_COLOR, Context, color_decision


def render(ctx: Context) -> None:
    P, risk, ind, n_conf, pip, digits = ctx.params, ctx.risk, ctx.ind, ctx.n_conf, ctx.pip, ctx.digits
    views = [strategies.evaluate(ind, n_conf, s, P, risk, ctx.max_bars) for s in strategies.STRATEGIES]
    last_c = ind.iloc[n_conf - 1]

    # ---------------- 相場の解説（フェーズ5） ----------------
    c = commentary.build(
        ind, n_conf, views, ctx.weights, ctx.ml_status,
        ctx.alerts if not ctx.alerts.empty else pd.DataFrame(columns=["source", "message"]),
        ctx.verdicts if not ctx.verdicts.empty else pd.DataFrame(columns=["source", "verdict"]),
        ctx.tf_key, pip, digits, P, risk, data.now_utc(), ctx.label,
    )
    with st.container(border=True):
        st.markdown("#### 📝 いまの相場の解説")
        st.markdown(f"**{c.headline}**")
        heading = "判断の根拠" if c.headline.startswith("様子見") else "なぜこのシグナルか"
        st.markdown(f"**{heading}**\n" + "\n".join(f"- {r}" for r in c.reasons))
        st.markdown("**注意点**\n" + "\n".join(f"- {x}" for x in c.cautions))
        st.caption(
            f"{ctx.label(ind.index[n_conf - 1])} の確定足までの数値から、決まったルールで自動作成した文章です（AIによる文章生成ではありません）。"
            "売買を勧めるものではありません。"
            + ("" if not ctx.verdicts.empty else " 戦略の検証結果がまだ計算されていないため、検証にもとづく注意点は含まれていません。")
        )

    counts = {k: sum(v.decision == k for v in views) for k in ("買い", "売り", "様子見")}
    line = f"**戦略の判断**：買い {counts['買い']}・売り {counts['売り']}・様子見 {counts['様子見']}"
    if ctx.weights:
        score = tracking.weighted_score({v.strategy.key: v.decision for v in views}, ctx.weights)
        line += f"　｜　**重み付き総合スコア**：{score:+.2f}（{tracking.score_label(score)}）"
    st.markdown(line)
    if ctx.weights:
        st.caption("重み付き総合スコアは、各戦略の判断（買い +1・売り −1・様子見 0）を、直近の成績で決めた重みで平均したもの（−1〜+1）です。"
                   "どの戦略にもはっきりした優位性は確認できていないので、あくまで判断の目安です。")
    else:
        st.caption("戦略の重み付けは停止中です（以前の重みは最終テスト期間のデータから計算していたため。"
                   "フェーズ4の改修で、本番の記録にもとづく方法に作り直します）。")

    # ---------------- 戦略ごとのシグナル ----------------
    def status_text(v: strategies.StrategyView) -> str:
        o = v.outcome
        if o is None:
            return "—"
        if o.status == "進行中":
            return f"進行中（{v.unrealized / pip:+.1f} pips）"
        pips = v.direction * (o.exit_price - v.entry) / pip
        name = "時間切れ" if o.status == "期限" else o.status
        return f"{name}に到達 {ctx.label(o.exit_time)}（{pips:+.1f} pips）"

    def size_text(v: strategies.StrategyView) -> str:
        if v.decision not in ("買い", "売り"):
            return "—"
        s = sizing.position_size(ctx.balance_jpy, ctx.risk_pct, v.entry, v.sl, ctx.ticker, ctx.usdjpy)
        if s is None:
            return "円換算できません"
        if s.units == 0:
            return s.note
        return f"{s.units:,}通貨（損切り時 −{s.loss_at_stop_jpy:,.0f}円）"

    rows = []
    for v in views:
        has = v.signal_time is not None
        rows.append({
            "戦略": v.strategy.name,
            "タイプ": v.strategy.kind,
            "判断": v.decision,
            "直近のシグナル": f"{'買い' if v.direction > 0 else '売り'}　{ctx.label(v.signal_time)}" if has else "なし",
            "何本前": v.bars_ago if has else None,
            "エントリー（次の足の始値）": (ctx.fmt_price(v.entry) + ("" if v.entry_fixed else "（見込み）")) if has else "—",
            "損切り": f"{ctx.fmt_price(v.sl)}（{-risk.sl_atr * v.atr / pip:.1f} pips）" if has else "—",
            "利確": f"{ctx.fmt_price(v.tp)}（+{risk.tp_atr * v.atr / pip:.1f} pips）" if has else "—",
            "数量の目安": size_text(v),
            "その後の状況": status_text(v),
        })
    st.dataframe(pd.DataFrame(rows).style.map(color_decision, subset=["判断"]), hide_index=True, width="stretch")
    st.caption(
        "シグナルは確定した足だけで判定します（未確定の足では出しません）。"
        "エントリーはシグナルの**次の足の始値**です（次の足がまだ始まっていない間は、シグナルの足の終値を「見込み」として表示）。"
        f"決済は**トリプルバリア**：損切り ATR×{risk.sl_atr:g}、利確 ATR×{risk.tp_atr:g}、"
        f"{ctx.max_bars}本で決着しなければ時間切れ。シグナルから {risk.valid_bars} 本以内で、"
        "まだ決着していないものを「買い／売り」、それ以外を「様子見」としています。"
        "1本の足の中で損切りと利確の両方に届いた場合は、損切りが先とみなしています。"
        f"数量の目安は、損切りになったときの損失が口座残高 {ctx.balance_jpy:,.0f}円 の {ctx.risk_pct:g}% 以内に収まる数量です"
        f"（{MIN_UNITS:,}通貨単位で切り捨て。注文はしません）。"
    )

    # ---------------- チャート ----------------
    c1, c2 = st.columns([2, 1])
    chart_key = c1.selectbox(
        "チャートに表示する戦略", [s.key for s in strategies.STRATEGIES],
        format_func=lambda k: strategies.STRATEGY_BY_KEY[k].name,
    )
    n_show = c2.slider("表示本数", 50, 1000, 300 if ctx.tf_key == "1h" else 250, step=50)

    strat = strategies.STRATEGY_BY_KEY[chart_key]
    view = next(v for v in views if v.strategy.key == chart_key)
    win = ind.iloc[-n_show:]
    x = ctx.labels(win.index)
    sig = strategies.signals(ind, strat, P, n_conf).loc[win.index]
    offset = (win["high"] - win["low"]).mean() * 0.8

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=("価格", f"RSI（{P.rsi_period}）", "MACD", f"ATR（{P.atr_period}）"),
    )
    fig.add_trace(go.Candlestick(
        x=x, open=win["open"], high=win["high"], low=win["low"], close=win["close"], name="価格",
        increasing_line_color=BUY_COLOR, decreasing_line_color=SELL_COLOR,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=win["sma_s"], name=f"SMA{P.sma_short}", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=win["sma_l"], name=f"SMA{P.sma_long}", line=dict(color="#2962ff", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=win["bb_up"], name=f"BB ±{P.bb_std:g}σ", legendgroup="bb",
                             line=dict(color=NEUTRAL_COLOR, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=win["bb_lo"], name="BB", legendgroup="bb", showlegend=False,
                             line=dict(color=NEUTRAL_COLOR, width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(158,158,158,0.08)"), row=1, col=1)
    if chart_key == "breakout":
        fig.add_trace(go.Scatter(x=x, y=win["dc_hi"], name=f"直前{P.donchian_period}本の高値・安値", legendgroup="dc",
                                 line=dict(color="#8d6e63", width=1, shape="hv")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=win["dc_lo"], name="dc", legendgroup="dc", showlegend=False,
                                 line=dict(color="#8d6e63", width=1, shape="hv")), row=1, col=1)

    buys, sells = win[sig == 1], win[sig == -1]
    fig.add_trace(go.Scatter(
        x=ctx.labels(buys.index), y=buys["low"] - offset, mode="markers", name=f"買いシグナル（{strat.name}）",
        marker=dict(symbol="triangle-up", size=13, color=BUY_COLOR, line=dict(width=1, color="white")),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ctx.labels(sells.index), y=sells["high"] + offset, mode="markers", name=f"売りシグナル（{strat.name}）",
        marker=dict(symbol="triangle-down", size=13, color=SELL_COLOR, line=dict(width=1, color="white")),
    ), row=1, col=1)

    # 直近シグナルのエントリー・損切り・利確ライン
    if view.signal_time is not None and view.signal_time >= win.index[0]:
        end_t = view.outcome.exit_time if view.outcome.exit_time is not None else win.index[-1]
        xs = [ctx.label(view.signal_time), ctx.label(end_t)]
        for y, name, color in (
            (view.entry, "エントリー目安", "#607d8b"), (view.sl, "損切り", SELL_COLOR), (view.tp, "利確", BUY_COLOR),
        ):
            fig.add_trace(go.Scatter(
                x=xs, y=[y, y], mode="lines+text", name=name, line=dict(color=color, width=1.5, dash="dash"),
                text=["", f"{name} {y:.{digits}f}"], textposition="top left", textfont=dict(color=color),
                showlegend=False,
            ), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=win["rsi"], name="RSI", line=dict(color="#7e57c2", width=1.5)), row=2, col=1)
    fig.add_hline(y=P.rsi_high, line=dict(color=SELL_COLOR, dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=P.rsi_low, line=dict(color=BUY_COLOR, dash="dash", width=1), row=2, col=1)

    fig.add_trace(go.Bar(x=x, y=win["macd_hist"], name="ヒストグラム", showlegend=False,
                         marker_color=[BUY_COLOR if h >= 0 else SELL_COLOR for h in win["macd_hist"].fillna(0)]),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=win["macd"], name="MACD", line=dict(color="#2962ff", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=win["macd_sig"], name="シグナル線", line=dict(color="#ff9800", width=1.5)), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=win["atr"] / pip, name="ATR（pips）", line=dict(color="#795548", width=1.5)), row=4, col=1)

    fig.update_xaxes(type="category", nticks=8, rangeslider_visible=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="pips", row=4, col=1)
    fig.update_layout(
        height=900, margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        ("時刻は日本時間です。" if ctx.tf_key == "1h" else "日足の日付は Yahoo Finance の区切り（ロンドン時間基準）です。")
        + (" 一番右の足はまだ確定していません。" if n_conf < len(ind) else "")
        + " 土日などの休場時間は詰めて表示しています。"
    )

    # ---------------- 指標の現在値 ----------------
    st.markdown(f"#### 指標の値（最新の確定足：{ctx.label(ind.index[n_conf - 1])}）")
    trend = "上昇" if last_c["sma_s"] > last_c["sma_l"] else "下降"
    if last_c["rsi"] < P.rsi_low:
        rsi_state = "売られすぎ"
    elif last_c["rsi"] > P.rsi_high:
        rsi_state = "買われすぎ"
    else:
        rsi_state = "中立"
    bb_pos = (last_c["close"] - last_c["bb_lo"]) / (last_c["bb_up"] - last_c["bb_lo"]) * 100
    ind_table = pd.DataFrame([
        {"指標": f"SMA{P.sma_short} / SMA{P.sma_long}", "値": f"{last_c['sma_s']:.{digits}f} / {last_c['sma_l']:.{digits}f}",
         "状態": f"短期線が長期線より{'上' if trend == '上昇' else '下'}（{trend}トレンド）"},
        {"指標": f"RSI（{P.rsi_period}）", "値": f"{last_c['rsi']:.1f}", "状態": rsi_state},
        {"指標": "MACD / シグナル線", "値": f"{last_c['macd']:.4f} / {last_c['macd_sig']:.4f}",
         "状態": "MACDがシグナル線より上（上向きの勢い）" if last_c["macd"] > last_c["macd_sig"] else "MACDがシグナル線より下（下向きの勢い）"},
        {"指標": f"ボリンジャー（{P.bb_period}, ±{P.bb_std:g}σ）",
         "値": f"{last_c['bb_lo']:.{digits}f} 〜 {last_c['bb_up']:.{digits}f}",
         "状態": f"バンド内の位置 {bb_pos:.0f}%（0%=下限、100%=上限）"},
        {"指標": f"ATR（{P.atr_period}）", "値": f"{last_c['atr']:.{digits}f}", "状態": f"{last_c['atr'] / pip:.1f} pips"},
        {"指標": f"直前{P.donchian_period}本の高値 / 安値", "値": f"{last_c['dc_hi']:.{digits}f} / {last_c['dc_lo']:.{digits}f}", "状態": ""},
    ])
    st.dataframe(ind_table, hide_index=True, width="stretch")

    with st.expander("📘 戦略ルールの説明"):
        st.dataframe(
            pd.DataFrame([{"戦略": s.name, "タイプ": s.kind, "ルール": s.rule} for s in strategies.STRATEGIES]),
            hide_index=True, width="stretch",
        )
        st.markdown(
            "- **順張り**：トレンド（流れ）に乗る考え方。トレンドが続く相場で強く、行ったり来たりの相場では損切りが続きやすい。\n"
            "- **逆張り**：行き過ぎた値動きが戻ることを狙う考え方。レンジ相場で強く、強いトレンドの相場では逆らって負けやすい。\n"
            "- 戦略同士の判断が食い違うのは普通のことです。過去にどの戦略が機能したかは「バックテスト」タブで確認できます。"
        )
