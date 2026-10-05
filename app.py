"""FX シグナル分析ダッシュボード（表示と記録のみ。注文は一切しない）。起動: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from dashboard import backtest_tab, ml_tab, signals_tab, tracking_tab
from dashboard.common import JST, Context
from fxtool import data, indicators, jobs, tracking
from fxtool.config import (
    DEFAULT_BALANCE_JPY, DEFAULT_RISK_PCT, PAIR_NAMES, PAIRS, TIMEFRAMES, IndicatorParams, RiskParams, pip_size,
    price_digits,
)

st.set_page_config(page_title="FXシグナル分析", page_icon="📈", layout="wide")

P = IndicatorParams()

# ---------------- サイドバー ----------------
with st.sidebar:
    st.header("設定")
    pair = st.selectbox("通貨ペア", list(PAIRS), format_func=lambda k: f"{k}（{PAIR_NAMES[k]}）")
    tf_key = st.radio("時間足", list(TIMEFRAMES), format_func=lambda k: TIMEFRAMES[k].label, horizontal=True)

    st.subheader("損切り・利確の目安")
    risk = RiskParams(
        sl_atr=st.number_input("損切り（ATR × ?）", 0.5, 5.0, 1.5, step=0.1),
        tp_atr=st.number_input("利確（ATR × ?）", 0.5, 10.0, 3.0, step=0.1),
        valid_bars=st.number_input(
            "シグナルの有効期間（本）", 1, 20, 3,
            help="シグナルが出てからこの本数の間、損切り・利確に届いていなければ「買い／売り」と表示します。",
        ),
    )
    st.subheader("数量の目安")
    balance = st.number_input("口座残高（円）", 10_000, 1_000_000_000, DEFAULT_BALANCE_JPY, step=100_000)
    risk_pct = st.number_input(
        "1回の許容損失（残高の%）", 0.1, 2.0, DEFAULT_RISK_PCT, step=0.1,
        help="損切りになったときの損失が、残高のこの割合に収まる数量を表示します（研究ノートの目安は1〜2%以内）。",
    )
    force = st.button("今すぐデータを更新", width="stretch")

ticker = PAIRS[pair]
tf = TIMEFRAMES[tf_key]

# ---------------- データ ----------------
st.title("📈 FXシグナル分析")
st.caption(
    "⚠️ テクニカル指標から機械的に計算した**参考情報**です。将来の値動きを保証するものではありません。"
    "このツールは表示と記録のみを行い、注文は一切行いません。"
)

con = data.connect()
try:
    with st.spinner("データの確認と記録をしています…（再学習が必要なときは10〜20秒かかります）"):
        cyc = jobs.run_cycle(con, pair, tf_key, origin="dashboard", force=force)
    raw = data.load(con, pair, tf_key)
    fetched_at = data.last_fetch(con, pair, tf_key)
    boundary = data.boundary(con, pair, tf_key) if len(raw) >= 100 else None
    # 戦略の重みは、フェーズ4で本番の記録にもとづく方法に作り直すまで停止中（以前の重みは最終テスト期間から計算していた）
    weights: dict[str, float] = {}
    active_alerts = tracking.load_alerts(con, pair, tf_key, active_only=True)
    ml_status = tracking.latest_ml_status(con, pair, tf_key)
    verdicts = tracking.load_verdicts(con, pair, tf_key)
    usdjpy = data.latest_close(con, "USD/JPY")
finally:
    con.close()

if cyc.data_error:
    st.warning(f"データの更新に失敗したため、保存済みのデータで表示しています。（{cyc.data_error}）")
if cyc.error:
    st.warning(f"記録・再学習の処理でエラーが起きました（表示は続けます）：{cyc.error}")
for a in active_alerts.itertuples():
    (st.error if a.level == "critical" else st.warning)(
        f"🔔 {a.message}（{(a.created_at + JST):%Y/%m/%d %H:%M} から・詳しくは「記録と成績」タブ）"
    )
if raw.empty:
    st.error("保存済みのデータがありません。インターネット接続を確認して「今すぐデータを更新」を押してください。")
    st.stop()

ind = indicators.add_all(raw, P)
n_conf = len(data.confirmed(raw, tf))
if n_conf < P.sma_long + P.atr_period:
    st.error(f"確定済みの足が {n_conf} 本しかなく、指標を計算できません。")
    st.stop()

ctx = Context(pair, ticker, tf_key, tf, pip_size(ticker), price_digits(ticker), P, risk, ind, n_conf, weights,
              ml_status, active_alerts, verdicts, boundary, float(balance), float(risk_pct), usdjpy)

with st.sidebar:
    st.divider()
    st.caption(
        f"**保存済みデータ**：{len(raw):,} 本（{ctx.label(raw.index[0])} 〜 {ctx.label(raw.index[-1])}）  \n"
        f"**最終取得**：{(fetched_at + JST):%Y/%m/%d %H:%M}（日本時間）"
        + ("" if cyc.skipped else f"  \n今回の処理：{cyc.summary()}")
    )
    if boundary is not None:
        st.caption(f"🔒 **最終テスト期間**：{ctx.label(boundary.start)} 以降（検証には使いません）")
    q = data.quality(raw, tf)
    with st.expander("💾 データについて"):
        gap_line = (f"週末以外の空き（祝日の休場・欠損）{q['gaps']} 回、日曜の足の時刻（UTC）{q['sunday_hours']}"
                    if tf_key == "1h" else f"日曜の日付 {q['sunday']} 本、0時以外の時刻 {q['not_midnight']} 本")
        st.markdown(
            f"- 取得元：Yahoo Finance（yfinance、無料）。ティッカー `{ticker}`\n"
            "- 保存先：`data/fx.db`（SQLite）。2回目以降は不足分だけ取得して追記します。"
            f"自動更新の間隔は {tf.stale_after.total_seconds() / 60:.0f} 分です。\n"
            "- 1時間足は Yahoo Finance からは直近約2年分しか取れませんが、保存したデータは消えないので、使い続けるほど蓄積されます。\n"
            "- 無料データのため、ときどき欠損や異常値があります（高値<安値などの明らかな異常は取り込み時に除外しています）。"
            "欠損は補完していません。\n"
            f"- **データの確認**：重複 {q['duplicates']} 本、土曜の足 {q['saturday']} 本、{gap_line}、"
            f"値動きゼロ（始値＝高値＝安値＝終値）の足 {q['flat']} 本。"
        )

# ---------------- 概要 ----------------
last_c, prev_c = ind.iloc[n_conf - 1], ind.iloc[n_conf - 2]
has_forming = n_conf < len(ind)
st.subheader(f"{pair}（{PAIR_NAMES[pair]}）　{tf.label}")
m1, m2, m3 = st.columns(3)
m1.metric("現在値", f"{ind['close'].iat[-1]:.{ctx.digits}f}",
          f"{(ind['close'].iat[-1] - prev_c['close']) / ctx.pip:+.1f} pips（1本前の確定足比）" if has_forming
          else f"{(last_c['close'] - prev_c['close']) / ctx.pip:+.1f} pips（前の足比）")
m2.metric(f"ATR（{P.atr_period}）", f"{last_c['atr'] / ctx.pip:.1f} pips",
          help="1本の足の平均的な値幅。損切り・利確の幅の基準です。")
m3.metric("最新の確定足", ctx.label(ind.index[n_conf - 1]))

tab_signals, tab_backtest, tab_ml, tab_tracking = st.tabs(
    ["📡 現在のシグナル", "🧪 バックテスト", "🤖 機械学習予測", "📒 記録と成績"]
)
with tab_signals:
    signals_tab.render(ctx)
with tab_backtest:
    backtest_tab.render(ctx)
with tab_ml:
    ml_tab.render(ctx)
with tab_tracking:
    tracking_tab.render(ctx)
