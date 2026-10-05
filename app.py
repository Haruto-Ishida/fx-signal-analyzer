"""FX シグナル分析ダッシュボード（表示と記録のみ。注文は一切しない）。起動: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from dashboard import backtest_tab, famous_tab, ga_tab, ml_tab, paper_tab, signals_tab, tracking_tab
from dashboard.common import JST, Context
from fxtool import data, events, indicators, jobs, killswitch, tracking, trials
from fxtool import settings as user_settings
from fxtool.config import (
    DAILY_MAX_LOSS_R, DEFAULT_BALANCE_JPY, DEFAULT_RISK_PCT, PAIR_NAMES, PAIRS, TIMEFRAMES, WEEKLY_MAX_LOSS_R,
    IndicatorParams, RiskParams, pip_size, price_digits,
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
    user_cfg = user_settings.load()
    sz = user_cfg.sizing
    balance = st.number_input(
        "口座残高（円）", 1_000, 1_000_000_000, int(user_cfg.paper.initial_balance), step=10_000,
        help="数量の目安の計算に使う残高です。初期値はペーパートレードの初期残高（settings.toml の [paper]）です。",
    )
    risk_pct = sz.fixed_risk_pct
    st.caption(
        f"1回の損失の目安：検証の取引が100回以上の戦略は **{sz.kelly_fraction:g}倍ケリー**（上限 {sz.max_risk_pct:g}%）、"
        f"それ以外は **{sz.fixed_risk_pct:g}%**。実効レバレッジは **{sz.max_leverage:g}倍まで**。",
        help="ケリー基準：勝率と「平均利益 ÷ 平均損失」から計算する、資産の増え方が理論上いちばん大きくなる賭け金の割合。"
             "見積もりの誤差に弱いので、その一部（初期値は4分の1）だけを使います。数値は settings.toml で変更できます。",
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
    # 戦略の重みは本番の記録から計算したもの（最終テスト期間のバックテストから計算していた以前の重みは使わない）
    weights = {r.source: r.weight for r in tracking.latest_weights(con, pair, tf_key).itertuples()}
    active_alerts = tracking.load_alerts(con, pair, tf_key, active_only=True)
    verdicts = tracking.load_verdicts(con, pair, tf_key)
    usdjpy = data.latest_close(con, "USD/JPY")
    signal_log = tracking.load_signal_log(con, pair, tf_key)
    meta_model = jobs.latest_meta(con, pair, tf_key)
    n_trials_total = trials.count(con, pair, tf_key)
    paper_acc, kill = killswitch.check(con, data.now_utc(), user_cfg, usdjpy)
    sched_gap = jobs.schedule_gap(con, data.now_utc())
finally:
    con.close()
closed = signal_log[signal_log["status"].isin(tracking.DONE)]
# 本番の記録は初期値の損切り幅（jobs.RISK）で記録しているので、左の設定ではなく初期値で R を計算する
r_halt = tracking.loss_halt(closed, data.now_utc(), jobs.RISK.sl_atr, pip_size(ticker), DAILY_MAX_LOSS_R,
                            WEEKLY_MAX_LOSS_R)
# 停止の理由をまとめる：R で数える最大損失・キルスイッチ（ペーパー口座全体）・経済指標の前後
blackout = events.blackout(ticker, data.now_utc(), user_cfg.event_filter)
stop = [x for x in (r_halt.reason if r_halt.active else "", kill.reason,
                    f"経済指標の前後（{blackout.name}・日本時間 {blackout.jst_label()}）のため" if blackout else "") if x]
halt = tracking.Halt(bool(stop), " ／ ".join(stop), r_halt.day_r, r_halt.week_r)

for w in user_cfg.warnings:
    st.warning(f"⚙️ 設定ファイル：{w}")
if sched_gap is not None:
    st.warning(
        f"⏰ **自動更新（毎時）が {sched_gap.total_seconds() / 3600:.0f} 時間動いていません。**"
        "パソコンがスリープ・電源オフ・ログオフだったか、電源につながっていなかった可能性があります。"
        "その間に確定した足のシグナルは、次に動いたときにまとめて記録します（いま画面を開いたので、この通貨ペア・時間足は記録しました）。"
    )
if cyc.data_error:
    st.warning(f"データの更新に失敗したため、保存済みのデータで表示しています。（{cyc.data_error}）")
if cyc.error:
    st.warning(f"記録・再学習の処理でエラーが起きました（表示は続けます）：{cyc.error}")
need_update, update_msg = events.schedule_status(events.load_schedule(), data.now_utc() + JST)
if need_update:
    st.warning(f"📅 **FOMC・日銀会合の日程ファイルの更新が必要です**：{update_msg}")
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
              [], active_alerts, verdicts, boundary, float(balance), float(risk_pct), usdjpy,
              signal_log=signal_log, meta_model=meta_model, halt=halt, settings=user_cfg)
ctx.paper, ctx.kill = paper_acc, kill

with st.sidebar:
    st.divider()
    st.caption(
        f"**保存済みデータ**：{len(raw):,} 本（{ctx.label(raw.index[0])} 〜 {ctx.label(raw.index[-1])}）  \n"
        f"**最終取得**：{(fetched_at + JST):%Y/%m/%d %H:%M}（日本時間）"
        + ("" if cyc.skipped else f"  \n今回の処理：{cyc.summary()}")
    )
    if boundary is not None:
        st.caption(f"🔒 **最終テスト期間**：{ctx.label(boundary.start)} 以降（検証には使いません）")
    st.caption(
        f"🧮 **試した組み合わせ（累計）**：{n_trials_total:,} 通り",
        help="これまでに検証した戦略・パラメータ・モデルの設定の数です。たくさん試すほど、偶然よく見えるものが"
             "見つかりやすくなるので、成績はこの数で割り引いて評価します（DSR）。",
    )
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

tab_signals, tab_backtest, tab_ml, tab_tracking, tab_paper, tab_famous, tab_ga = st.tabs(
    ["📡 現在のシグナル", "🧪 バックテスト", "🤖 機械学習予測", "📒 記録と成績", "💰 ペーパー口座", "📚 有名手法", "🧬 GA探索"]
)
with tab_signals:
    signals_tab.render(ctx)
with tab_backtest:
    backtest_tab.render(ctx)
with tab_ml:
    ml_tab.render(ctx)
with tab_tracking:
    tracking_tab.render(ctx)
with tab_paper:
    paper_tab.render(ctx)
with tab_famous:
    famous_tab.render(ctx)
with tab_ga:
    ga_tab.render(ctx)
