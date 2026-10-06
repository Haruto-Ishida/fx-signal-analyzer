"""タブ「有名手法」（要件定義書 12章）。候補の一覧と、ユーザーの承認後の検証。"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from fxtool import backtest as bt
from fxtool import costmodel, data, famous, jobs, tracking, trials
from fxtool import settings as user_settings
from fxtool.config import VERTICAL_BARRIER

from .backtest_tab import VERDICT_ICONS, fmt_ratio
from .common import Context, fmt_pct, fmt_pips

TF = "1d"  # 元のルールは日足で考えられている


def evaluate_and_save(pair: str) -> int:
    """有名手法を元のルールのまま、日足の開発用データだけで検証し、判定を保存する。試した組み合わせも記録する。"""
    con = data.connect()
    try:
        dev = data.load_dev(con, pair, TF)
        b = data.boundary(con, pair, TF)
        holdout_start = b.start
        costs, pip = jobs.default_costs(pair), jobs.pip_size(jobs.PAIRS[pair])
        s = user_settings.load()
        mult = costmodel.spread_multipliers(dev.index, TF, s.costs, s.event_filter)
        rules = jobs.exit_rules(TF)
        planned = bt.planned_trials(jobs.P, jobs.RISK, rules, famous.FAMOUS)
        n_trials = trials.count_with(con, pair, TF, planned, dev.index[0], dev.index[-1])
        rep = bt.evaluate_all(dev[["open", "high", "low", "close"]], jobs.P, jobs.RISK, rules, costs, pip, runs=200,
                              holdout_start=holdout_start, rw_paths=20, n_trials=n_trials, strategy_list=famous.FAMOUS,
                              spread_mult=mult)
        now = data.now_utc()
        trials.record(con, rep.trials, pair=pair, tf=TF, origin="dashboard", now=now,
                      data_from=dev.index[0], data_to=dev.index[-1], n_bars=len(dev), note="有名手法（元のルール）")
        con.executemany(
            """INSERT OR REPLACE INTO backtest_verdicts
               (pair, tf, source, computed_at, bar_time, verdict, reason, wf_total, wf_beat, dsr, pbo, n_trials, wf_n,
                wf_win_rate, wf_payoff)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(pair, TF, r.strategy.key, tracking.ts_str(now), tracking.ts_str(dev.index[-1]), r.verdict[0], r.verdict[1],
              r.wf.test_metrics["total"], r.wf_beat, jobs._num(r.dsr.get("dsr")), jobs._num(r.pbo.get("pbo")),
              rep.n_trials, r.wf.test_metrics["n"], jobs._num(r.wf.test_metrics["win_rate"]),
              jobs._num(r.wf.test_metrics["payoff"])) for r in rep.reports],
        )
        yearly = {r.strategy.key: r.yearly_full for r in rep.reports}
        st.session_state[f"famous_yearly_{pair}"] = yearly
        con.commit()
        return rep.n_trials
    finally:
        con.close()


def render(ctx: Context) -> None:
    st.markdown(
        "公開されている有名な売買手法を、**数値のルールとして再現して**、今の相場でも通用するかを検証します。"
        "「有名だから良い」とは考えず、ほかの戦略と同じ基準（コスト込み・ランダムとの比較・DSR・PBO）で正直に評価します。"
        "元のルールは日足で考えられているので、**日足**で検証します。"
    )
    con = data.connect()
    try:
        verdicts = tracking.load_verdicts(con, ctx.pair, TF)
        n_now = trials.count(con, ctx.pair, TF)
    finally:
        con.close()
    vmap = {r["source"]: r for r in verdicts.to_dict("records")}

    st.markdown("### 候補の一覧（元のルールのまま）")
    rows = []
    for s in famous.FAMOUS:
        v = vmap.get(s.key)
        rows.append({
            "手法": s.name, "考案者・出典": s.source, "考え方": s.rule,
            "再現できなかった点（簡略化）": "／".join(s.spec.get("simplifications", [])),
            "検証の判定": "未検証" if v is None else f"{VERDICT_ICONS.get(v['verdict'], '')} {v['verdict']}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("ルールの説明は、公開されている内容をもとに自分の言葉でまとめたものです（書籍などの文章は転載していません）。")

    tested = [s for s in famous.FAMOUS if s.key in vmap]
    if not tested:
        st.info(
            f"まだ検証していません。検証すると、試した組み合わせが {len(famous.FAMOUS)} 通り増えます"
            f"（日足の累計：いま {n_now} 通り）。候補の一覧を確認して、よければ下のボタンを押してください。"
        )
    if st.button(f"✅ 候補を承認して、元のルールのまま検証する（{ctx.pair}・日足）", key="famous_run"):
        with st.spinner("検証しています…（1〜2分）"):
            evaluate_and_save(ctx.pair)
        st.rerun()

    if tested:
        st.markdown("### 検証の結果（日足・開発用データのウォークフォワード）")
        st.dataframe(pd.DataFrame([{
            "手法": s.name,
            "判定": f"{VERDICT_ICONS.get(vmap[s.key]['verdict'], '')} {vmap[s.key]['verdict']}",
            "検証区間の損益(pips)": fmt_pips(vmap[s.key]["wf_total"]),
            "取引回数": "—" if pd.isna(vmap[s.key]["wf_n"]) else f"{int(vmap[s.key]['wf_n'])}",
            "勝率": fmt_pct(vmap[s.key]["wf_win_rate"]),
            "平均利益/平均損失": fmt_ratio(vmap[s.key]["wf_payoff"]),
            "ランダムを上回った割合": fmt_pct(vmap[s.key]["wf_beat"]),
            "DSR": fmt_pct(vmap[s.key]["dsr"]), "PBO": fmt_pct(vmap[s.key]["pbo"]),
            "理由": vmap[s.key]["reason"],
        } for s in tested]), hide_index=True, width="stretch")
        yearly = st.session_state.get(f"famous_yearly_{ctx.pair}")
        if yearly:
            st.markdown("**年ごとの損益（pips）**：公開当時は効いていても、広く知られた後に効かなくなっていないかを確認します。")
            table = {famous.FAMOUS_BY_KEY[k].name: {int(r.year): f"{r.total:+,.0f}" for r in y.itertuples()}
                     for k, y in yearly.items() if len(y)}
            st.dataframe(pd.DataFrame(table).sort_index(ascending=False), width="stretch")
        st.caption(f"DSR は、この時点までの日足の累計の試行数（{n_now} 通り）で割り引いています。")
