"""SMA20/100 候補の確認の検証（事前登録の固定と、検証の仕組み）。実際のデータは使わない。"""
import json

import numpy as np
import pandas as pd
import pytest

from fxtool import backtest as bt
from fxtool import confirm, ga, trials

PRE = confirm.load_prereg()


def test_frozen_rule_has_not_changed():
    """事前登録したルールの計算が、登録したときと同じ（変わっていたら、検証が終わるまで変えてはいけない）。"""
    st, p, risk, rules, costs = confirm.frozen(PRE)
    assert st.key == PRE["rule"]["strategy_key"] == "ga_71600cc843"
    assert (risk.sl_atr, risk.tp_atr, rules.max_bars, rules.opposite_exit) == (1.5, 3.0, 48, True)
    assert costs.round_trip == pytest.approx(0.7)
    assert confirm.fingerprint(PRE) == PRE["rule"]["fingerprint"]


def test_fingerprint_detects_a_change():
    changed = json.loads(json.dumps(PRE))
    changed["rule"]["exits"]["sl_atr"] = 2.0
    assert confirm.fingerprint(changed) != PRE["rule"]["fingerprint"]


def test_trial_is_the_same_setting_as_the_exhaustive_search_but_counted_separately(tmp_path):
    from fxtool import data
    from fxtool.trials import strategy_config
    st, p, risk, rules, costs = confirm.frozen(PRE)
    exh = strategy_config(ga.key_of(confirm.genome(PRE)), p, risk, rules, spec={"ga_genes": ga.genes_of(confirm.genome(PRE))})
    row = confirm.trial_row(PRE)
    assert trials.config_hash("strategy", row["config"]) == trials.config_hash("strategy", exh)
    con = data.connect(tmp_path / "c.db")
    a, b = pd.Timestamp("2023-12-19"), pd.Timestamp("2026-03-16 07:00")
    trials.record(con, [{**row, "purpose": "ga_exhaustive"}], pair="USD/JPY", tf="1h", origin="x", now=b, data_from=a, data_to=b)
    n = trials.count(con, "USD/JPY", "1h")
    t0, t1 = pd.Timestamp(PRE["test_period"]["from"]), pd.Timestamp(PRE["test_period"]["to"])
    assert trials.count_with(con, "USD/JPY", "1h", [row], t0, t1, source="dukascopy") == n + 1
    # 同じ期間でもデータ元が違えば別に数える
    assert trials.count_with(con, "USD/JPY", "1h", [row], a, b, source="dukascopy") == n + 1
    assert trials.count_with(con, "USD/JPY", "1h", [row], a, b) == n


def synthetic(n_years=3, trend=0.0, seed=0):
    idx = pd.date_range("2012-10-01", periods=24 * 365 * n_years, freq="h")
    idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(trend, 0.05, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close}, index=idx)


def test_run_gives_two_verdicts_and_uses_only_the_test_period():
    pre = json.loads(json.dumps(PRE))
    pre["test_period"] = {"from": "2013-01-01 00:00:00", "to": "2014-12-31 23:00:00"}
    df = synthetic()
    c = confirm.run(df, pre, 1154, runs=20, rw_paths=5)
    assert c.verdict_single[0] in ("合格", "不合格") and c.verdict_cumulative[0] in ("合格", "不合格")
    assert len(c.verdict_single[1]) == 7
    assert c.dsr_cumulative <= c.psr_single + 1e-9            # 累計の試行数で割り引く方が厳しい
    assert set(c.yearly["year"]) <= {2013, 2014, 2015}          # 確認の期間の外では売買しない（決済が年をまたぐ分だけ）
    # 優位性のないデータ（ランダムウォーク）では合格しない
    assert c.verdict_cumulative[0] == "不合格"
    # 準備の足が足りなければ止まる
    with pytest.raises(ValueError):
        confirm.run(df[df.index >= "2012-12-31"], pre, 1154, runs=5, rw_paths=2)


def test_consistency_check():
    a = synthetic(1)
    b = a.copy()
    b["close"] += 0.005                                          # 0.5 pips のずれ
    r = confirm.compare_sources(a, b, 0.01)
    assert r["match_ratio"] == 1.0 and r["median_close_diff_pips"] == pytest.approx(0.5)
    lim = PRE["data"]["consistency_limits"]["dukascopy_vs_yahoo"]
    assert confirm.consistency_ok(r, lim)[0]
    b["close"] += 0.05                                           # 5.5 pips のずれ → 止める
    assert not confirm.consistency_ok(confirm.compare_sources(a, b, 0.01), lim)[0]
    assert not confirm.consistency_ok(confirm.compare_sources(a, b.iloc[::2], 0.01), lim)[0]
