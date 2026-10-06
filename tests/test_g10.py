"""G10 の仮説の検証の仕組み（合成データだけ。実際のデータは使わない）。"""
import json

import numpy as np
import pandas as pd
import pytest

from fxtool import g10
from fxtool.config import ROOT

PRE = json.loads((ROOT / "preregistration" / "g10_hypotheses.json").read_text(encoding="utf-8"))
CM = g10.CostModel(PRE["costs"]["one_way"], PRE["costs"]["swap_haircut"]["judge"])


@pytest.fixture(scope="module")
def panel():
    return g10.build_panel(*g10.synthetic_inputs())


def test_fingerprint_is_registered():
    assert g10.fingerprint(CM) == PRE["fingerprint"]


def test_preregistration_matches_code():
    assert PRE["universe"]["currencies"] == g10.ALL
    assert set(PRE["costs"]["one_way"]) == set(g10.CURRENCIES)
    assert PRE["costs"]["one_way"]["SEK"] > PRE["costs"]["one_way"]["EUR"]      # SEK・NOK は多めに見積もる
    assert (g10.RATE_LAG, g10.CPI_LAG_MONTHLY, g10.CPI_LAG_QUARTERLY) == (1, 2, 3)
    assert g10.DEV_CUTOFF == pd.Timestamp("2020-10-14")


def test_cutoff_drops_later_data():
    fx, rates, cpi = g10.synthetic_inputs()
    later = pd.bdate_range("2020-10-15", periods=30)
    fx2 = pd.concat([fx, pd.DataFrame(999.0, index=later, columns=fx.columns)])
    pn = g10.build_panel(fx2, rates, cpi)
    assert pn.months[-1] == pd.Period("2020-10", "M") and (pn.price < 900).all().all()


def test_no_lookahead(panel):
    """ある月より後のデータを書き換えても、その月までの重みは変わらない（5つすべて）。"""
    fx, rates, cpi = g10.synthetic_inputs()
    t = pd.Timestamp("2016-06-30")
    fx2 = fx.copy()
    fx2[fx2.index > t] *= 1.3
    rates2 = {k: v.where(v.index <= t, v + 5) for k, v in rates.items()}
    cpi2 = {k: v.where(v.index <= t, v * 2) for k, v in cpi.items()}
    pn2 = g10.build_panel(fx2, rates2, cpi2)
    cut = pd.Period("2016-06", "M")
    for h, f in g10.WEIGHT_FUNCS.items():
        a, b = f(panel), f(pn2)
        pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut], obj=h)


def test_quarterly_cpi_is_used_after_three_months():
    q = pd.Series([100.0, 101.0], index=pd.to_datetime(["2012-01-01", "2012-04-01"]))   # 1〜3月・4〜6月
    m = pd.period_range("2012-03", "2012-10", freq="M")
    s = g10.quarterly_available(q, m)
    assert s[pd.Period("2012-05", "M")] != s[pd.Period("2012-05", "M")]               # まだ使えない
    assert s[pd.Period("2012-06", "M")] == 100.0 and s[pd.Period("2012-08", "M")] == 100.0
    assert s[pd.Period("2012-09", "M")] == 101.0


def test_rank_weights_and_usd(panel):
    score = pd.DataFrame([[5, 4, 3, 2, 1, 0, -1, -2, -3, -4]], columns=g10.ALL, index=[pd.Period("2015-01", "M")])
    w = g10._rank_weights(score).iloc[0]
    assert w["EUR"] == w["JPY"] == pytest.approx(1 / 6)                               # USD が1位 → 買いは2通貨分
    assert list(w[["CAD", "SEK", "NOK"]]) == pytest.approx([-1 / 6] * 3)
    assert w.abs().sum() == pytest.approx(5 / 6)


def test_monthly_returns_parts(panel):
    w = pd.DataFrame(0.0, index=panel.months, columns=g10.CURRENCIES)
    w["EUR"] = 1.0
    r = g10.monthly_returns(w, panel, CM)
    m = pd.Period("2015-03", "M")
    expect_spot = panel.spot_next.loc[m - 1, "EUR"]
    diff = (panel.rate_next.loc[m - 1, "EUR"] - panel.rate_next.loc[m - 1, "USD"]) / 100
    assert r.loc[m, "spot"] == pytest.approx(expect_spot)
    assert r.loc[m, "carry"] == pytest.approx(diff / 12 - 0.005 / 12)                 # 割り引き年0.5%
    assert r.loc[m, "cost"] == 0.0                                                     # 入れ替えなし
    assert r.iloc[0]["cost"] == pytest.approx(-0.0002)                                 # 最初に入るとき
    w2 = w.copy()
    w2["EUR"] = 0.0
    w2["SEK"] = 1.0
    assert g10.monthly_returns(w2, panel, CM).iloc[0]["cost"] == pytest.approx(-0.0015)   # SEK は高い


def test_run_all_gives_verdicts_on_random_data(panel):
    res = g10.run_all(panel, CM, PRE["pass_criteria"], n_trials_tool=1300, random_runs=50)
    assert set(res) == set(g10.HYPOTHESES)
    for h in ("H1", "H2", "H3", "H4"):
        r = res[h]
        assert r.returns.index.min() >= g10.DEV_FIRST_RETURN and r.returns.index.max() == g10.DEV_LAST_RETURN
        assert len(r.checks) == 7 and r.verdict in ("合格", "不合格")
        assert r.dsr_tool <= r.dsr_plan + 1e-9 <= r.psr_single + 2e-9 or np.isnan(r.psr_single)
    assert len(res["H5"].checks) == 2
    # でたらめな値動きでは、主な判定で合格しない
    assert all(res[h].verdict == "不合格" for h in ("H1", "H2", "H3", "H4"))


def test_haircut_reference_is_worse(panel):
    w = g10.weights_h2(panel)
    a = g10.dev_window(g10.monthly_returns(w, panel, CM))["total"].sum()
    b = g10.dev_window(g10.monthly_returns(w, panel, g10.CostModel(CM.one_way, 0.010)))["total"].sum()
    assert b < a
