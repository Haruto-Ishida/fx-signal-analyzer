"""Step 3：資金管理（分数ケリー・上限・固定比率）と、口座全体のシミュレーション（積立・時間加重リターン）のテスト。"""
import math

import numpy as np
import pandas as pd
import pytest

from fxtool import account, money, sizing
from fxtool.settings import Sizing

S = Sizing()  # 4分の1ケリー、固定1%、上限2%、レバレッジ5倍


# ---------------- ケリー基準 ----------------

def test_kelly_full():
    assert money.kelly_full(0.5, 2.0) == pytest.approx(0.25)
    assert money.kelly_full(0.4, 1.6) == pytest.approx(0.025)
    assert money.kelly_full(0.3, 1.5) < 0
    assert math.isnan(money.kelly_full(math.nan, 2.0))


def test_decide_risk():
    few = money.decide_risk(80, 0.5, 2.0, S)
    assert (few.method, few.risk_pct) == ("fixed", 1.0) and "100回未満" in few.explanation
    capped = money.decide_risk(150, 0.5, 2.0, S)            # 4分の1でも 6.25% → 上限 2%
    assert (capped.method, capped.risk_pct) == ("kelly", 2.0) and "上限" in capped.explanation
    small = money.decide_risk(150, 0.4, 1.6, S)             # 2.5% × 1/4 = 0.625%
    assert small.risk_pct == pytest.approx(0.625)
    neg = money.decide_risk(150, 0.3, 1.5, S)
    assert (neg.method, neg.risk_pct) == ("none", 0.0)
    half = money.decide_risk(150, 0.4, 1.6, Sizing(kelly_fraction=0.9))  # 2分の1を超えて設定しても 1/2 まで
    assert half.risk_pct == pytest.approx(1.25)


def test_leverage_cap():
    # 残高10万円・損切りまで0.3円・1% → 3,333通貨だが、5倍（50万円分）までなので 150円なら 3,333 → 3,000通貨
    s = sizing.position_size(100_000, 1.0, 150.0, 149.7, "USDJPY=X", None, max_leverage=5)
    assert s.units == 3_000
    # 損切りが近いと、損失の上限より先にレバレッジの上限に当たる
    s2 = sizing.position_size(100_000, 2.0, 150.0, 149.95, "USDJPY=X", None, max_leverage=5)
    assert s2.units == 3_000 and "レバレッジ" in s2.note
    # すでに45万円分持っていれば、残りは5万円分（333通貨 → 最小単位未満で0）
    s3 = sizing.position_size(100_000, 1.0, 150.0, 149.7, "USDJPY=X", None, max_leverage=5, open_notional_jpy=450_000)
    assert s3.units == 0 and "レバレッジ" in s3.note


# ---------------- 口座全体のシミュレーション ----------------

def trades(rows):
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "dir", "entry", "sl", "pips"]).assign(
        entry_time=lambda d: pd.to_datetime(d["entry_time"]), exit_time=lambda d: pd.to_datetime(d["exit_time"]))


def test_sequential_trades_compound():
    t = trades([("2026-01-05 01:00", "2026-01-05 05:00", 1, 150.0, 149.7, 60.0),
                ("2026-01-06 01:00", "2026-01-06 05:00", 1, 150.0, 149.7, -30.0)])
    r = account.simulate(t, "USDJPY=X", 0.01, risk_pct=1.0, initial=100_000, max_leverage=None)
    # 1回目：1,000円 ÷ 0.3円 = 3,333 → 3,000通貨、+60pips → +1,800円
    # 2回目：残高 101,800円の 1% ÷ 0.3 = 3,393 → 3,000通貨、−30pips → −900円
    assert r.trades["units"].tolist() == [3000, 3000]
    assert r.trades["pnl_jpy"].tolist() == pytest.approx([1800, -900])
    assert r.metrics["final"] == pytest.approx(100_900) and r.metrics["profit"] == pytest.approx(900)
    assert r.metrics["twr"] == pytest.approx(0.009)


def test_deposits_are_separated_from_performance():
    t = trades([("2026-01-05 01:00", "2026-01-05 05:00", 1, 150.0, 149.7, 0.0),
                ("2026-03-02 01:00", "2026-03-02 05:00", 1, 150.0, 149.7, 0.0)])
    r = account.simulate(t, "USDJPY=X", 0.01, 1.0, 100_000, monthly_deposit=10_000)
    m = r.metrics
    assert m["deposits"] == 20_000 and m["final"] == pytest.approx(120_000)  # 2月・3月の入金
    assert m["profit"] == pytest.approx(0) and m["twr"] == pytest.approx(0)  # 入金は運用成績に入らない
    d = r.daily
    assert d["deposits_cum"].iat[-1] == 20_000 and d["pnl_cum"].abs().max() == pytest.approx(0)


def test_overlapping_positions_respect_leverage():
    t = trades([("2026-01-05 01:00", "2026-01-05 10:00", 1, 150.0, 149.7, 10.0),
                ("2026-01-05 02:00", "2026-01-05 09:00", 1, 150.0, 149.7, 10.0)])
    r = account.simulate(t, "USDJPY=X", 0.01, 1.0, 100_000, max_leverage=5)
    # 1つ目で 3,000通貨（45万円分）。2つ目は残り5万円分しかなく、最小単位に届かないので見送り
    assert r.trades["units"].tolist() == [3000, 0] and "見送り" in r.trades["note"].iat[1]
    assert r.metrics["n"] == 1 and r.metrics["n_skipped"] == 1


def test_bankruptcy_stops_trading():
    t = trades([("2026-01-05 01:00", "2026-01-05 05:00", 1, 150.0, 149.7, -5000.0),
                ("2026-01-06 01:00", "2026-01-06 05:00", 1, 150.0, 149.7, 10.0)])
    r = account.simulate(t, "USDJPY=X", 0.01, 1.0, 100_000, max_leverage=None)
    assert r.metrics["bankrupt"] and r.trades["units"].iat[1] == 0


def test_usd_quote_needs_usdjpy():
    t = trades([("2026-01-05 01:00", "2026-01-05 05:00", 1, 1.10, 1.095, 20.0)])
    with pytest.raises(ValueError):
        account.simulate(t, "EURUSD=X", 0.0001, 1.0, 100_000)
    r = account.simulate(t, "EURUSD=X", 0.0001, 1.0, 100_000, usdjpy=150.0)
    # 1,000円 ÷ (0.005ドル × 150円) = 1,333 → 1,000通貨、+20pips × 0.0001 × 150円 = 300円
    assert r.trades["units"].iat[0] == 1000 and r.metrics["profit"] == pytest.approx(300)
