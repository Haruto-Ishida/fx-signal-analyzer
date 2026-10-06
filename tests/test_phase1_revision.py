"""フェーズ1改修（研究ノート反映）のテスト：最終テスト期間・試行の記録・次の足の始値・時間切れ・数量の目安・データの確認。"""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlc, make_path_ohlc
from fxtool import backtest as bt
from fxtool import data, holdout, indicators, jobs, ml, sizing, strategies, tracking, trials
from fxtool.config import TIMEFRAMES, IndicatorParams, RiskParams

P, RISK = IndicatorParams(), RiskParams()
NOW = pd.Timestamp("2026-10-05 12:00")


# ---------------- 最終テスト期間 ----------------

def test_boundary_is_fixed_at_80_percent_and_never_moves(tmp_path):
    df = make_path_ohlc(1000)
    b = holdout.fix("USD/JPY", "1h", df, NOW, "メモ")
    assert b.start == df.index[800] and b.n_bars_at_fix == 1000
    saved = json.loads(holdout.HOLDOUT_FILE.read_text(encoding="utf-8"))
    assert saved["boundaries"]["USD/JPY|1h"]["start"] == df.index[800].strftime("%Y-%m-%d %H:%M:%S")
    # データが増えても、もう一度 fix しても境界は動かない
    bigger = make_path_ohlc(2000)
    assert holdout.fix("USD/JPY", "1h", bigger, NOW + pd.Timedelta(days=30), "別") == b
    assert holdout.get("USD/JPY", "1h") == b


def test_dev_part_and_guards():
    df = make_path_ohlc(1000)
    b = holdout.fix("EUR/USD", "1h", df, NOW, "")
    dev = holdout.dev_part(df, b)
    assert len(dev) == 800 and dev.index[-1] < b.start
    holdout.assert_dev_only(dev, b)
    with pytest.raises(holdout.FinalTestAccessError):
        holdout.assert_dev_only(df, b)
    ind = indicators.add_all(df, P)
    with pytest.raises(holdout.FinalTestAccessError):
        bt.evaluate_all(df, P, RISK, bt.ExitRules(), bt.Costs(), 0.01, runs=5, holdout_start=b.start)
    with pytest.raises(holdout.FinalTestAccessError):
        ml.evaluate_model(ind, "1h", "logistic", 1, 0.55, RISK, bt.Costs(), 0.01, runs=5, holdout_start=b.start)


def test_final_part_requires_confirmation_and_is_logged(tmp_path):
    con = data.connect(tmp_path / "f.db")
    df = make_path_ohlc(500)
    b = holdout.fix("USD/JPY", "1d", df, NOW, "")
    with pytest.raises(holdout.FinalTestAccessError):
        holdout.final_part(df, b, confirm="はい", con=con, reason="x", now=NOW)
    out = holdout.final_part(df, b, confirm=holdout.FINAL_TEST_CONFIRM, con=con, reason="最終テスト", now=NOW)
    assert len(out) == 100
    assert con.execute("SELECT reason FROM final_test_access").fetchall() == [("最終テスト",)]


def test_data_boundary_marks_usdjpy_as_touched(tmp_path):
    con = data.connect(tmp_path / "b.db")
    data.save(con, "USD/JPY", "1h", make_path_ohlc(600))
    data.save(con, "EUR/JPY", "1h", make_path_ohlc(600))
    now = pd.Timestamp("2021-01-01")
    assert "すでに見ている" in data.boundary(con, "USD/JPY", "1h", now).touched
    assert "可能性" in data.boundary(con, "EUR/JPY", "1h", now).touched
    assert data.load_dev(con, "USD/JPY", "1h", now).index[-1] < holdout.get("USD/JPY", "1h").start


# ---------------- 試行の記録 ----------------

def test_retroactive_trials_are_seeded_once(tmp_path):
    con = data.connect(tmp_path / "t.db")
    n_retro = trials.count(con, "USD/JPY", "1h", origin="retroactive")
    assert n_retro == 37 + 6  # ウォークフォワードの候補37 + 機械学習6
    assert trials.count(con, "USD/JPY", "mixed", origin="retroactive") == 35  # 最初の試作版
    data.connect(tmp_path / "t.db")  # もう一度つないでも増えない
    assert trials.count(con, "USD/JPY", "1h", origin="retroactive") == n_retro


def test_trial_count_is_by_distinct_config(tmp_path):
    con = data.connect(tmp_path / "t2.db")
    cfg = trials.strategy_config("ma_cross", P, RISK, bt.ExitRules(max_bars=48))
    row = {"family": "strategy", "name": "ma_cross", "purpose": "full_period", "config": cfg}
    trials.record(con, [row, row], pair="EUR/JPY", tf="1h", origin="dashboard", now=NOW)
    assert trials.count(con, "EUR/JPY", "1h") == 1
    other = dict(row, config=trials.strategy_config("ma_cross", P, RiskParams(sl_atr=2.0), bt.ExitRules(max_bars=48)))
    trials.record(con, [other], pair="EUR/JPY", tf="1h", origin="dashboard", now=NOW)
    assert trials.count(con, "EUR/JPY", "1h") == 2  # 損切りの幅を変えたら別の試行


def test_trial_count_separates_data_periods(tmp_path):
    """2026-10-07：データの期間が変わったら、同じ設定でも別の試行として数える（年1回の見直しのため）。"""
    con = data.connect(tmp_path / "t3.db")
    cfg = trials.strategy_config("ma_cross", P, RISK, bt.ExitRules(max_bars=48))
    row = {"family": "strategy", "name": "ma_cross", "purpose": "full_period", "config": cfg}
    a, b, c = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
    trials.record(con, [row], pair="EUR/JPY", tf="1h", origin="job", now=NOW, data_from=a, data_to=b)
    trials.record(con, [row], pair="EUR/JPY", tf="1h", origin="dashboard", now=NOW, data_from=a, data_to=b)
    assert trials.count(con, "EUR/JPY", "1h") == 1                      # 同じ設定・同じ期間 → 1つ
    trials.record(con, [row], pair="EUR/JPY", tf="1h", origin="job", now=NOW, data_from=a, data_to=c)
    assert trials.count(con, "EUR/JPY", "1h") == 2                      # 期間が変わった → 別の試行
    # これから試す予定の数え方も同じ
    assert trials.count_with(con, "EUR/JPY", "1h", [row], a, b) == 2
    assert trials.count_with(con, "EUR/JPY", "1h", [row], a, pd.Timestamp("2027-01-01")) == 3
    # 以前の記録（trial_key がない）は、つないだときに付け直す
    con.execute("UPDATE trials SET trial_key=NULL")
    con.commit()
    trials.migrate(con)
    assert trials.count(con, "EUR/JPY", "1h") == 2


def test_evaluate_all_reports_every_candidate(random_walk):
    rep = bt.evaluate_all(random_walk, P, RISK, bt.ExitRules(max_bars=48), bt.Costs(), 0.01, runs=5, k=3)
    by_purpose = pd.Series([r["purpose"] for r in rep.trials]).value_counts()
    assert by_purpose["full_period"] == 5
    assert by_purpose["wf_candidate"] == sum(len(g) for g in bt.PARAM_GRIDS.values())
    sel = [r for r in rep.trials if r["purpose"] == "wf_candidate" and r["metrics"]["selected_folds"]]
    assert sum(len(r["metrics"]["selected_folds"]) for r in sel) == 5 * 3  # 各戦略・各区間で1つずつ選ばれる


# ---------------- 次の足の始値・時間切れ ----------------

def test_screen_entry_is_next_open_or_estimate(random_walk):
    ind = indicators.add_all(random_walk, P)
    st = strategies.STRATEGY_BY_KEY["macd_cross"]
    i = int(np.flatnonzero(strategies.signals(ind, st, P).to_numpy())[-3])
    v = strategies.evaluate(ind.iloc[: i + 2], i + 1, st, P, RISK)  # i+1 本目が形成中
    assert v.entry_fixed and v.entry == ind["open"].iat[i + 1]
    sl, tp = strategies.levels(v.entry, v.direction, ind["atr"].iat[i], RISK)
    assert (v.sl, v.tp) == (sl, tp)
    v2 = strategies.evaluate(ind.iloc[: i + 1], i + 1, st, P, RISK)  # 次の足がまだない
    assert not v2.entry_fixed and v2.entry == ind["close"].iat[i]


def test_screen_time_limit():
    flat = [[100, 100.05, 99.95, 100]] * 30
    df = pd.DataFrame(flat, columns=["open", "high", "low", "close"], dtype=float,
                      index=pd.date_range("2025-01-01", periods=30, freq="h"))
    o = strategies.resolve_trade(df, 0, 1, sl=99, tp=102, max_bars=10)
    assert (o.status, o.bars_held) == ("期限", 10)


def test_logged_signal_waits_for_next_open_then_fills(tmp_path, monkeypatch):
    con = data.connect(tmp_path / "l.db")
    df = make_path_ohlc(400)
    ind = indicators.add_all(df, P)
    st = strategies.STRATEGY_BY_KEY["macd_cross"]
    i = int(np.flatnonzero(strategies.signals(ind, st, P).to_numpy())[-1])
    jobs.log_strategy_signals(con, "USD/JPY", "1h", ind.iloc[: i + 1], {}, NOW)
    row = tracking.load_signal_log(con, "USD/JPY", "1h").iloc[-1]
    assert row["status"] == "待機" and np.isnan(row["entry"]) and row["entry_basis"] == "next_open"
    assert jobs.fill_entries(con, "USD/JPY", "1h", df.iloc[: i + 1]) == 0  # 次の足がまだない
    assert jobs.fill_entries(con, "USD/JPY", "1h", df.iloc[: i + 2]) >= 1
    row = tracking.load_signal_log(con, "USD/JPY", "1h").set_index("source").loc["macd_cross"]
    assert row["status"] == "進行中" and row["entry"] == df["open"].iat[i + 1]
    assert row["sl"] == pytest.approx(row["entry"] - row["direction"] * RISK.sl_atr * row["atr"])


def test_old_signal_log_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE signal_log (id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT NOT NULL, tf TEXT NOT NULL,
            source TEXT NOT NULL, signal_time TEXT NOT NULL, direction INTEGER NOT NULL, entry REAL NOT NULL,
            sl REAL NOT NULL, tp REAL NOT NULL, atr REAL NOT NULL, max_bars INTEGER NOT NULL, prob REAL, weight REAL,
            logged_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT '進行中', exit_time TEXT, exit_price REAL, pips REAL,
            UNIQUE (pair, tf, source, signal_time));
        INSERT INTO signal_log (pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars, logged_at)
            VALUES ('AUD/JPY', '1h', 'macd_cross', '2026-10-05 08:00:00', 1, 100, 99.7, 100.6, 0.2, 48, '2026-10-05 09:00:00');
    """)
    old.commit()
    old.close()
    con = data.connect(path)
    sl = tracking.load_signal_log(con, "AUD/JPY", "1h")
    assert len(sl) == 1 and sl["entry_basis"].iat[0] == "signal_close" and sl["entry"].iat[0] == 100
    assert sl["status"].iat[0] == "進行中"


# ---------------- 数量の目安 ----------------

def test_position_size_jpy_quote():
    s = sizing.position_size(1_000_000, 1.0, 150.00, 149.70, "USDJPY=X", None)  # 損切りまで 0.30円
    assert s.risk_jpy == 10_000 and s.units == 33_000  # 10,000 / 0.30 = 33,333 → 1,000単位で切り捨て
    assert s.loss_at_stop_jpy == pytest.approx(9_900)


def test_position_size_usd_quote_and_limits():
    s = sizing.position_size(1_000_000, 1.0, 1.1000, 1.0950, "EURUSD=X", 150.0)  # 0.005ドル × 150円
    assert s.units == 13_000
    assert sizing.position_size(1_000_000, 1.0, 1.1, 1.095, "EURUSD=X", None) is None  # 円に換算できない
    tiny = sizing.position_size(10_000, 0.1, 150.0, 149.0, "USDJPY=X", None)
    assert tiny.units == 0 and "最小単位" in tiny.note


# ---------------- データの確認 ----------------

def test_quality_report():
    idx = pd.DatetimeIndex(["2026-10-02 19:00", "2026-10-02 20:00", "2026-10-04 23:00", "2026-10-05 00:00",
                            "2026-10-05 03:00", "2026-10-03 10:00"]).sort_values()
    df = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}, index=idx)
    df.iloc[0] = 1.0  # 値動きゼロ
    q = data.quality(df, TIMEFRAMES["1h"])
    assert q["saturday"] == 1 and q["flat"] == 1 and q["sunday_hours"] == [23] and q["duplicates"] == 0
    # 土曜の足があるので「金曜→日曜/月曜」の週末にならず、金曜→土曜・土曜→日曜・月曜0時→3時 の3回
    assert q["gaps"] == 3
