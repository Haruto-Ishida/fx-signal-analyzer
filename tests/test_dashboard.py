"""画面のタブを、記録の状態をいろいろ変えて最後まで描画できるか確かめる（Streamlit の bare モードで実行）。"""
import logging

import pandas as pd
import pytest

from conftest import make_path_ohlc
from dashboard import signals_tab, tracking_tab
from dashboard.common import Context
from fxtool import data, indicators, tracking
from fxtool.config import TIMEFRAMES, IndicatorParams, RiskParams

logging.getLogger("streamlit").setLevel(logging.ERROR)
PAIR, TF = "AUD/JPY", "1h"


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "ui.db"
    real_connect = data.connect
    monkeypatch.setattr(data, "connect", lambda *a, **k: real_connect(path))
    con = real_connect(path)
    yield con
    con.close()


@pytest.fixture
def ctx():
    df = make_path_ohlc(600, seed=21)
    ind = indicators.add_all(df, IndicatorParams())
    return Context(PAIR, "AUDJPY=X", TF, TIMEFRAMES[TF], 0.01, 3, IndicatorParams(), RiskParams(), ind, len(ind) - 1)


def add_signal(con, t, source="macd_cross", status="進行中", pips=None, exit_time=None):
    con.execute(
        """INSERT INTO signal_log (pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars,
           prob, weight, logged_at, status, exit_time, exit_price, pips)
           VALUES (?, ?, ?, ?, 1, 150, 149.7, 150.6, 0.2, 48, NULL, 1.0, ?, ?, ?, ?, ?)""",
        (PAIR, TF, source, t, t, status, exit_time, None if pips is None else 150 + pips / 100, pips),
    )
    con.commit()


def add_prediction(con, t, y=None):
    con.execute(
        "INSERT INTO prediction_log VALUES (?, ?, 'gbm', 1, ?, 0.55, 'シグナルなし', 'v1', ?, ?)", (PAIR, TF, t, t, y)
    )
    con.execute(
        """INSERT OR IGNORE INTO model_registry VALUES ('v1', ?, ?, 'gbm', 1, ?, ?, ?, 100,
           0.53, 0.52, 0.69, 0.69, -10, 0.5, 0, 'data/models/v1.joblib')""", (PAIR, TF, t, t, t),
    )
    con.commit()


def test_tracking_tab_empty(db, ctx):
    tracking_tab.render(ctx)


def test_tracking_tab_only_open_signals(db, ctx):
    """決済前のシグナルしかない（損益の列がすべて空）状態。実際に画面でエラーになった再現テスト。"""
    add_signal(db, "2026-10-05 08:00:00")
    add_signal(db, "2026-10-05 08:00:00", source="breakout")
    sl = tracking.load_signal_log(db, PAIR, TF)
    assert sl["pips"].dtype == float and sl["exit_price"].dtype == float and sl["prob"].dtype == float
    tracking_tab.render(ctx)


def test_tracking_tab_mixed_signals_and_predictions(db, ctx):
    add_signal(db, "2026-10-05 01:00:00", status="利確", pips=58.9, exit_time="2026-10-05 04:00:00")
    add_signal(db, "2026-10-05 02:00:00", status="損切り", pips=-30.7, exit_time="2026-10-05 05:00:00")
    add_signal(db, "2026-10-05 06:00:00")
    add_prediction(db, "2026-10-05 07:00:00", y=1.0)
    add_prediction(db, "2026-10-05 08:00:00")  # 答え合わせ待ち
    assert tracking.load_predictions(db, PAIR, TF)["y"].dtype == float
    tracking_tab.render(ctx)


def test_tracking_tab_only_unlabeled_predictions(db, ctx):
    add_prediction(db, "2026-10-05 08:00:00")
    tracking_tab.render(ctx)


def test_signals_tab_with_and_without_tracking_data(db, ctx):
    signals_tab.render(ctx)
    ctx.weights = {"ma_cross": 1.2, "macd_cross": 0.8, "breakout": 1.0, "rsi_reversal": 0.6, "bb_reversion": 1.4}
    ctx.alerts = pd.DataFrame({"source": ["rsi_reversal"], "message": ["RSI反転：悪化"]})
    ctx.verdicts = pd.DataFrame({"source": ["macd_cross"], "verdict": ["偶然の範囲"]})
    ctx.ml_status = [tracking.MLStatus("gbm", "勾配ブースティング", 1, 0.55, "シグナルなし", False, ctx.ind.index[-2])]
    signals_tab.render(ctx)
