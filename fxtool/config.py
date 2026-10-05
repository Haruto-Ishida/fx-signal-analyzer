"""通貨ペア・時間足・各種パラメータの設定。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fx.db"

PAIRS = {
    "USD/JPY": "USDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
}

PAIR_NAMES = {
    "USD/JPY": "ドル円", "EUR/JPY": "ユーロ円", "GBP/JPY": "ポンド円", "AUD/JPY": "豪ドル円",
    "EUR/USD": "ユーロドル", "GBP/USD": "ポンドドル", "AUD/USD": "豪ドル米ドル",
}


# バックテストの取引コストの初期値（pips）。国内FX業者の一般的な水準を参考にした目安
DEFAULT_SPREAD = {
    "USD/JPY": 0.3, "EUR/JPY": 0.5, "GBP/JPY": 1.0, "AUD/JPY": 0.6,
    "EUR/USD": 0.4, "GBP/USD": 0.9, "AUD/USD": 0.6,
}
DEFAULT_SLIPPAGE = 0.2  # 片道


@dataclass(frozen=True)
class Timeframe:
    key: str
    label: str
    interval: str              # yfinance に渡す足の種類
    bar: pd.Timedelta          # 1本の足の長さ
    periods: tuple[str, ...]   # 差分取得で使う期間（短い順）。最後が初回取得に使う最長期間
    stale_after: pd.Timedelta  # 最終取得からこの時間が経ったら自動で取り直す


TIMEFRAMES = {
    "1h": Timeframe("1h", "1時間足", "1h", pd.Timedelta(hours=1),
                    ("5d", "1mo", "3mo", "6mo", "1y", "730d"), pd.Timedelta(minutes=10)),
    "1d": Timeframe("1d", "日足", "1d", pd.Timedelta(days=1),
                    ("1mo", "6mo", "1y", "5y", "max"), pd.Timedelta(hours=1)),
}

PERIOD_DAYS = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825, "730d": 730}


@dataclass(frozen=True)
class TrackingSettings:
    log_max_bars: int     # 記録したシグナルを最大何本で打ち切って評価するか
    recent_bars: int      # 重み・警告に使う「直近」の本数
    retrain_every: int    # 何本ごとに機械学習モデルを再学習するか
    stale_bars: int       # 平日にこの本数以上データが増えなければ警告


TRACKING = {
    "1h": TrackingSettings(log_max_bars=48, recent_bars=720, retrain_every=24, stale_bars=6),
    "1d": TrackingSettings(log_max_bars=20, recent_bars=250, retrain_every=5, stale_bars=4),
}

# 定期実行で処理する通貨ペア
JOB_PAIRS = ("USD/JPY",)

# 記録・定期再学習する機械学習の設定（画面の機械学習タブの初期値と同じ）
ML_HORIZON = 1
ML_THRESHOLD = 0.55

MODELS_DIR = ROOT / "data" / "models"


@dataclass(frozen=True)
class IndicatorParams:
    sma_short: int = 20
    sma_long: int = 75
    rsi_period: int = 14
    rsi_low: float = 30
    rsi_high: float = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    donchian_period: int = 20


@dataclass(frozen=True)
class RiskParams:
    sl_atr: float = 1.5      # 損切り = エントリー ∓ ATR × この倍率
    tp_atr: float = 3.0      # 利確   = エントリー ± ATR × この倍率
    valid_bars: int = 3      # シグナルを「有効」とみなす本数


def pip_size(ticker: str) -> float:
    return 0.01 if "JPY" in ticker.upper() else 0.0001


def price_digits(ticker: str) -> int:
    return 3 if "JPY" in ticker.upper() else 5
