"""各タブで共通に使う、画面の状態と表示用の関数。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from fxtool import holdout
from fxtool.config import (
    DEFAULT_BALANCE_JPY, DEFAULT_RISK_PCT, VERTICAL_BARRIER, IndicatorParams, RiskParams, Timeframe,
)

JST = pd.Timedelta(hours=9)
BUY_COLOR, SELL_COLOR, NEUTRAL_COLOR = "#26a69a", "#ef5350", "#9e9e9e"
STRATEGY_COLORS = ["#2962ff", "#ff9800", "#8e24aa", "#00897b", "#e53935"]


@dataclass
class Context:
    pair: str
    ticker: str
    tf_key: str
    tf: Timeframe
    pip: float
    digits: int
    params: IndicatorParams
    risk: RiskParams
    ind: pd.DataFrame        # 未確定の足も含む、指標付きデータ
    n_conf: int              # 確定済みの足の本数
    weights: dict[str, float] = field(default_factory=dict)  # 戦略の重み（フェーズ4）
    ml_status: list = field(default_factory=list)            # 機械学習の最新の予測（フェーズ4）
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)    # 有効な警告
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)  # 戦略の検証結果（フェーズ2の判定）
    boundary: holdout.Boundary | None = None                 # 最終テスト期間の境界
    balance_jpy: float = DEFAULT_BALANCE_JPY                 # ポジションサイズの目安用
    risk_pct: float = DEFAULT_RISK_PCT
    usdjpy: float | None = None                              # 円以外の通貨の損失を円に直すため

    @property
    def confirmed(self) -> pd.DataFrame:
        """確定足すべて（本番の表示用）。"""
        return self.ind.iloc[: self.n_conf]

    @property
    def dev(self) -> pd.DataFrame:
        """開発用データ：確定足のうち最終テスト期間より前。検証（バックテスト・機械学習）は必ずこれを使う。"""
        if self.boundary is None:
            raise holdout.FinalTestAccessError("最終テスト期間の境界が決まっていません")
        return holdout.dev_part(self.confirmed, self.boundary)

    @property
    def max_bars(self) -> int:
        """トリプルバリアの時間切れの本数。"""
        return VERTICAL_BARRIER[self.tf_key]

    def label(self, ts: pd.Timestamp) -> str:
        """UTC の時刻を画面表示用（日本時間）の文字列にする。日足は日付のみ。"""
        if self.tf_key == "1d":
            return ts.strftime("%Y/%m/%d")
        return (ts + JST).strftime("%Y/%m/%d %H:%M")

    def labels(self, index) -> list[str]:
        return [self.label(t) for t in index]

    def to_display_time(self, index: pd.Index) -> pd.Index:
        """チャートの時間軸用（1時間足は日本時間に直す）。"""
        return index if self.tf_key == "1d" else index + JST

    def fmt_price(self, x: float | None) -> str:
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{self.digits}f}"


def fmt_pf(x: float) -> str:
    if math.isnan(x):
        return "—"
    return "∞" if math.isinf(x) else f"{x:.2f}"


def fmt_pct(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:.0%}"


def fmt_pips(x: float, sign: bool = True) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:+,.1f}" if sign else f"{x:,.1f}"


def color_decision(val: str) -> str:
    if val == "買い":
        return f"background-color: {BUY_COLOR}; color: white; font-weight: bold"
    if val == "売り":
        return f"background-color: {SELL_COLOR}; color: white; font-weight: bold"
    return ""
