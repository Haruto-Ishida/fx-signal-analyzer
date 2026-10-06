"""ユーザーが変えてよい数値（settings.toml）の読み込み。

書き方を間違えた場合や項目がない場合は初期値を使い、その内容を warnings に残す（画面に表示する）。
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .config import ROOT

SETTINGS_FILE = ROOT / "settings.toml"


@dataclass(frozen=True)
class Paper:
    initial_balance: float = 100_000
    monthly_deposit: float = 0


@dataclass(frozen=True)
class Sizing:
    kelly_fraction: float = 0.25
    fixed_risk_pct: float = 1.0
    max_risk_pct: float = 2.0
    max_leverage: float = 5.0


@dataclass(frozen=True)
class KillSwitch:
    max_consecutive_losses: int = 5
    max_signals_per_day: int = 10
    daily_loss_pct: float = 3.0
    weekly_loss_pct: float = 6.0


@dataclass(frozen=True)
class EventFilter:
    minutes_before: int = 60
    minutes_after: int = 60
    unfixed_window_minutes: int = 180


@dataclass(frozen=True)
class CostSettings:
    event_spread_multiplier: float = 3.0
    weekend_open_spread_multiplier: float = 3.0
    rollover_spread_multiplier: float = 2.0


@dataclass(frozen=True)
class TrialBudget:
    ga_full_search: int = 500
    ga_weekly_search: int = 150         # 定期的な再探索 1回あたり（前回の上位候補の周辺だけ）
    ga_weekly_enabled: bool = True      # 定期的な再探索をするか
    ga_interval_weeks: int = 3          # 再探索の間隔（週）。日本時間の土曜に行う


@dataclass(frozen=True)
class Papers:
    enabled: bool = False   # 週末の自動収集（外部への接続なので、ユーザーが有効にするまで止めておく）
    keywords: tuple[str, ...] = ("foreign exchange", "forex", "exchange rate forecasting", "currency trading",
                                 "financial time series machine learning")
    categories: tuple[str, ...] = ("q-fin.TR", "q-fin.ST", "q-fin.CP", "cs.LG", "stat.ML")
    max_results: int = 50


# 安全のための上限・下限（設定ファイルで危ない値にされても、ここで止める）
LIMITS = {
    ("sizing", "kelly_fraction"): (0.0, 0.5),
    ("sizing", "max_risk_pct"): (0.0, 2.0),
    ("sizing", "fixed_risk_pct"): (0.0, 2.0),
    ("sizing", "max_leverage"): (0.0, 25.0),
    ("paper", "initial_balance"): (1_000, 1e10),
    # ユーザーの決定：本格的な探索は1回500通りまで、週末の再探索は1回100通りまで
    ("trial_budget", "ga_full_search"): (0, 500),
    ("trial_budget", "ga_weekly_search"): (0, 150),
    ("trial_budget", "ga_interval_weeks"): (1, 52),
    ("kill_switch", "max_signals_per_day"): (1, 1000),
    ("papers", "max_results"): (1, 200),
}


@dataclass(frozen=True)
class Settings:
    paper: Paper = Paper()
    sizing: Sizing = Sizing()
    kill_switch: KillSwitch = KillSwitch()
    event_filter: EventFilter = EventFilter()
    costs: CostSettings = CostSettings()
    trial_budget: TrialBudget = TrialBudget()
    papers: Papers = Papers()
    signal_caps: dict = field(default_factory=dict)  # 通貨ペアごとの1日のシグナルの回数の上限（[signal_caps]）
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def signal_cap(self, pair: str) -> int:
        return int(self.signal_caps.get(pair, self.kill_switch.max_signals_per_day))


_SECTIONS = {"paper": Paper, "sizing": Sizing, "kill_switch": KillSwitch, "event_filter": EventFilter,
             "costs": CostSettings, "trial_budget": TrialBudget, "papers": Papers}


def load(path: Path | None = None) -> Settings:
    path = path or SETTINGS_FILE
    warnings: list[str] = []
    raw: dict = {}
    if path.exists():
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            warnings.append(f"settings.toml の書き方に誤りがあるため、すべて初期値を使っています（{e}）")
    else:
        warnings.append("settings.toml が見つからないため、初期値を使っています")

    parts = {}
    for name, cls in _SECTIONS.items():
        sec = raw.get(name, {})
        values = {}
        for f in fields(cls):
            default = f.default
            if f.name not in sec:
                values[f.name] = default
                continue
            v = sec[f.name]
            if isinstance(default, bool):
                # bool("false") は True になってしまうので、true / false 以外は初期値にする
                if not isinstance(v, bool):
                    warnings.append(f"[{name}] {f.name} の値「{v}」が true / false ではないため、初期値 {str(default).lower()} を使っています")
                    v = default
                values[f.name] = v
                continue
            if isinstance(default, tuple):
                if not (isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)):
                    warnings.append(f"[{name}] {f.name} は文字列のリスト（例：[\"a\", \"b\"]）で書いてください。初期値を使っています")
                    v = default
                values[f.name] = tuple(x.strip() for x in v)
                continue
            try:
                v = type(default)(v)
            except (TypeError, ValueError):
                warnings.append(f"[{name}] {f.name} の値「{v}」が数値ではないため、初期値 {default} を使っています")
                v = default
            lo_hi = LIMITS.get((name, f.name))
            if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
                clipped = min(max(v, lo_hi[0]), lo_hi[1])
                warnings.append(f"[{name}] {f.name} = {v} は安全のための範囲（{lo_hi[0]:g}〜{lo_hi[1]:g}）の外なので、{clipped:g} を使っています")
                v = type(default)(clipped)
            values[f.name] = v
        parts[name] = cls(**values)
    caps = {}
    for pair, v in (raw.get("signal_caps") or {}).items():
        if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 1000:
            caps[pair] = v
        else:
            warnings.append(f"[signal_caps] {pair} の値「{v}」は 1〜1000 の整数ではないため、使いません")
    return Settings(**parts, signal_caps=caps, warnings=tuple(warnings))
