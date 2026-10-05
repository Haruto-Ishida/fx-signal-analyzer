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
    ga_weekly_search: int = 100


# 安全のための上限・下限（設定ファイルで危ない値にされても、ここで止める）
LIMITS = {
    ("sizing", "kelly_fraction"): (0.0, 0.5),
    ("sizing", "max_risk_pct"): (0.0, 2.0),
    ("sizing", "fixed_risk_pct"): (0.0, 2.0),
    ("sizing", "max_leverage"): (0.0, 25.0),
    ("paper", "initial_balance"): (1_000, 1e10),
}


@dataclass(frozen=True)
class Settings:
    paper: Paper = Paper()
    sizing: Sizing = Sizing()
    kill_switch: KillSwitch = KillSwitch()
    event_filter: EventFilter = EventFilter()
    costs: CostSettings = CostSettings()
    trial_budget: TrialBudget = TrialBudget()
    warnings: tuple[str, ...] = field(default=(), compare=False)


_SECTIONS = {"paper": Paper, "sizing": Sizing, "kill_switch": KillSwitch, "event_filter": EventFilter,
             "costs": CostSettings, "trial_budget": TrialBudget}


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
    return Settings(**parts, warnings=tuple(warnings))
