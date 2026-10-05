"""取引コストを時間帯によって広めに見積もる（要件定義書 5-2）。

次の時間帯は、スプレッドに settings.toml の倍率を掛ける（スリッページはそのまま）：
- 重要な経済指標の前後（米雇用統計・FOMC・日銀会合。events.py と同じ決まった時刻で判断）
- 週明け（週末の空きのあと最初の数時間）
- ロールオーバーの時間帯（ニューヨーク市場が閉まる 17:00（米東部時間）からの1時間。夏時間は自動で計算）

注意：FOMC・日銀会合は日程ファイル（central_bank_schedule.csv）に登録された年（2026年〜）だけが対象で、
それより前の会合はわからないため、過去の検証では米雇用統計だけが「重要な指標」として反映される。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import events
from .settings import CostSettings, EventFilter

COST_MODEL_VERSION = "v2-time-varying-spread"  # 変えたら、保存済みの戦略の判定を計算し直す
WEEKEND_OPEN_HOURS = 3


def spread_multipliers(index: pd.DatetimeIndex, tf_key: str, cs: CostSettings | None = None,
                       ef: EventFilter | None = None, schedule: pd.DataFrame | None = None) -> np.ndarray:
    """足ごとのスプレッドの倍率（1 = 通常）。index は足の開始時刻（UTC）。"""
    cs, ef = cs or CostSettings(), ef or EventFilter()
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    m = np.ones(n)
    if n == 0:
        return m
    bar = pd.Timedelta(hours=1) if tf_key == "1h" else pd.Timedelta(days=1)
    starts = idx.to_numpy()
    ends = (idx + bar).to_numpy()

    # 重要な経済指標の前後
    sched = events.load_schedule() if schedule is None else schedule
    evs = [e for e in events.recurring(idx[0], idx[-1] + bar) if e.importance == "high"]
    evs += [e for e in events.meetings(sched) if idx[0] <= e.time < idx[-1] + bar]
    for e in evs:
        if e.time_fixed:
            lo, hi = e.time - pd.Timedelta(minutes=ef.minutes_before), e.time + pd.Timedelta(minutes=ef.minutes_after)
        else:
            w = pd.Timedelta(minutes=ef.unfixed_window_minutes)
            lo, hi = e.time - w, e.time + w
        hit = (starts < np.datetime64(hi)) & (ends > np.datetime64(lo))  # 足の時間と、指標の前後の時間が重なる
        m[hit] = np.maximum(m[hit], cs.event_spread_multiplier)

    # 週明け：前の足から24時間以上空いた足から数時間（最初の足は前の足がわからないので週明けとみなさない）
    gaps = np.r_[0.0, np.diff(starts).astype("timedelta64[s]").astype(float) / 3600]
    if tf_key == "1h":
        reopen = np.flatnonzero(gaps >= 24)
        for i in reopen:
            j = np.searchsorted(starts, starts[i] + np.timedelta64(WEEKEND_OPEN_HOURS, "h"))
            m[i:j] = np.maximum(m[i:j], cs.weekend_open_spread_multiplier)
        # ロールオーバー：17:00（米東部時間）からの1時間に始まる足
        et_hour = np.array([(t - pd.Timedelta(hours=4 if events.us_dst(t) else 5)).hour for t in idx])
        roll = et_hour == 17
        m[roll] = np.maximum(m[roll], cs.rollover_spread_multiplier)
    else:
        mondays = idx.dayofweek == 0
        m[mondays] = np.maximum(m[mondays], cs.weekend_open_spread_multiplier)
    return m
