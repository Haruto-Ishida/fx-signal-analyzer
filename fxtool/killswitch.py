"""キルスイッチ（要件定義書 8章の1・ユーザーの決定 5）。

ペーパートレードの口座（paper.py、口座全体）の記録をもとに、次のどれかに達したら、その期間は新しいシグナルを止める。
数値は settings.toml の [kill_switch]。
- 1日の損失：日本時間のその日に決済した損益が、その日の始まりの残高の daily_loss_pct% 以上の損失 → その日の終わりまで
- 1週間の損失：日本時間の月曜からの損益が、月曜の始まりの残高の weekly_loss_pct% 以上の損失 → その週の終わりまで
- 連敗：max_consecutive_losses 回続けて負けた → 最後の負けから24時間（その後は連敗の数を0から数え直す）
- 取引の回数：日本時間のその日の、その通貨ペアの新しいシグナルが上限（settings.toml の [signal_caps]。
  過去のデータで「100日に1日くらいしか起きない回数」）に達した → その日の終わりまで、その通貨ペアだけ止める
解除は、期間が過ぎたら自動。画面から手動でも解除できる（解除した記録は kill_switch_releases に残る）。
以前の「R（1回の損失の目安）で数える最大損失（1日2R・1週間5R）」は、2026-10-07 のユーザーの決定で外した（% の基準にまとめた）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .settings import KillSwitch
from .tracking import JST, ts_str

SCHEMA = """
CREATE TABLE IF NOT EXISTS kill_switch_releases (
    key TEXT PRIMARY KEY, released_at TEXT NOT NULL, note TEXT
);
"""
EPS = 1e-9


@dataclass(frozen=True)
class Trip:
    kind: str            # daily_loss / weekly_loss / consecutive / frequency
    key: str             # 手動で解除するときの目印（期間ごとに1つ）
    reason: str
    until: pd.Timestamp  # 自動で解除される時刻（UTC）
    pair: str | None = None  # frequency のときだけ：止める通貨ペア（ほかは口座全体）


@dataclass
class KillState:
    trips: list[Trip] = field(default_factory=list)       # 作動中（解除されていない）
    released: list[Trip] = field(default_factory=list)    # 作動したが、手動で解除されたもの
    day_loss_pct: float = 0.0
    week_loss_pct: float = 0.0
    streak: int = 0
    signals_today: dict = field(default_factory=dict)     # 通貨ペア → 今日の新しいシグナルの回数

    @property
    def active(self) -> bool:
        return bool(self.trips)

    @property
    def reason(self) -> str:
        return " ／ ".join(t.reason for t in self.trips)

    def trips_for(self, pair: str) -> list[Trip]:
        """その通貨ペアの新しいシグナルを止めている停止（口座全体の停止＋その通貨ペアの回数の停止）。"""
        return [t for t in self.trips if t.pair is None or t.pair == pair]

    def reason_for(self, pair: str) -> str:
        return " ／ ".join(t.reason for t in self.trips_for(pair))


def _start_balance(daily: pd.DataFrame, initial: float, day_jst: pd.Timestamp) -> float:
    if daily is None or daily.empty:
        return float(initial)
    before = daily[daily.index < day_jst]
    return float(before["balance"].iat[-1]) if len(before) else float(initial)


def evaluate(trades: pd.DataFrame, daily: pd.DataFrame, initial: float, n_signals_today: dict[str, int],
             now: pd.Timestamp, ks: KillSwitch, released_keys: set[str] | frozenset[str] = frozenset(),
             caps: dict[str, int] | None = None) -> KillState:
    """trades は account.simulate の結果の trades（units, pnl_jpy, exit_time）。daily は日ごとの残高（日本時間の日付）。
    n_signals_today は通貨ペアごとの今日の回数、caps は通貨ペアごとの上限（ないペアは ks.max_signals_per_day）。"""
    caps = caps or {}
    st = KillState(signals_today=dict(n_signals_today))
    jst_now = now + JST
    today = jst_now.normalize()
    monday = today - pd.Timedelta(days=today.dayofweek)
    end_of_day = today + pd.Timedelta(days=1) - JST
    end_of_week = monday + pd.Timedelta(days=7) - JST
    done = trades[(trades["units"] > 0) & trades["pnl_jpy"].notna()] if len(trades) else trades
    trips: list[Trip] = []

    if len(done):
        exit_jst = pd.DatetimeIndex(done["exit_time"]) + JST
        pnl = done["pnl_jpy"].to_numpy(dtype=float)
        day_pnl = float(pnl[exit_jst >= today].sum())
        week_pnl = float(pnl[exit_jst >= monday].sum())
        st.day_loss_pct = -day_pnl / _start_balance(daily, initial, today) * 100
        st.week_loss_pct = -week_pnl / _start_balance(daily, initial, monday) * 100
        if st.day_loss_pct >= ks.daily_loss_pct - EPS:
            trips.append(Trip("daily_loss", f"daily_loss:{today:%Y-%m-%d}",
                              f"今日（日本時間）のペーパー口座の損失が残高の {st.day_loss_pct:.1f}% に達しました"
                              f"（上限 {ks.daily_loss_pct:g}%）。今日は新しいシグナルを出しません", end_of_day))
        if st.week_loss_pct >= ks.weekly_loss_pct - EPS:
            trips.append(Trip("weekly_loss", f"weekly_loss:{monday:%Y-%m-%d}",
                              f"今週（日本時間の月曜から）のペーパー口座の損失が残高の {st.week_loss_pct:.1f}% に達しました"
                              f"（上限 {ks.weekly_loss_pct:g}%）。今週は新しいシグナルを出しません", end_of_week))
        # 連敗：負けが続いたら24時間止め、その後は0から数え直す
        order = done.sort_values("exit_time", kind="stable")
        streak, until = 0, None
        for t, p in zip(order["exit_time"], order["pnl_jpy"]):
            if until is not None and t < until:
                continue  # 止まっている間の決済（止まる前に入っていた取引）は数えない
            streak = streak + 1 if p < 0 else 0
            if streak >= ks.max_consecutive_losses:
                until = pd.Timestamp(t) + pd.Timedelta(hours=24)
                last_trip = Trip("consecutive", f"consecutive:{ts_str(t)}",
                                 f"ペーパー口座で {streak} 回続けて負けました（上限 {ks.max_consecutive_losses} 回）。"
                                 f"{(until + JST):%m/%d %H:%M}（日本時間）まで新しいシグナルを出しません", until)
                streak = 0
        st.streak = streak
        if until is not None and now < until:
            trips.append(last_trip)

    for pair, n in sorted(n_signals_today.items()):
        cap = int(caps.get(pair, ks.max_signals_per_day))
        if n >= cap:
            trips.append(Trip("frequency", frequency_key(pair, today),
                              f"{pair} の今日（日本時間）の新しいシグナルが {n} 回に達しました（上限 {cap} 回＝過去のデータで"
                              f"100日に1日くらいしか起きない回数）。いつもより多すぎるため、{pair} は今日これ以上出しません",
                              end_of_day, pair))
    st.trips = [t for t in trips if t.key not in released_keys]
    st.released = [t for t in trips if t.key in released_keys]
    return st


def frequency_key(pair: str, day_jst: pd.Timestamp) -> str:
    return f"frequency:{pair}:{day_jst:%Y-%m-%d}"


# ---------------- 記録 ----------------

def ensure(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


def released_keys(con: sqlite3.Connection) -> set[str]:
    ensure(con)
    return {k for (k,) in con.execute("SELECT key FROM kill_switch_releases")}


def release(con: sqlite3.Connection, key: str, now: pd.Timestamp, note: str = "画面から手動で解除") -> None:
    ensure(con)
    con.execute("INSERT OR REPLACE INTO kill_switch_releases (key, released_at, note) VALUES (?, ?, ?)",
                (key, ts_str(now), note))
    con.commit()


def check(con: sqlite3.Connection, now: pd.Timestamp, s, usdjpy: float | None):
    """ペーパー口座を計算して、キルスイッチの状態を返す。（口座の結果, 状態）"""
    from . import paper
    acc = paper.build(con, s, usdjpy)
    state = evaluate(acc.trades, acc.daily, s.paper.initial_balance, paper.signals_today(con, now), now,
                     s.kill_switch, released_keys(con), s.signal_caps)
    return acc, state
