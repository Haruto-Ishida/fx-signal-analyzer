"""ペーパートレード（仮想のお金での口座全体のシミュレーション。要件定義書 8章の3・ユーザーの決定 4）。

本番の記録（signal_log）のうち、このツールが「出した」シグナルを、仮想の口座（初期残高は settings.toml の [paper]）で
すべて取引したとみなして、残高・損益を計算する。実際の注文は一切しない。

口座が取引するシグナル
- 戦略のシグナル（機械学習の予測そのもの ml_ は除く）で、次の足の始値でエントリーした記録（entry_basis = next_open）
- 停止中（キルスイッチ・経済指標・最大損失）として記録だけしたシグナルは取引しない
- 機械学習（メタラベリング）が「見送り」と判定したシグナルは取引しない
- 1回の損失の目安は固定比率（settings.toml の fixed_risk_pct）。ケリー基準は「有望」と判定された戦略にしか
  使わない決まりで、いまはどの戦略も「有望」ではないため。実効レバレッジの上限も守る（超える分は見送り）。
シグナルごとの記録（signal_log）は、戦略の評価用にそのまま残る。
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import account, tracking
from .config import PAIRS, TIMEFRAMES, pip_size
from .settings import Settings

ELIGIBLE_SQL = """
    SELECT pair, tf, source, signal_time, direction, entry, sl, tp, status, exit_time, pips, meta_decision
    FROM signal_log
    WHERE source NOT LIKE 'ml\\_%' ESCAPE '\\' AND suppressed = 0 AND entry_basis = 'next_open'
      AND (meta_decision IS NULL OR meta_decision != '見送り')
"""


def eligible(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(ELIGIBLE_SQL, con)
    for c in ("signal_time", "exit_time"):
        df[c] = pd.to_datetime(df[c])
    return df


def trades(con: sqlite3.Connection) -> pd.DataFrame:
    """決済済みの、口座が取引したシグナル（account.simulate に渡す形）。"""
    df = eligible(con)
    df = df[df["status"].isin(tracking.DONE) & df["entry"].notna() & df["pips"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "dir", "entry", "sl", "pips", "ticker", "pip",
                                     "pair", "tf", "source"])
    bar = df["tf"].map(lambda k: TIMEFRAMES[k].bar)
    return pd.DataFrame({
        "entry_time": df["signal_time"] + bar,   # 次の足の始値
        "exit_time": df["exit_time"], "dir": df["direction"], "entry": df["entry"], "sl": df["sl"],
        "pips": df["pips"], "ticker": df["pair"].map(PAIRS), "pip": df["pair"].map(lambda p: pip_size(PAIRS[p])),
        "pair": df["pair"], "tf": df["tf"], "source": df["source"],
    }).reset_index(drop=True)


def build(con: sqlite3.Connection, s: Settings, usdjpy: float | None) -> account.AccountResult:
    t = trades(con)
    if usdjpy is None:  # 円以外が決済通貨のペアは換算できないので、円のペアだけで計算する
        t = t[t["ticker"].str.replace("=X", "").str[3:6] == "JPY"]
    return account.simulate(t, "USDJPY=X", 0.01, s.sizing.fixed_risk_pct, s.paper.initial_balance,
                            s.paper.monthly_deposit, s.sizing.max_leverage, usdjpy)


def signals_today(con: sqlite3.Connection, now: pd.Timestamp) -> int:
    """日本時間の今日に出した（停止中でない）新しいシグナルの数（口座全体）。"""
    df = eligible(con)
    if df.empty:
        return 0
    today = (now + tracking.JST).normalize()
    return int(((df["signal_time"] + tracking.JST) >= today).sum())
