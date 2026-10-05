"""試行の記録。研究ノート 2-3「試した戦略・パラメータの組み合わせの数を必ず記録する」。

- バックテスト・ウォークフォワードの候補・機械学習の評価をするたびに、設定と結果を記録する。
- 「試した数」は設定の種類（config_hash）の数で数える。同じ設定を何度評価しても1つ。
  取引コストは戦略の選択ではないのでハッシュに含めないが、損切り・利確・決済ルールは含める。
- 研究ノートを反映する前の試行は、会話の記録から再構成して origin='retroactive' で登録する。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict

import pandas as pd

from .config import IndicatorParams, RiskParams

RETRO_NOTE = ("研究ノート反映前の試行（会話の記録から再構成した推定）。"
              "このほかに、画面の検証タブで手動で試した設定がある可能性がある（記録なし）。")


def strategy_config(key: str, params: IndicatorParams, risk: RiskParams, rules) -> dict:
    return {"strategy": key, "params": asdict(params),
            "risk": {"sl_atr": risk.sl_atr, "tp_atr": risk.tp_atr},
            "rules": {"opposite_exit": rules.opposite_exit, "max_bars": rules.max_bars}}


def ml_config(kind: str, horizon: int, threshold: float, label: str = "direction", features: str = "v1",
              k: int = 5, test_frac: float = 0.5) -> dict:
    return {"model": kind, "horizon": horizon, "threshold": threshold, "label": label, "features": features,
            "k": k, "test_frac": test_frac}


def config_hash(family: str, config: dict) -> str:
    return hashlib.sha1(json.dumps([family, config], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def record(con: sqlite3.Connection, rows: list[dict], *, pair: str, tf: str, origin: str, now: pd.Timestamp,
           data_from=None, data_to=None, n_bars: int | None = None, note: str | None = None) -> int:
    """rows の各要素：family, name, purpose, config, metrics（任意）。"""
    def ts(x):
        return None if x is None else pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")

    con.executemany(
        """INSERT INTO trials (created_at, pair, tf, family, name, purpose, config, config_hash,
           data_from, data_to, n_bars, metrics, origin, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ts(now), pair, tf, r["family"], r["name"], r["purpose"],
          json.dumps(r["config"], sort_keys=True, ensure_ascii=False), config_hash(r["family"], r["config"]),
          ts(data_from), ts(data_to), n_bars,
          None if r.get("metrics") is None else json.dumps(r["metrics"], ensure_ascii=False, default=float),
          origin, note) for r in rows],
    )
    con.commit()
    return len(rows)


def count(con: sqlite3.Connection, pair: str | None = None, tf: str | None = None, family: str | None = None,
          origin: str | None = None) -> int:
    """試した設定の種類の数。"""
    q, params = "SELECT COUNT(DISTINCT config_hash) FROM trials WHERE 1=1", []
    for col, val in (("pair", pair), ("tf", tf), ("family", family), ("origin", origin)):
        if val is not None:
            q += f" AND {col}=?"
            params.append(val)
    return int(con.execute(q, params).fetchone()[0])


def count_with(con: sqlite3.Connection, pair: str, tf: str, planned: list[dict]) -> int:
    """記録済みの設定に、これから試す予定の設定（planned）を加えた種類の数。
    同じ処理の中で複数の評価をするとき、すべて同じ「試した数」で割り引くために使う。"""
    have = {r[0] for r in con.execute("SELECT DISTINCT config_hash FROM trials WHERE pair=? AND tf=?", (pair, tf))}
    return len(have | {config_hash(r["family"], r["config"]) for r in planned})


def summary(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT family, origin, COUNT(DISTINCT config_hash) AS n_configs, COUNT(*) AS n_runs
           FROM trials WHERE pair=? AND tf=? GROUP BY family, origin ORDER BY family, origin""",
        con, params=(pair, tf),
    )


def seed_retroactive(con: sqlite3.Connection) -> None:
    """研究ノート反映前（2026-10-05 まで）に USD/JPY で試した組み合わせを、一度だけ登録する。"""
    if con.execute("SELECT 1 FROM trials WHERE origin='retroactive' LIMIT 1").fetchone():
        return
    from . import backtest as bt
    from dataclasses import replace

    now = pd.Timestamp("2026-10-05 10:00:00")
    base, risk = IndicatorParams(), RiskParams()
    old_rules = bt.ExitRules(opposite_exit=True, max_bars=0)

    # 最初の試作版（fx_signals.py）：5つのルール × 7つの時間足（全期間）
    proto = [{"family": "strategy", "name": name, "purpose": "prototype",
              "config": {"prototype": "fx_signals.py v0", "strategy": name, "timeframe": tf}}
             for name in ("総合スコア", "移動平均クロス", "RSI逆張り", "MACDクロス", "ボリンジャー逆張り")
             for tf in ("5m", "15m", "30m", "1h", "4h", "1d", "1wk")]
    record(con, proto, pair="USD/JPY", tf="mixed", origin="retroactive", now=now, note=RETRO_NOTE)

    for tf in ("1h", "1d"):
        rows = []
        # フェーズ2：ウォークフォワードのパラメータ候補（初期パラメータもこの中に含まれる）
        for key, grid in bt.PARAM_GRIDS.items():
            for ov in grid:
                rows.append({"family": "strategy", "name": key, "purpose": "wf_candidate",
                             "config": strategy_config(key, replace(base, **ov), risk, old_rules)})
        # フェーズ3：機械学習（2モデル × 予測先3種類、確信度 0.55）
        horizons = (1, 4, 24) if tf == "1h" else (1, 5, 20)
        for kind in ("logistic", "gbm"):
            for h in horizons:
                rows.append({"family": "ml", "name": kind, "purpose": "ml_eval", "config": ml_config(kind, h, 0.55)})
        record(con, rows, pair="USD/JPY", tf=tf, origin="retroactive", now=now, note=RETRO_NOTE)
