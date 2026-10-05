"""1回分の定期処理（フェーズ4）：取得 → 記録 → 答え合わせ → 再学習 → 重み → 警告。

記録の決まりごと
- 戦略のシグナルは、記録を始めた後に確定した足の分だけ記録する（過去にさかのぼって作らない）。
- 機械学習の予測は「その足より前のデータだけで学習したモデル」でのみ行う（学習に使った最後の足より後の足だけ）。
  再学習の前に、それまでのモデルで新しい足を予測しておく。
- 記録した予測・シグナルの結果は、確定した足だけで答え合わせする。
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import backtest as bt
from . import data, indicators, ml, strategies, tracking
from .config import (
    DEFAULT_SLIPPAGE, DEFAULT_SPREAD, ML_HORIZON, ML_THRESHOLD, MODELS_DIR, PAIRS, ROOT, TIMEFRAMES, TRACKING,
    IndicatorParams, RiskParams, pip_size,
)
from .tracking import ts_str

P = IndicatorParams()
RISK = RiskParams()


def default_costs(pair: str) -> bt.Costs:
    return bt.Costs(DEFAULT_SPREAD[pair], DEFAULT_SLIPPAGE)


@dataclass
class CycleResult:
    pair: str
    tf: str
    skipped: bool = False
    data_added: int = 0
    data_error: str | None = None
    signals_logged: int = 0
    predictions_logged: int = 0
    outcomes_closed: int = 0
    labels_filled: int = 0
    retrained: list[str] = field(default_factory=list)
    verdicts_updated: bool = False
    new_alerts: int = 0
    error: str | None = None

    def summary(self) -> str:
        if self.skipped:
            return "直前に実行済みのためスキップ"
        parts = [
            f"データ+{self.data_added}本", f"シグナル記録{self.signals_logged}件", f"予測記録{self.predictions_logged}件",
            f"決済の答え合わせ{self.outcomes_closed}件", f"予測の答え合わせ{self.labels_filled}件",
        ]
        if self.retrained:
            parts.append("再学習：" + "・".join(tracking.MODEL_NAMES[k] for k in self.retrained))
        if self.verdicts_updated:
            parts.append("戦略の検証結果を更新")
        if self.new_alerts:
            parts.append(f"新しい警告{self.new_alerts}件")
        if self.data_error:
            parts.append(f"データ取得エラー：{self.data_error}")
        return "、".join(parts)


def run_cycle(
    con: sqlite3.Connection, pair: str, tf_key: str, origin: str = "manual", now: pd.Timestamp | None = None,
    force: bool = False, min_interval: pd.Timedelta = pd.Timedelta(seconds=60), retrain: bool = True,
) -> CycleResult:
    now = data.now_utc() if now is None else now
    res = CycleResult(pair, tf_key)
    if not force:
        last = con.execute(
            "SELECT MAX(finished_at) FROM job_runs WHERE pair=? AND tf=? AND error IS NULL", (pair, tf_key)
        ).fetchone()[0]
        if last is not None and now - pd.Timestamp(last) < min_interval:
            res.skipped = True
            return res

    run_id = con.execute(
        "INSERT INTO job_runs (pair, tf, origin, started_at) VALUES (?, ?, ?, ?)", (pair, tf_key, origin, ts_str(now))
    ).lastrowid
    con.commit()
    try:
        _cycle(con, pair, tf_key, now, force, retrain, res)
    except Exception as e:  # 1つの通貨ペア・時間足の失敗で全体を止めない
        con.rollback()
        res.error = f"{type(e).__name__}: {e}"
    con.execute(
        "UPDATE job_runs SET finished_at=?, summary=?, error=? WHERE id=?",
        (ts_str(data.now_utc() if now is None else now), res.summary(), res.error, run_id),
    )
    con.commit()
    return res


def _cycle(con, pair, tf_key, now, force, retrain, res: CycleResult) -> None:
    tf = TIMEFRAMES[tf_key]
    settings = TRACKING[tf_key]
    pip = pip_size(PAIRS[pair])
    costs = default_costs(pair)

    upd = data.update(con, pair, PAIRS[pair], tf_key, force=force)
    res.data_added, res.data_error = upd.added, upd.error
    conf = data.confirmed(data.load(con, pair, tf_key), tf, now)
    if len(conf) < bt.WARMUP + 50:
        return
    ind = indicators.add_all(conf, P)
    X = ml.make_features(ind, tf_key)

    recent = tracking.recent_performance(ind, P, RISK, costs, pip, settings.recent_bars)
    weights = {r.source: r.weight for r in recent}
    con.executemany(
        "INSERT OR REPLACE INTO weight_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(pair, tf_key, ts_str(ind.index[-1]), r.source, r.weight, r.n, r.total,
          r.pf if math.isfinite(r.pf) else None, r.z) for r in recent],
    )

    res.signals_logged = log_strategy_signals(con, pair, tf_key, ind, weights, now)

    for kind in ml.MODELS:
        reg = latest_model(con, pair, tf_key, kind)
        if reg is not None:  # 再学習の前に、それまでのモデルで新しい足を予測する
            res.predictions_logged += predict_and_log(con, pair, tf_key, kind, reg, ind, X, now)
        if retrain and retrain_due(reg, ind, settings.retrain_every):
            reg = train_and_register(con, pair, tf_key, kind, ind, costs, pip, now)
            res.retrained.append(kind)
            res.predictions_logged += predict_and_log(con, pair, tf_key, kind, reg, ind, X, now)

    if retrain and verdicts_due(con, pair, tf_key, ind, settings.retrain_every):
        save_verdicts(con, pair, tf_key, conf, costs, pip, now)
        res.verdicts_updated = True

    res.outcomes_closed = resolve_signals(con, pair, tf_key, ind, costs, pip)
    res.labels_filled = fill_labels(con, pair, tf_key, ind)

    closed = tracking.load_signal_log(con, pair, tf_key)
    closed = closed[(closed["status"] != "進行中") & ~closed["source"].str.startswith("ml_")]
    period = "直近約1か月" if tf_key == "1h" else "直近約1年"
    conds = tracking.strategy_conditions(recent, closed, period)
    registry = tracking.load_registry(con, pair, tf_key)
    preds = tracking.load_predictions(con, pair, tf_key)
    for kind in ml.MODELS:
        conds += tracking.model_conditions(
            registry[(registry["model"] == kind) & (registry["horizon"] == ML_HORIZON)].reset_index(drop=True),
            preds[(preds["model"] == kind) & (preds["horizon"] == ML_HORIZON)],
        )
    conds += tracking.data_condition(ind.index[-1], now, tf.bar, settings.stale_bars)
    res.new_alerts = tracking.sync_alerts(con, pair, tf_key, conds, now)
    con.commit()


def log_strategy_signals(con, pair, tf_key, ind, weights, now) -> int:
    """前回処理した足より後に確定した足のシグナルを記録する。初回は最新の確定足の分だけ。"""
    n = len(ind)
    last = tracking.get_state(con, pair, tf_key, "strategy_last_bar")
    start = n - 1 if last is None else int(ind.index.searchsorted(pd.Timestamp(last), side="right"))
    max_bars = TRACKING[tf_key].log_max_bars
    rows = []
    close, atr = ind["close"].to_numpy(), ind["atr"].to_numpy()
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, P).to_numpy()
        for i in range(start, n):
            if sig[i]:
                sl, tp = strategies.levels(close[i], int(sig[i]), atr[i], RISK)
                rows.append((pair, tf_key, s.key, ts_str(ind.index[i]), int(sig[i]), float(close[i]), sl, tp,
                             float(atr[i]), max_bars, None, weights.get(s.key), ts_str(now)))
    added = _insert_signals(con, rows)
    tracking.set_state(con, pair, tf_key, "strategy_last_bar", ts_str(ind.index[-1]))
    return added


def _insert_signals(con, rows) -> int:
    before = con.total_changes
    con.executemany(
        """INSERT OR IGNORE INTO signal_log
           (pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars, prob, weight, logged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows,
    )
    return con.total_changes - before


def latest_model(con, pair, tf_key, kind) -> dict | None:
    cur = con.execute(
        "SELECT * FROM model_registry WHERE pair=? AND tf=? AND model=? AND horizon=? ORDER BY trained_at DESC LIMIT 1",
        (pair, tf_key, kind, ML_HORIZON),
    )
    row = cur.fetchone()
    return dict(zip([c[0] for c in cur.description], row)) if row else None


def retrain_due(reg: dict | None, ind: pd.DataFrame, every: int) -> bool:
    if reg is None:
        return True
    pos = int(ind.index.searchsorted(pd.Timestamp(reg["trained_bar"]), side="right"))
    return len(ind) - pos >= every


def train_and_register(con, pair, tf_key, kind, ind, costs, pip, now) -> dict:
    ev = ml.evaluate_model(ind, tf_key, kind, ML_HORIZON, ML_THRESHOLD, RISK, costs, pip)
    version = f"{pair.replace('/', '')}_{tf_key}_{kind}_h{ML_HORIZON}_{now:%Y%m%d%H%M%S}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{version}.joblib"
    joblib.dump(ev.model, path)
    reg = dict(
        version=version, pair=pair, tf=tf_key, model=kind, horizon=ML_HORIZON, trained_at=ts_str(now),
        trained_bar=ts_str(ind.index[-1]), trained_through=ts_str(ev.trained_through),
        n_train=int(ind.index.searchsorted(ev.trained_through) + 1),
        acc=ev.acc, majority_acc=ev.majority_acc, logloss=ev.logloss, base_logloss=ev.base_logloss,
        trade_pips=ev.trade_metrics["total"], beat=ev.beat, ok=int(ev.ok),
        path=path.relative_to(ROOT).as_posix(),
    )
    con.execute(f"INSERT OR REPLACE INTO model_registry ({', '.join(reg)}) VALUES ({', '.join('?' * len(reg))})",
                tuple(reg.values()))
    return reg


def predict_and_log(con, pair, tf_key, kind, reg: dict, ind: pd.DataFrame, X: pd.DataFrame, now) -> int:
    """学習に使った最後の足より後で、まだ記録していない足を予測する。初めて記録するときは最新の確定足だけ。"""
    through = pd.Timestamp(reg["trained_through"])
    last_logged = con.execute(
        "SELECT MAX(bar_time) FROM prediction_log WHERE pair=? AND tf=? AND model=? AND horizon=?",
        (pair, tf_key, kind, ML_HORIZON),
    ).fetchone()[0]
    n = len(ind)
    if last_logged is None:
        rows = [n - 1] if ind.index[-1] > through else []
    else:
        after = max(through, pd.Timestamp(last_logged))
        rows = list(range(int(ind.index.searchsorted(after, side="right")), n))
    Xv = X.to_numpy()
    rows = [i for i in rows if not np.isnan(Xv[i]).any()]
    if not rows:
        return 0

    model = joblib.load(ROOT / reg["path"])
    p = model.predict_proba(Xv[rows])[:, 1]
    ok = bool(reg["ok"])
    preds, sigs = [], []
    close, atr = ind["close"].to_numpy(), ind["atr"].to_numpy()
    for i, pi in zip(rows, p):
        dec = ml.decide(float(pi), ok, ML_THRESHOLD)
        preds.append((pair, tf_key, kind, ML_HORIZON, ts_str(ind.index[i]), float(pi), dec, reg["version"], ts_str(now)))
        if dec in ("買い", "売り"):
            d = 1 if dec == "買い" else -1
            sl, tp = strategies.levels(close[i], d, atr[i], RISK)
            sigs.append((pair, tf_key, f"ml_{kind}", ts_str(ind.index[i]), d, float(close[i]), sl, tp, float(atr[i]),
                         ML_HORIZON, float(pi), None, ts_str(now)))
    before = con.total_changes
    con.executemany(
        """INSERT OR IGNORE INTO prediction_log (pair, tf, model, horizon, bar_time, p, decision, version, logged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", preds,
    )
    added = con.total_changes - before
    _insert_signals(con, sigs)
    return added


def verdicts_due(con, pair, tf_key, ind, every: int) -> bool:
    last = con.execute("SELECT MAX(bar_time) FROM backtest_verdicts WHERE pair=? AND tf=?", (pair, tf_key)).fetchone()[0]
    if last is None:
        return True
    return len(ind) - int(ind.index.searchsorted(pd.Timestamp(last), side="right")) >= every


def save_verdicts(con, pair, tf_key, conf: pd.DataFrame, costs, pip, now) -> None:
    """フェーズ2の評価（初期設定）を行い、戦略ごとの判定を保存する（相場の解説で使う）。"""
    rep = bt.evaluate_all(conf[["open", "high", "low", "close"]], P, RISK, bt.ExitRules(), costs, pip, runs=100)
    con.executemany(
        "INSERT OR REPLACE INTO backtest_verdicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(pair, tf_key, r.strategy.key, ts_str(now), ts_str(conf.index[-1]), r.verdict[0], r.verdict[1],
          r.wf.test_metrics["total"], r.wf_beat) for r in rep.reports],
    )


def resolve_signals(con, pair, tf_key, ind, costs, pip) -> int:
    rows = con.execute(
        "SELECT id, signal_time, direction, entry, sl, tp, max_bars FROM signal_log "
        "WHERE pair=? AND tf=? AND status='進行中'", (pair, tf_key),
    ).fetchall()
    pos = {t: i for i, t in enumerate(ind.index)}
    closed = 0
    for rid, t, d, entry, sl, tp, max_bars in rows:
        i = pos.get(pd.Timestamp(t))
        if i is None:
            continue
        o = strategies.resolve_trade(ind, i, d, sl, tp, max_bars)
        if o.status == "進行中":
            continue
        pips = d * (o.exit_price - entry) / pip - costs.round_trip
        con.execute("UPDATE signal_log SET status=?, exit_time=?, exit_price=?, pips=? WHERE id=?",
                    (o.status, ts_str(o.exit_time), float(o.exit_price), float(pips), rid))
        closed += 1
    return closed


def fill_labels(con, pair, tf_key, ind) -> int:
    rows = con.execute(
        "SELECT model, horizon, bar_time FROM prediction_log WHERE pair=? AND tf=? AND y IS NULL", (pair, tf_key)
    ).fetchall()
    pos = {t: i for i, t in enumerate(ind.index)}
    close = ind["close"].to_numpy()
    filled = 0
    for model, h, t in rows:
        i = pos.get(pd.Timestamp(t))
        if i is None or i + h >= len(ind):
            continue
        con.execute("UPDATE prediction_log SET y=? WHERE pair=? AND tf=? AND model=? AND horizon=? AND bar_time=?",
                    (float(close[i + h] > close[i]), pair, tf_key, model, h, t))
        filled += 1
    return filled
