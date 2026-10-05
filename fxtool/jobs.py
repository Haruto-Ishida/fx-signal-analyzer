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
from . import data, holdout, indicators, ml, strategies, tracking, trials
from .config import (
    DEFAULT_SLIPPAGE, DEFAULT_SPREAD, ML_HORIZON, ML_THRESHOLD, MODELS_DIR, PAIRS, ROOT, TIMEFRAMES, TRACKING,
    VERTICAL_BARRIER, IndicatorParams, RiskParams, pip_size,
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
    raw = data.load(con, pair, tf_key)
    conf = data.confirmed(raw, tf, now)
    if len(conf) < bt.WARMUP + 50:
        return
    b = data.boundary(con, pair, tf_key, now)
    ind = indicators.add_all(conf, P)
    ind_dev = holdout.dev_part(ind, b)  # 選ぶ判断（検証・条件の判定）にはこれだけを使う
    X = ml.make_features(ind, tf_key)

    # 戦略の重みは、直近のバックテスト（＝最終テスト期間）から計算していたため、フェーズ4で本番の記録にもとづく
    # 方法に作り直すまで停止する（全戦略 1.0）。直近成績の悪化警告も同じ理由で、本番の記録にもとづくものだけ出す。
    weights = {s.key: 1.0 for s in strategies.STRATEGIES}

    res.signals_logged = log_strategy_signals(con, pair, tf_key, ind, weights, now)

    for kind in ml.MODELS:
        reg = latest_model(con, pair, tf_key, kind)
        if reg is not None:  # 再学習の前に、それまでのモデルで新しい足を予測する
            res.predictions_logged += predict_and_log(con, pair, tf_key, kind, reg, ind, X, now)
        if retrain and retrain_due(reg, ind, settings.retrain_every):
            reg = train_and_register(con, pair, tf_key, kind, ind, ind_dev, b, costs, pip, now)
            res.retrained.append(kind)
            res.predictions_logged += predict_and_log(con, pair, tf_key, kind, reg, ind, X, now)

    if retrain and verdicts_due(con, pair, tf_key, ind_dev):
        save_verdicts(con, pair, tf_key, holdout.dev_part(conf, b), b, costs, pip, now)
        res.verdicts_updated = True

    fill_entries(con, pair, tf_key, raw)
    res.outcomes_closed = resolve_signals(con, pair, tf_key, ind, costs, pip)
    res.labels_filled = fill_labels(con, pair, tf_key, ind)

    closed = tracking.load_signal_log(con, pair, tf_key)
    closed = closed[closed["status"].isin(["利確", "損切り", "期限"]) & ~closed["source"].str.startswith("ml_")]
    period = "直近約1か月" if tf_key == "1h" else "直近約1年"
    conds = tracking.strategy_conditions([], closed, period)
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
    """前回処理した足より後に確定した足のシグナルを記録する。初回は最新の確定足の分だけ。
    エントリーは次の足の始値なので、ここでは「待機」として記録し、fill_entries で埋める。"""
    n = len(ind)
    last = tracking.get_state(con, pair, tf_key, "strategy_last_bar")
    start = n - 1 if last is None else int(ind.index.searchsorted(pd.Timestamp(last), side="right"))
    max_bars = TRACKING[tf_key].log_max_bars
    rows = []
    atr = ind["atr"].to_numpy()
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, P).to_numpy()
        for i in range(start, n):
            if sig[i]:
                rows.append((pair, tf_key, s.key, ts_str(ind.index[i]), int(sig[i]), float(atr[i]), max_bars,
                             None, weights.get(s.key), ts_str(now)))
    added = _insert_signals(con, rows)
    tracking.set_state(con, pair, tf_key, "strategy_last_bar", ts_str(ind.index[-1]))
    return added


def _insert_signals(con, rows) -> int:
    """rows：(pair, tf, source, signal_time, direction, atr, max_bars, prob, weight, logged_at)。"""
    before = con.total_changes
    con.executemany(
        """INSERT OR IGNORE INTO signal_log
           (pair, tf, source, signal_time, direction, atr, max_bars, prob, weight, logged_at, status, entry_basis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '待機', 'next_open')""", rows,
    )
    return con.total_changes - before


def fill_entries(con, pair, tf_key, raw: pd.DataFrame) -> int:
    """「待機」のシグナルに、次の足の始値（形成中の足でも始値は確定している）でエントリー・損切り・利確を入れる。"""
    rows = con.execute(
        "SELECT id, signal_time, direction, atr FROM signal_log WHERE pair=? AND tf=? AND status='待機'",
        (pair, tf_key),
    ).fetchall()
    pos = {t: i for i, t in enumerate(raw.index)}
    opens = raw["open"].to_numpy()
    filled = 0
    for rid, t, d, atr in rows:
        i = pos.get(pd.Timestamp(t))
        if i is None or i + 1 >= len(raw):
            continue
        entry = float(opens[i + 1])
        sl, tp = strategies.levels(entry, d, atr, RISK)
        con.execute("UPDATE signal_log SET entry=?, sl=?, tp=?, status='進行中' WHERE id=?", (entry, sl, tp, rid))
        filled += 1
    return filled


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
    if reg.get("eval_to") is None:
        return True  # 研究ノート反映前のモデル（条件を全期間のデータで判定していた）は、すぐに判定し直す
    pos = int(ind.index.searchsorted(pd.Timestamp(reg["trained_bar"]), side="right"))
    return len(ind) - pos >= every


def train_and_register(con, pair, tf_key, kind, ind, ind_dev, b, costs, pip, now) -> dict:
    """条件の判定（検証）は開発用データだけで行い、本番モデルは同じ設定のまま最新データまで使って学習する。
    開発用データは境界が固定なので変わらない。同じ設定・同じ期間の検証結果があれば使い回す。"""
    eval_to = ts_str(ind_dev.index[-1])
    prev = con.execute(
        """SELECT acc, majority_acc, logloss, base_logloss, trade_pips, beat, ok, eval_from FROM model_registry
           WHERE pair=? AND tf=? AND model=? AND horizon=? AND eval_to=? ORDER BY trained_at DESC LIMIT 1""",
        (pair, tf_key, kind, ML_HORIZON, eval_to),
    ).fetchone()
    if prev is None:
        ev = ml.evaluate_model(ind_dev, tf_key, kind, ML_HORIZON, ML_THRESHOLD, RISK, costs, pip,
                               holdout_start=b.start)
        evald = dict(acc=ev.acc, majority_acc=ev.majority_acc, logloss=ev.logloss, base_logloss=ev.base_logloss,
                     trade_pips=ev.trade_metrics["total"], beat=ev.beat, ok=int(ev.ok),
                     eval_from=ts_str(ind_dev.index[int(ev.pred["i"].min())]))
        trials.record(con, [ml.trial_row(ev, 5, 0.5)], pair=pair, tf=tf_key, origin="job", now=now,
                      data_from=ind_dev.index[0], data_to=ind_dev.index[-1], n_bars=len(ind_dev))
    else:
        evald = dict(zip(["acc", "majority_acc", "logloss", "base_logloss", "trade_pips", "beat", "ok", "eval_from"], prev))

    model, train_from, train_to = ml.fit_production(ind, tf_key, kind, ML_HORIZON)
    version = f"{pair.replace('/', '')}_{tf_key}_{kind}_h{ML_HORIZON}_{now:%Y%m%d%H%M%S}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{version}.joblib"
    joblib.dump(model, path)
    reg = dict(
        version=version, pair=pair, tf=tf_key, model=kind, horizon=ML_HORIZON, trained_at=ts_str(now),
        trained_bar=ts_str(ind.index[-1]), trained_through=ts_str(train_to),
        n_train=int(ind.index.searchsorted(train_to) - ind.index.searchsorted(train_from) + 1),
        **evald, path=path.relative_to(ROOT).as_posix(), eval_to=eval_to, train_from=ts_str(train_from),
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
            sigs.append((pair, tf_key, f"ml_{kind}", ts_str(ind.index[i]), d, float(atr[i]),
                         ML_HORIZON, float(pi), None, ts_str(now)))
    before = con.total_changes
    con.executemany(
        """INSERT OR IGNORE INTO prediction_log (pair, tf, model, horizon, bar_time, p, decision, version, logged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", preds,
    )
    added = con.total_changes - before
    _insert_signals(con, sigs)
    return added


def exit_rules(tf_key: str) -> bt.ExitRules:
    """トリプルバリア：利確・損切り（RISK）と、VERTICAL_BARRIER 本での時間切れ。反対シグナルでもドテン。"""
    return bt.ExitRules(opposite_exit=True, max_bars=VERTICAL_BARRIER[tf_key])


def verdicts_due(con, pair, tf_key, ind_dev) -> bool:
    """開発用データ（境界で固定）の最後の足で、DSR・PBO まで計算した判定がまだなければ計算する。"""
    have = con.execute(
        "SELECT COUNT(*) FROM backtest_verdicts WHERE pair=? AND tf=? AND bar_time=? AND n_trials IS NOT NULL",
        (pair, tf_key, ts_str(ind_dev.index[-1]))).fetchone()[0]
    return have < len(strategies.STRATEGIES)


def save_verdicts(con, pair, tf_key, dev: pd.DataFrame, b, costs, pip, now) -> None:
    """フェーズ2の評価（初期設定）を開発用データだけで行い、戦略ごとの判定を保存する（相場の解説で使う）。"""
    rep = bt.evaluate_all(dev[["open", "high", "low", "close"]], P, RISK, exit_rules(tf_key), costs, pip, runs=100,
                          holdout_start=b.start, rw_paths=20)
    trials.record(con, rep.trials, pair=pair, tf=tf_key, origin="job", now=now,
                  data_from=dev.index[0], data_to=dev.index[-1], n_bars=len(dev))
    rep = bt.finalize(rep, trials.count(con, pair, tf_key))  # 記録済みの試行すべてで DSR を割り引く
    con.executemany(
        """INSERT OR REPLACE INTO backtest_verdicts
           (pair, tf, source, computed_at, bar_time, verdict, reason, wf_total, wf_beat, dsr, pbo, n_trials, wf_n)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(pair, tf_key, r.strategy.key, ts_str(now), ts_str(dev.index[-1]), r.verdict[0], r.verdict[1],
          r.wf.test_metrics["total"], r.wf_beat, _num(r.dsr.get("dsr")), _num(r.pbo.get("pbo")), rep.n_trials,
          r.wf.test_metrics["n"]) for r in rep.reports],
    )


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


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
