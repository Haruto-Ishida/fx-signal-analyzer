"""1回分の定期処理：取得 → 記録 → 機械学習の判定 → 再学習 → 答え合わせ → 重み → 警告。

記録の決まりごと
- 戦略のシグナルは、記録を始めた後に確定した足の分だけ記録する（過去にさかのぼって作らない）。
  エントリーは次の足の始値（それまでは「待機」）。
- 機械学習（メタラベリング）の判定は、そのシグナルの足までのデータで学習したモデルでだけ行う
  （モデルを学習した時点の最新の足 trained_bar 以降のシグナルだけ）。再学習の前に、それまでのモデルで判定しておく。
- 戦略・設定・モデルを選ぶ判断（戦略の判定・機械学習の条件・DSR）は開発用データだけで行う。
  本番モデルは、選んだ設定を固定したまま、決着したすべてのラベルで学習する。
- 重みと成績悪化の警告は、本番の記録（＝最も信頼できる最終テスト）から計算する。
- 本番の記録の損益が、日本時間のその日・その週の上限に達したら、新しいシグナルは「停止中」として記録する。
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import backtest as bt
from . import (costmodel, data, events, holdout, indicators, killswitch, ml, ml_eval, mlops, strategies, tracking,
               trials)
from . import settings as user_settings
from .config import (
    DAILY_MAX_LOSS_R, DEFAULT_SLIPPAGE, DEFAULT_SPREAD, LIVE_WEIGHT_MIN_TRADES, LIVE_WEIGHT_WINDOW, ML_META_FOLDS,
    ML_META_THRESHOLD, MODELS_DIR, PAIRS, ROOT, TIMEFRAMES, TRACKING, VERTICAL_BARRIER, WEEKLY_MAX_LOSS_R,
    IndicatorParams, RiskParams, pip_size,
)
from .tracking import ts_str

P = IndicatorParams()
RISK = RiskParams()
LIVE_LABEL = f"本番の記録（直近{LIVE_WEIGHT_WINDOW}回）"

# 定期処理での検証の細かさ（ランダム売買・ランダムウォークの試行回数）。テストでは小さくして速くする
VERDICT_RANDOM_RUNS = 100
VERDICT_RW_PATHS = 20
META_RANDOM_RUNS = 200


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
    meta_judged: int = 0
    outcomes_closed: int = 0
    labels_filled: int = 0
    retrained: bool = False
    verdicts_updated: bool = False
    new_alerts: int = 0
    halted: str = ""
    error: str | None = None

    def summary(self) -> str:
        if self.skipped:
            return "直前に実行済みのためスキップ"
        parts = [f"データ+{self.data_added}本", f"シグナル記録{self.signals_logged}件",
                 f"機械学習の判定{self.meta_judged}件", f"決済の答え合わせ{self.outcomes_closed}件"]
        if self.retrained:
            parts.append("機械学習を再学習")
        if self.verdicts_updated:
            parts.append("戦略の検証結果を更新")
        if self.halted:
            parts.append("最大損失・キルスイッチのため新しいシグナルを停止中")
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
    con.execute("UPDATE job_runs SET finished_at=?, summary=?, error=? WHERE id=?",
                (ts_str(now), res.summary(), res.error, run_id))
    con.commit()
    return res


SCHEDULE_GAP = pd.Timedelta(hours=3)  # 毎時の定期実行がこれ以上動いていなければ、画面で知らせる


def schedule_gap(con, now: pd.Timestamp) -> pd.Timedelta | None:
    """最後の定期実行（タスクスケジューラ）から SCHEDULE_GAP 以上たっていれば、その時間。定期実行の記録がなければ None。
    （パソコンが止まっていた・ログオフしていた・電源につながっていなかった など。要件定義書 9-1）"""
    last = con.execute("SELECT MAX(started_at) FROM job_runs WHERE origin='schedule'").fetchone()[0]
    if last is None:
        return None
    gap = now - pd.Timestamp(last)
    return gap if gap >= SCHEDULE_GAP else None


def closed_signals(con, pair, tf_key) -> pd.DataFrame:
    sl = tracking.load_signal_log(con, pair, tf_key)
    return sl[sl["status"].isin(tracking.DONE)]


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

    # 先に答え合わせをして、最新の決済まで反映した記録で重みと最大損失を判断する
    fill_entries(con, pair, tf_key, raw)
    res.outcomes_closed = resolve_signals(con, pair, tf_key, ind, costs, pip)

    # 本番の記録にもとづく重みと、最大損失による停止
    closed = closed_signals(con, pair, tf_key)
    closed_strat = closed[~closed["source"].str.startswith("ml_")]
    live = tracking.live_performance(closed_strat, LIVE_WEIGHT_WINDOW, LIVE_WEIGHT_MIN_TRADES)
    con.executemany(
        """INSERT OR REPLACE INTO weight_history
           (pair, tf, bar_time, source, weight, recent_n, recent_pips, recent_pf, z, basis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'live')""",
        [(pair, tf_key, ts_str(ind.index[-1]), r.source, r.weight, r.n, r.total,
          r.pf if math.isfinite(r.pf) else None, r.z) for r in live],
    )
    halt = tracking.loss_halt(closed, now, RISK.sl_atr, pip, DAILY_MAX_LOSS_R, WEEKLY_MAX_LOSS_R)
    # キルスイッチ（ペーパー口座全体の記録から。要件定義書 8章）
    s = user_settings.load()
    _, kill = killswitch.check(con, now, s, data.latest_close(con, "USD/JPY"))
    # 1日の回数は、シグナルごとにその日の回数で判断する（まとめて記録するときに、別の日の分まで止めないように）
    stop_reasons = [x for x in (halt.reason if halt.active else "",
                                " ／ ".join(t.reason for t in kill.trips if t.kind != "frequency")) if x]
    res.halted = " ／ ".join([*stop_reasons, *(t.reason for t in kill.trips if t.kind == "frequency")])

    res.signals_logged = log_strategy_signals(con, pair, tf_key, ind, {r.source: r.weight for r in live}, now,
                                              suppressed=bool(stop_reasons), reason=" ／ ".join(stop_reasons),
                                              max_per_day=s.kill_switch.max_signals_per_day,
                                              event_filter=s.event_filter)
    fill_entries(con, pair, tf_key, raw)

    # この処理で試す予定の設定も含めた「試した数」。戦略の判定と機械学習の条件を、同じ数で割り引く
    spec = mlops.current_spec(con, pair, tf_key)
    planned = (bt.planned_trials(P, RISK, exit_rules(tf_key))
               + ml_eval.planned_trials("meta", ML_META_THRESHOLD, VERTICAL_BARRIER[tf_key], ML_META_FOLDS, spec.features))
    n_trials = trials.count_with(con, pair, tf_key, planned)

    # 機械学習（メタラベリング）：再学習の前に、それまでのモデルで新しいシグナルを判定する。
    # 前のバージョンに戻して固定しているときは、そのモデルを使い、自動の再学習はしない
    reg = mlops.active_meta(con, pair, tf_key)
    pinned = mlops.pinned_version(con, pair, tf_key) is not None
    if reg is not None:
        res.meta_judged += judge_signals(con, pair, tf_key, reg, ind)
    if (retrain and not pinned and (meta_retrain_due(reg, ind, settings.retrain_every, n_trials)
                                    or (reg is not None and (reg.get("features") or "v2") != spec.features))):
        reg = train_meta(con, pair, tf_key, ind, ind_dev, b, costs, pip, now, n_trials, spec)
        res.retrained = True
        res.meta_judged += judge_signals(con, pair, tf_key, reg, ind)

    # チャレンジャー（シャドーモード）：裏で判定だけ記録し、条件を満たしたらユーザーに提案する
    if retrain:
        mlops.shadow_cycle(con, pair, tf_key, ind, RISK, P, costs, pip, now)
        mlops.review(con, pair, tf_key, ind_dev, b, P, RISK, costs, pip, now, n_trials)

    if retrain and verdicts_due(con, pair, tf_key, ind_dev, n_trials):
        save_verdicts(con, pair, tf_key, holdout.dev_part(conf, b), b, costs, pip, now, n_trials)
        res.verdicts_updated = True

    res.outcomes_closed += resolve_signals(con, pair, tf_key, ind, costs, pip)
    res.labels_filled = fill_labels(con, pair, tf_key, ind)  # 研究ノート反映前の予測の記録の答え合わせ

    closed = closed_signals(con, pair, tf_key)
    closed_strat = closed[~closed["source"].str.startswith("ml_")]
    live = tracking.live_performance(closed_strat, LIVE_WEIGHT_WINDOW, LIVE_WEIGHT_MIN_TRADES)
    conds = tracking.strategy_conditions(live, closed_strat, LIVE_LABEL)
    registry = tracking.load_registry(con, pair, tf_key)
    conds += tracking.meta_conditions(registry[registry["model"] == "meta"].reset_index(drop=True),
                                      tracking.load_signal_log(con, pair, tf_key))
    conds += tracking.data_condition(ind.index[-1], now, tf.bar, settings.stale_bars)
    if halt.active:
        conds.append(tracking.Condition("risk", "max_loss", "critical", halt.reason))
    for t in kill.trips:
        conds.append(tracking.Condition("risk", f"kill_{t.kind}", "critical", "キルスイッチ：" + t.reason))
    ch = mlops.challengers(con, pair, tf_key)
    for c in ch[ch["status"] == "ready"].itertuples():
        conds.append(tracking.Condition("ml_meta", f"challenger_{c.id}", "warning",
                                        f"機械学習：チャレンジャー「{mlops.SPECS[c.spec].label}」が入れ替えの条件を満たしました。"
                                        "「機械学習予測」タブで確認して、承認するか決めてください"))
    if mlops.pinned_version(con, pair, tf_key):
        conds.append(tracking.Condition("ml_meta", "pinned", "warning",
                                        f"機械学習：前のバージョン（{mlops.pinned_version(con, pair, tf_key)}）に戻して固定しています。"
                                        "自動の再学習は止まっています"))
    res.new_alerts = tracking.sync_alerts(con, pair, tf_key, conds, now)
    con.commit()


# ---------------- シグナルの記録 ----------------

def log_strategy_signals(con, pair, tf_key, ind, weights, now, suppressed: bool = False, reason: str = "",
                         max_per_day: int | None = None, event_filter=None) -> int:
    """前回処理した足より後に確定した足のシグナルを記録する。初回は最新の確定足の分だけ。
    エントリーは次の足の始値なので、ここでは「待機」として記録し、fill_entries で埋める。
    次のときは suppressed=1 と理由を付けて記録だけする（停止のルールの効果を後で確かめられるように）：
    - suppressed=True（最大損失・キルスイッチで停止中。理由は reason）
    - シグナルが出た時刻（足が確定した時刻）が重要な経済指標の前後（event_filter を渡したとき）
    - そのシグナルの日（日本時間）に、口座全体でもう max_per_day 回シグナルを出していた（キルスイッチの「1日の回数」。
      その日の停止を手動で解除していれば止めない）"""
    from . import paper
    used: dict[pd.Timestamp, int] = {}
    released_days: set[str] = set()
    if max_per_day is not None:
        el = paper.eligible(con)
        used = ((el["signal_time"] + tracking.JST).dt.normalize()).value_counts().to_dict() if len(el) else {}
        released_days = {k.split(":", 1)[1] for k in killswitch.released_keys(con) if k.startswith("frequency:")}
    n = len(ind)
    last = tracking.get_state(con, pair, tf_key, "strategy_last_bar")
    start = n - 1 if last is None else int(ind.index.searchsorted(pd.Timestamp(last), side="right"))
    max_bars = TRACKING[tf_key].log_max_bars
    bar = TIMEFRAMES[tf_key].bar
    found = []
    atr = ind["atr"].to_numpy()
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, P).to_numpy()
        for i in range(start, n):
            if sig[i]:
                found.append((i, s.key, int(sig[i])))
    rows = []
    for i, key, d in sorted(found):
        why = reason if suppressed else ""
        if not why and event_filter is not None:
            e = events.blackout(PAIRS[pair], ind.index[i] + bar, event_filter)
            if e is not None:
                why = f"経済指標の前後（{e.name}・日本時間 {e.jst_label()}）"
        if not why and max_per_day is not None:
            day = (ind.index[i] + tracking.JST).normalize()
            if used.get(day, 0) >= max_per_day and f"{day:%Y-%m-%d}" not in released_days:
                why = f"1日のシグナルの回数の上限（{max_per_day}回。キルスイッチ）"
            else:
                used[day] = used.get(day, 0) + 1
        rows.append((pair, tf_key, key, ts_str(ind.index[i]), d, float(atr[i]), max_bars,
                     None, weights.get(key), ts_str(now), int(bool(why)), why or None))
    added = _insert_signals(con, rows)
    tracking.set_state(con, pair, tf_key, "strategy_last_bar", ts_str(ind.index[-1]))
    return added


def _insert_signals(con, rows) -> int:
    """rows：(pair, tf, source, signal_time, direction, atr, max_bars, prob, weight, logged_at, suppressed[, 理由])。"""
    rows = [r if len(r) == 12 else (*r, None) for r in rows]
    before = con.total_changes
    con.executemany(
        """INSERT OR IGNORE INTO signal_log
           (pair, tf, source, signal_time, direction, atr, max_bars, prob, weight, logged_at, suppressed,
            suppress_reason, status, entry_basis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '待機', 'next_open')""", rows,
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


# ---------------- 機械学習（メタラベリング） ----------------

def latest_meta(con, pair, tf_key) -> dict | None:
    """本番で使うメタラベリングのモデル（前のバージョンに戻して固定していればそれ。mlops.active_meta）。"""
    return mlops.active_meta(con, pair, tf_key)


def meta_retrain_due(reg: dict | None, ind: pd.DataFrame, every: int, n_trials: int) -> bool:
    if reg is None or reg.get("eval_to") is None:
        return True
    if (reg.get("n_trials") or 0) < n_trials:
        return True  # 試した数が増えたら、DSR の割り引きをやり直す
    pos = int(ind.index.searchsorted(pd.Timestamp(reg["trained_bar"]), side="right"))
    return len(ind) - pos >= every


def train_meta(con, pair, tf_key, ind, ind_dev, b, costs, pip, now, n_trials: int, spec=None) -> dict:
    """条件の判定とモデルの選択は開発用データだけで行い（同じ期間・同じ試行数・同じ特徴量なら使い回す）、
    本番モデルは選んだモデルの種類のまま、決着したすべてのラベルで学習する。spec は本番の設定（mlops.MetaSpec）。"""
    spec = spec or mlops.SPECS["base"]
    max_bars = VERTICAL_BARRIER[tf_key]
    holdout.assert_dev_only(ind_dev, b)
    eval_to = ts_str(ind_dev.index[-1])
    cur = con.execute(
        """SELECT kind, acc, majority_acc, logloss, base_logloss, trade_pips, beat, ok, eval_from, n_trials
           FROM model_registry WHERE pair=? AND tf=? AND model='meta' AND eval_to=? AND threshold=?
             AND COALESCE(features, 'v2')=?
           ORDER BY trained_at DESC LIMIT 1""", (pair, tf_key, eval_to, ML_META_THRESHOLD, spec.features))
    prev = cur.fetchone()
    if prev is not None and prev[-1] == n_trials:
        evald = dict(zip([c[0] for c in cur.description], prev))
    else:
        smp_dev = ml_eval.build_meta(ind_dev, tf_key, P, RISK, max_bars, costs.round_trip, pip, fracdiff=spec.fracdiff)
        rep = ml_eval.evaluate_all(smp_dev, ML_META_THRESHOLD, ML_META_FOLDS, runs=META_RANDOM_RUNS)
        trials.record(con, rep.trials, pair=pair, tf=tf_key, origin="job", now=now,
                      data_from=ind_dev.index[0], data_to=ind_dev.index[-1], n_bars=len(ind_dev))
        n_now = n_trials
        rep = ml_eval.finalize(rep, smp_dev, n_now)
        ad = rep.results[rep.adopted]
        evald = dict(kind=rep.adopted, acc=ad.acc, majority_acc=ad.majority_acc, logloss=ad.logloss,
                     base_logloss=ad.base_logloss, trade_pips=ad.trade_metrics["total"], beat=ad.beat, ok=int(ad.ok),
                     eval_from=ts_str(ind_dev.index[ad.folds[0][0]]), n_trials=n_now)

    smp = ml_eval.build_meta(ind, tf_key, P, RISK, max_bars, costs.round_trip, pip, fracdiff=spec.fracdiff)  # 決着したラベルだけ
    model = ml_eval.fit_production(smp, evald["kind"])
    version = (f"{pair.replace('/', '')}_{tf_key}_meta_{evald['kind']}{'_fd' if spec.fracdiff else ''}"
               f"_{now:%Y%m%d%H%M%S}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{version}.joblib"
    joblib.dump(model, path)
    reg = dict(
        version=version, pair=pair, tf=tf_key, model="meta", horizon=max_bars, trained_at=ts_str(now),
        trained_bar=ts_str(ind.index[-1]), trained_through=ts_str(ind.index[int(smp.t0.max())]), n_train=len(smp.y),
        **evald, path=path.relative_to(ROOT).as_posix(), eval_to=eval_to,
        train_from=ts_str(ind.index[int(smp.t0.min())]), mode="meta", threshold=ML_META_THRESHOLD,
        features=spec.features,
    )
    con.execute(f"INSERT OR REPLACE INTO model_registry ({', '.join(reg)}) VALUES ({', '.join('?' * len(reg))})",
                tuple(reg.values()))
    return reg


def judge_signals(con, pair, tf_key, reg: dict, ind: pd.DataFrame) -> int:
    """まだ判定していない戦略のシグナルのうち、モデルを学習した時点の最新の足（trained_bar）以降のものを判定する。
    それより前のシグナルは、判定の時点で使えなかったはずの情報で学習したモデルになるので判定しない。"""
    rows = con.execute(
        """SELECT id, source, signal_time, direction FROM signal_log
           WHERE pair=? AND tf=? AND meta_p IS NULL AND entry_basis='next_open' AND source NOT LIKE 'ml_%'
             AND signal_time >= ?""", (pair, tf_key, reg["trained_bar"]),
    ).fetchall()
    pos = {t: i for i, t in enumerate(ind.index)}
    rows = [(rid, src, pos[pd.Timestamp(t)], d) for rid, src, t, d in rows if pd.Timestamp(t) in pos]
    if not rows:
        return 0
    F = ml.make_features(ind, tf_key, fracdiff=mlops.model_features(reg))  # そのモデルを学習したときと同じ特徴量
    X = ml_eval.meta_features(F, np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                              np.array([r[1] for r in rows]))
    ok = ~X.isna().any(axis=1).to_numpy()
    if not ok.any():
        return 0
    model = joblib.load(ROOT / reg["path"])
    p = model.predict_proba(X.to_numpy()[ok])[:, 1]
    threshold = reg.get("threshold") or ML_META_THRESHOLD
    for (rid, *_), pi in zip([r for r, k in zip(rows, ok) if k], p):
        con.execute("UPDATE signal_log SET meta_p=?, meta_decision=?, meta_version=? WHERE id=?",
                    (float(pi), "乗る" if pi >= threshold else "見送り", reg["version"], rid))
    return int(ok.sum())


# ---------------- 戦略の判定（フェーズ2） ----------------

VERDICT_STATE = f"{costmodel.COST_MODEL_VERSION}|{bt.VERDICT_VERSION}"  # コストの見積もり方か判定の基準が変わったら計算し直す


def exit_rules(tf_key: str) -> bt.ExitRules:
    """トリプルバリア：利確・損切り（RISK）と、VERTICAL_BARRIER 本での時間切れ。反対シグナルでもドテン。"""
    return bt.ExitRules(opposite_exit=True, max_bars=VERTICAL_BARRIER[tf_key])


def verdicts_due(con, pair, tf_key, ind_dev, n_trials: int) -> bool:
    """開発用データ（境界で固定）の最後の足で計算した判定がまだないか、試した数が増えて DSR の割り引きが古くなったか、
    コストの見積もり方が変わったら計算する。"""
    if tracking.get_state(con, pair, tf_key, "verdict_cost_model") != VERDICT_STATE:
        return True
    # 最初からある5つの戦略の判定だけを見る（有名手法の判定は画面のボタンで計算し直すので、ここでは見ない）
    keys = [s.key for s in strategies.STRATEGIES]
    rows = con.execute(
        f"""SELECT n_trials FROM backtest_verdicts WHERE pair=? AND tf=? AND bar_time=? AND n_trials IS NOT NULL
            AND source IN ({",".join("?" * len(keys))})""",
        (pair, tf_key, ts_str(ind_dev.index[-1]), *keys)).fetchall()
    if len(rows) < len(strategies.STRATEGIES):
        return True
    return min(r[0] for r in rows) < n_trials


def save_verdicts(con, pair, tf_key, dev: pd.DataFrame, b, costs, pip, now, n_trials: int) -> None:
    """フェーズ2の評価（初期設定）を開発用データだけで行い、戦略ごとの判定を保存する（相場の解説で使う）。
    コストは時間帯によって広めに見積もる（要件定義書 5-2）。"""
    s = user_settings.load()
    mult = costmodel.spread_multipliers(dev.index, tf_key, s.costs, s.event_filter)
    rep = bt.evaluate_all(dev[["open", "high", "low", "close"]], P, RISK, exit_rules(tf_key), costs, pip,
                          runs=VERDICT_RANDOM_RUNS, holdout_start=b.start, rw_paths=VERDICT_RW_PATHS, spread_mult=mult)
    tracking.set_state(con, pair, tf_key, "verdict_cost_model", VERDICT_STATE)
    trials.record(con, rep.trials, pair=pair, tf=tf_key, origin="job", now=now,
                  data_from=dev.index[0], data_to=dev.index[-1], n_bars=len(dev))
    rep = bt.finalize(rep, n_trials)  # この処理で試す予定の分を含めた、すべての試行の数で DSR を割り引く
    con.executemany(
        """INSERT OR REPLACE INTO backtest_verdicts
           (pair, tf, source, computed_at, bar_time, verdict, reason, wf_total, wf_beat, dsr, pbo, n_trials, wf_n,
            wf_win_rate, wf_payoff)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(pair, tf_key, r.strategy.key, ts_str(now), ts_str(dev.index[-1]), r.verdict[0], r.verdict[1],
          r.wf.test_metrics["total"], r.wf_beat, _num(r.dsr.get("dsr")), _num(r.pbo.get("pbo")), rep.n_trials,
          r.wf.test_metrics["n"], _num(r.wf.test_metrics["win_rate"]), _num(r.wf.test_metrics["payoff"]))
         for r in rep.reports],
    )


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ---------------- 答え合わせ ----------------

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
    """研究ノート反映前の方向予測（prediction_log）の答え合わせ。新しい予測はもう記録しない。"""
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
