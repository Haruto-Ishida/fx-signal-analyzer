"""シグナル・予測の記録、戦略の重み、成績悪化の警告（フェーズ4）。"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest as bt
from . import strategies
from .config import IndicatorParams, RiskParams

SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, tf TEXT NOT NULL, source TEXT NOT NULL,
    signal_time TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry REAL, sl REAL, tp REAL,     -- 次の足の始値がわかるまでは NULL（status = '待機'）
    atr REAL NOT NULL,
    max_bars INTEGER NOT NULL,
    prob REAL, weight REAL,
    logged_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '待機',
    exit_time TEXT, exit_price REAL, pips REAL,
    entry_basis TEXT NOT NULL DEFAULT 'next_open',  -- next_open / signal_close（研究ノート反映前の記録）
    UNIQUE (pair, tf, source, signal_time)
);
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    pair TEXT NOT NULL, tf TEXT NOT NULL,
    family TEXT NOT NULL,          -- strategy / ml
    name TEXT NOT NULL,            -- 戦略のキー、またはモデルの種類
    purpose TEXT NOT NULL,         -- full_period / wf_candidate / ml_eval など
    config TEXT NOT NULL,          -- 設定（JSON）
    config_hash TEXT NOT NULL,     -- 同じ設定かどうかの判定用
    data_from TEXT, data_to TEXT, n_bars INTEGER,
    metrics TEXT,                  -- 結果（JSON）
    origin TEXT NOT NULL,          -- dashboard / job / retroactive など
    note TEXT
);
CREATE TABLE IF NOT EXISTS final_test_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, tf TEXT NOT NULL, accessed_at TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_log (
    pair TEXT NOT NULL, tf TEXT NOT NULL, model TEXT NOT NULL, horizon INTEGER NOT NULL,
    bar_time TEXT NOT NULL,
    p REAL NOT NULL, decision TEXT NOT NULL, version TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    y REAL,
    PRIMARY KEY (pair, tf, model, horizon, bar_time)
);
CREATE TABLE IF NOT EXISTS model_registry (
    version TEXT PRIMARY KEY,
    pair TEXT NOT NULL, tf TEXT NOT NULL, model TEXT NOT NULL, horizon INTEGER NOT NULL,
    trained_at TEXT NOT NULL,
    trained_bar TEXT NOT NULL,       -- 学習した時点の最新の確定足（再学習の間隔の基準）
    trained_through TEXT NOT NULL,   -- 学習に使った最後の足（これより後の足の予測は学習データ外）
    n_train INTEGER NOT NULL,
    acc REAL, majority_acc REAL, logloss REAL, base_logloss REAL, trade_pips REAL, beat REAL,
    ok INTEGER NOT NULL,
    path TEXT NOT NULL,
    eval_from TEXT, eval_to TEXT,    -- 条件の判定（検証）に使った期間：開発用データのみ
    train_from TEXT                  -- 本番モデルの学習に使った期間の始め（終わりは trained_through）
);
CREATE TABLE IF NOT EXISTS weight_history (
    pair TEXT NOT NULL, tf TEXT NOT NULL, bar_time TEXT NOT NULL, source TEXT NOT NULL,
    weight REAL NOT NULL, recent_n INTEGER, recent_pips REAL, recent_pf REAL, z REAL,
    PRIMARY KEY (pair, tf, bar_time, source)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, tf TEXT NOT NULL, source TEXT NOT NULL, kind TEXT NOT NULL,
    level TEXT NOT NULL, message TEXT NOT NULL,
    created_at TEXT NOT NULL, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, tf TEXT NOT NULL, origin TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, summary TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS backtest_verdicts (
    pair TEXT NOT NULL, tf TEXT NOT NULL, source TEXT NOT NULL,
    computed_at TEXT NOT NULL, bar_time TEXT NOT NULL,
    verdict TEXT NOT NULL, reason TEXT NOT NULL, wf_total REAL, wf_beat REAL,
    dsr REAL, pbo REAL, n_trials INTEGER, wf_n INTEGER,
    PRIMARY KEY (pair, tf, source)
);
CREATE TABLE IF NOT EXISTS ga_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, tf TEXT NOT NULL,
    kind TEXT NOT NULL,              -- full（本格的な探索）/ neighborhood（週末の再探索）
    seed INTEGER NOT NULL, budget INTEGER NOT NULL, n_evaluated INTEGER NOT NULL,
    started_at TEXT NOT NULL, data_from TEXT, data_to TEXT, origin TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ga_candidates (
    run_id INTEGER NOT NULL, rank INTEGER NOT NULL,
    key TEXT NOT NULL, genes TEXT NOT NULL, rule TEXT NOT NULL,
    fitness REAL, trades INTEGER, total_pips REAL, segment_sharpe TEXT,
    PRIMARY KEY (run_id, rank)
);
CREATE TABLE IF NOT EXISTS challengers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL, tf TEXT NOT NULL,
    spec TEXT NOT NULL,              -- チャレンジャーの設定の名前（mlops.SPECS）
    kind TEXT NOT NULL,              -- モデルの種類（登録したときの本番と同じ）
    status TEXT NOT NULL,            -- shadow（シャドーモード）/ ready（入れ替えの提案中）/ approved / rejected
    created_at TEXT NOT NULL, decided_at TEXT, note TEXT, last_check TEXT
);
CREATE TABLE IF NOT EXISTS shadow_judgments (
    challenger_id INTEGER NOT NULL, signal_id INTEGER NOT NULL,
    p REAL NOT NULL, decision TEXT NOT NULL, version TEXT NOT NULL, judged_at TEXT NOT NULL,
    PRIMARY KEY (challenger_id, signal_id)
);
CREATE TABLE IF NOT EXISTS log_state (
    pair TEXT NOT NULL, tf TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (pair, tf, key)
);
"""

TS_FMT = "%Y-%m-%d %H:%M:%S"
JST = pd.Timedelta(hours=9)


def ts_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime(TS_FMT)


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def migrate(con: sqlite3.Connection) -> None:
    """古い形式の DB を新しい形式にそろえる（何度呼んでもよい）。"""
    cols = _columns(con, "signal_log")
    if "entry_basis" not in cols:
        # 研究ノート反映前の記録：エントリーはシグナルの足の終値。entry 等を NULL 可にするため作り直す
        con.execute("ALTER TABLE signal_log RENAME TO signal_log_old")
        con.executescript(SCHEMA)
        old_cols = ", ".join(c for c in [
            "id", "pair", "tf", "source", "signal_time", "direction", "entry", "sl", "tp", "atr", "max_bars",
            "prob", "weight", "logged_at", "status", "exit_time", "exit_price", "pips"])
        con.execute(f"INSERT INTO signal_log ({old_cols}, entry_basis) SELECT {old_cols}, 'signal_close' FROM signal_log_old")
        con.execute("DROP TABLE signal_log_old")
    reg_cols = _columns(con, "model_registry")
    for c in ("eval_from", "eval_to", "train_from"):
        if c not in reg_cols:
            con.execute(f"ALTER TABLE model_registry ADD COLUMN {c} TEXT")
    ver_cols = _columns(con, "backtest_verdicts")
    for c, typ in (("dsr", "REAL"), ("pbo", "REAL"), ("n_trials", "INTEGER"), ("wf_n", "INTEGER"),
                   ("wf_win_rate", "REAL"), ("wf_payoff", "REAL")):
        if c not in ver_cols:
            con.execute(f"ALTER TABLE backtest_verdicts ADD COLUMN {c} {typ}")
    # フェーズ4改修：メタラベリングの判定・最大損失による停止を記録に残す
    sl_cols = _columns(con, "signal_log")
    for c, typ in (("meta_p", "REAL"), ("meta_decision", "TEXT"), ("meta_version", "TEXT"),
                   ("suppressed", "INTEGER NOT NULL DEFAULT 0")):
        if c not in sl_cols:
            con.execute(f"ALTER TABLE signal_log ADD COLUMN {c} {typ}")
    reg_cols = _columns(con, "model_registry")
    for c, typ in (("kind", "TEXT"), ("mode", "TEXT"), ("threshold", "REAL"), ("n_trials", "INTEGER")):
        if c not in reg_cols:
            con.execute(f"ALTER TABLE model_registry ADD COLUMN {c} {typ}")
    # Step 8：モデルの特徴量の版（NULL ＝ v2）
    if "features" not in _columns(con, "model_registry"):
        con.execute("ALTER TABLE model_registry ADD COLUMN features TEXT")
    # Step 7：シグナルを止めた理由（最大損失・キルスイッチ・経済指標）
    if "suppress_reason" not in _columns(con, "signal_log"):
        con.execute("ALTER TABLE signal_log ADD COLUMN suppress_reason TEXT")
    con.executescript("""CREATE TABLE IF NOT EXISTS kill_switch_releases (
        key TEXT PRIMARY KEY, released_at TEXT NOT NULL, note TEXT);""")
    # Step 6：GA の候補の検証結果
    ga_cols = _columns(con, "ga_candidates")
    for c, typ in (("verdict", "TEXT"), ("reason", "TEXT"), ("wf_total", "REAL"), ("wf_n", "INTEGER"),
                   ("wf_beat", "REAL"), ("rw_beat", "REAL"), ("dsr", "REAL"), ("pbo", "REAL"), ("n_trials", "INTEGER"),
                   ("yearly", "TEXT"), ("validated_at", "TEXT")):
        if c not in ga_cols:
            con.execute(f"ALTER TABLE ga_candidates ADD COLUMN {c} {typ}")
    # 重みの計算元：以前のものは直近のバックテスト（＝最終テスト期間）から計算していた
    if "basis" not in _columns(con, "weight_history"):
        con.execute("ALTER TABLE weight_history ADD COLUMN basis TEXT NOT NULL DEFAULT 'backtest_recent'")
    from .trials import seed_retroactive
    seed_retroactive(con)
    con.commit()


def get_state(con: sqlite3.Connection, pair: str, tf: str, key: str) -> str | None:
    row = con.execute("SELECT value FROM log_state WHERE pair=? AND tf=? AND key=?", (pair, tf, key)).fetchone()
    return row[0] if row else None


def set_state(con: sqlite3.Connection, pair: str, tf: str, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO log_state VALUES (?, ?, ?, ?)", (pair, tf, key, value))


# ---------------- 重み ----------------

@dataclass
class RecentPerf:
    source: str
    n: int
    total: float
    pf: float
    z: float
    weight: float = 1.0


def z_score(pips: np.ndarray) -> float:
    """1取引あたり平均が 0 から標準誤差いくつ分離れているか。取引が少ないときは 0。"""
    if len(pips) < 5:
        return 0.0
    sd = pips.std(ddof=1)
    return 0.0 if sd == 0 else float(pips.mean() / (sd / math.sqrt(len(pips))))


def weight_from_z(z: float) -> float:
    """控えめな重み：z ±1 ごとに ±0.25。0.25〜1.75 の範囲に収める。"""
    return float(np.clip(1 + 0.25 * np.clip(z, -3, 3), 0.25, 1.75))


def recent_performance(
    ind: pd.DataFrame, p: IndicatorParams, risk: RiskParams, costs: bt.Costs, pip: float, recent_bars: int,
) -> list[RecentPerf]:
    """直近 recent_bars 本で各戦略を売買した成績（フェーズ2と同じエンジン）と、そこから決めた重み。"""
    bars = bt.Bars(ind)
    n = len(ind)
    start = max(bt.WARMUP, n - recent_bars)
    out = []
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, p).tolist()
        t = bt.simulate(bars, sig, risk, bt.ExitRules(), costs, pip, start, n)
        m = bt.metrics(t["pips"])
        z = z_score(t["pips"].to_numpy())
        out.append(RecentPerf(s.key, m["n"], m["total"], m["pf"], z, weight_from_z(z)))
    mean_w = float(np.mean([r.weight for r in out]))
    for r in out:
        r.weight /= mean_w  # 平均が 1 になるようにそろえる
    return out


def weighted_score(decisions: dict[str, str], weights: dict[str, float]) -> float:
    """各戦略の判断（買い=+1・売り=−1・様子見=0）の重み付き平均。−1〜+1。"""
    num = sum(weights.get(k, 1.0) * {"買い": 1, "売り": -1}.get(d, 0) for k, d in decisions.items())
    den = sum(weights.get(k, 1.0) for k in decisions)
    return num / den if den else 0.0


def score_label(score: float) -> str:
    if score >= 0.4:
        return "買い寄り"
    if score <= -0.4:
        return "売り寄り"
    if score >= 0.15:
        return "やや買い寄り"
    if score <= -0.15:
        return "やや売り寄り"
    return "中立"


def latest_weights(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    """本番の記録から計算した最新の重み（最終テスト期間のバックテストから計算していた以前の重みは使わない）。"""
    return pd.read_sql_query(
        """SELECT * FROM weight_history WHERE pair=? AND tf=? AND basis='live' AND bar_time =
           (SELECT MAX(bar_time) FROM weight_history WHERE pair=? AND tf=? AND basis='live')""",
        con, params=(pair, tf, pair, tf),
    )


DONE = ("利確", "損切り", "期限")


def live_performance(closed: pd.DataFrame, window: int, min_trades: int) -> list[RecentPerf]:
    """本番の記録（決済済み）から、戦略ごとの直近 window 回の成績と重みを計算する。
    取引が min_trades 回未満の戦略は z = 0（重み 1）。重みは平均 1 にそろえる。"""
    out = []
    for s in strategies.STRATEGIES:
        g = closed[closed["source"] == s.key].sort_values("exit_time").tail(window)
        p = g["pips"].to_numpy(dtype=float)
        m = bt.metrics(p)
        z = z_score(p) if len(p) >= min_trades else 0.0
        out.append(RecentPerf(s.key, m["n"], m["total"], m["pf"], z, weight_from_z(z)))
    mean_w = float(np.mean([r.weight for r in out]))
    for r in out:
        r.weight /= mean_w
    return out


@dataclass
class Halt:
    active: bool
    reason: str
    day_r: float      # 日本時間の今日の損益（R）
    week_r: float     # 日本時間の今週（月曜から）の損益（R）


def r_multiple(closed: pd.DataFrame, sl_atr: float, pip: float) -> pd.Series:
    """損益を「1回の損切り幅（1R）」の何倍かで表す。1R は口座残高の許容損失（初期値 1%）に相当する。"""
    return closed["pips"] / (sl_atr * closed["atr"] / pip)


def loss_halt(closed: pd.DataFrame, now: pd.Timestamp, sl_atr: float, pip: float, daily_max: float,
              weekly_max: float) -> Halt:
    """日本時間のその日・その週に決済した本番の記録の損益が上限に達したら、新しいシグナルを止める。"""
    if closed.empty:
        return Halt(False, "", 0.0, 0.0)
    jst_now = now + JST
    today = jst_now.normalize()
    week_start = today - pd.Timedelta(days=today.dayofweek)
    exit_jst = pd.DatetimeIndex(closed["exit_time"]) + JST
    r = r_multiple(closed, sl_atr, pip).to_numpy()
    day_r = float(r[exit_jst >= today].sum())
    week_r = float(r[exit_jst >= week_start].sum())
    eps = 1e-9  # 小数の計算の誤差（例：−1.9999999999999998）で上限の判定がずれないように
    if day_r <= -daily_max + eps:
        return Halt(True, f"今日（日本時間）の損失が {day_r:+.1f}R に達したため、今日は新しいシグナルを出しません"
                          f"（上限 −{daily_max:g}R ＝ 口座残高の約{daily_max:g}%）", day_r, week_r)
    if week_r <= -weekly_max + eps:
        return Halt(True, f"今週（日本時間の月曜から）の損失が {week_r:+.1f}R に達したため、今週は新しいシグナルを出しません"
                          f"（上限 −{weekly_max:g}R ＝ 口座残高の約{weekly_max:g}%）", day_r, week_r)
    return Halt(False, "", day_r, week_r)


def meta_conditions(registry: pd.DataFrame, signals: pd.DataFrame) -> list[Condition]:
    """メタラベリングの本番モデルについての警告。registry はメタのモデルの履歴（古い順）、signals は本番の記録。"""
    out = []
    if registry.empty:
        return out
    latest = registry.iloc[-1]
    if not latest["ok"] and registry["ok"].iloc[:-1].astype(bool).any():
        out.append(Condition("ml_meta", "lost_edge", "warning",
                             "機械学習（メタラベリング）：再学習の結果、シグナルを出す条件を満たさなくなりました（以前は満たしていました）"))
    done = signals[signals["status"].isin(DONE) & signals["meta_decision"].isin(["乗る", "見送り"])]
    taken, skipped = done[done["meta_decision"] == "乗る"], done[done["meta_decision"] == "見送り"]
    if len(done) >= 30 and len(taken) >= 10 and len(skipped) >= 10:
        a, b = taken["pips"].mean(), skipped["pips"].mean()
        if a < b:
            out.append(Condition("ml_meta", "filter_hurts", "warning",
                                 f"機械学習（メタラベリング）：本番の記録では、乗ったシグナルの1回あたり {a:+.1f} pips が、"
                                 f"見送ったシグナルの {b:+.1f} pips を下回っています（見送りが逆効果）"))
    return out


def monthly(closed: pd.DataFrame) -> pd.DataFrame:
    """決済した月（日本時間）× 戦略 の損益（pips）。"""
    if closed.empty:
        return pd.DataFrame()
    month = (pd.DatetimeIndex(closed["exit_time"]) + JST).strftime("%Y-%m")
    return closed.assign(month=month).pivot_table(index="month", columns="source", values="pips", aggfunc="sum")


# ---------------- 警告 ----------------

@dataclass
class Condition:
    source: str
    kind: str
    level: str      # "warning" / "critical"
    message: str


def sync_alerts(con: sqlite3.Connection, pair: str, tf: str, active: list[Condition], now: pd.Timestamp) -> int:
    """いま成り立っている警告を登録し、成り立たなくなった警告は解決済みにする。新しく出した件数を返す。"""
    open_rows = con.execute(
        "SELECT id, source, kind FROM alerts WHERE pair=? AND tf=? AND resolved_at IS NULL", (pair, tf)
    ).fetchall()
    open_keys = {(s, k): i for i, s, k in open_rows}
    active_keys = {(c.source, c.kind) for c in active}
    added = 0
    for c in active:
        if (c.source, c.kind) in open_keys:
            con.execute("UPDATE alerts SET message=?, level=? WHERE id=?", (c.message, c.level, open_keys[(c.source, c.kind)]))
        else:
            con.execute(
                "INSERT INTO alerts (pair, tf, source, kind, level, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pair, tf, c.source, c.kind, c.level, c.message, ts_str(now)),
            )
            added += 1
    for key, i in open_keys.items():
        if key not in active_keys:
            con.execute("UPDATE alerts SET resolved_at=? WHERE id=?", (ts_str(now), i))
    return added


def strategy_conditions(recent: list[RecentPerf], closed_logs: pd.DataFrame, period_label: str) -> list[Condition]:
    names = {s.key: s.name for s in strategies.STRATEGIES}
    out = []
    for r in recent:
        name = names[r.source]
        if r.n >= 10 and r.z <= -2:
            out.append(Condition(r.source, "recent_bad", "critical",
                                 f"{name}：{period_label}の成績が統計的にはっきりマイナスです（{r.total:+,.0f} pips、{r.n}回、PF {r.pf:.2f}）"))
        elif r.n >= 10 and r.pf < 0.8:
            out.append(Condition(r.source, "recent_bad", "warning",
                                 f"{name}：{period_label}の成績が悪化しています（{r.total:+,.0f} pips、{r.n}回、PF {r.pf:.2f}）"))
    if not closed_logs.empty:
        for source, g in closed_logs.groupby("source"):
            name = names.get(source, source)
            last = g.sort_values("exit_time").tail(20)
            m = bt.metrics(last["pips"])
            if m["n"] >= 10 and m["pf"] < 0.8:
                out.append(Condition(source, "live_bad", "warning",
                                     f"{name}：記録したシグナルの直近{m['n']}回の成績が悪化しています（{m['total']:+,.0f} pips、PF {m['pf']:.2f}）"))
            streak = 0
            for v in g.sort_values("exit_time")["pips"].to_numpy()[::-1]:
                if v >= 0:
                    break
                streak += 1
            if streak >= 6:
                out.append(Condition(source, "losing_streak", "warning", f"{name}：記録したシグナルが {streak} 連敗中です"))
    return out


def model_conditions(registry: pd.DataFrame, preds: pd.DataFrame) -> list[Condition]:
    """registry は1モデル分の履歴（古い順）、preds はそのモデルの答え合わせ済みの予測。"""
    out = []
    if registry.empty:
        return out
    latest = registry.iloc[-1]
    source = f"ml_{latest['model']}"
    label = MODEL_NAMES.get(latest["model"], latest["model"])
    if not latest["ok"] and registry["ok"].iloc[:-1].astype(bool).any():
        out.append(Condition(source, "lost_edge", "warning",
                             f"{label}：再学習の結果、予測力の条件を満たさなくなりました（以前は満たしていました）"))
    recent = preds.dropna(subset=["y"]).sort_values("bar_time").tail(200)
    n = len(recent)
    if n >= 50:
        acc = float(((recent["p"] >= 0.5) == (recent["y"] == 1)).mean())
        h = int(latest["horizon"])
        se = math.sqrt(acc * (1 - acc) / (n / h))
        if acc + 1.96 * se < 0.5:
            out.append(Condition(source, "live_acc", "critical",
                                 f"{label}：実際の予測の正解率が {acc:.1%}（直近{n}件）で、50%をはっきり下回っています"))
        elif pd.notna(latest["acc"]) and acc < latest["acc"] - 0.05:
            out.append(Condition(source, "live_acc", "warning",
                                 f"{label}：実際の予測の正解率 {acc:.1%}（直近{n}件）が、検証時の {latest['acc']:.1%} より大きく低くなっています"))
    return out


MODEL_NAMES = {"logistic": "ロジスティック回帰", "gbm": "勾配ブースティング"}


def data_condition(last_bar: pd.Timestamp, now: pd.Timestamp, bar: pd.Timedelta, stale_bars: int) -> list[Condition]:
    """平日（土日を除く）に、最新の足が古すぎる場合に警告する。"""
    if now.dayofweek >= 5 or (now.dayofweek == 0 and now.hour < 6):  # 週末と週明け直後は休場
        return []
    if now - (last_bar + bar) > bar * stale_bars:
        return [Condition("data", "stale", "warning",
                          f"データが更新されていません（最新の確定足：{last_bar:%Y-%m-%d %H:%M} UTC）。接続や取得元を確認してください")]
    return []


# ---------------- 読み出し ----------------

def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """数値の列を必ず数値型にする。列がすべて NULL（例：決済前のシグナルだけ）だと、
    pandas は型のわからない列（object）として読み込み、四捨五入などが失敗するため。"""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def load_signal_log(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM signal_log WHERE pair=? AND tf=? ORDER BY signal_time", con, params=(pair, tf),
        parse_dates=["signal_time", "exit_time", "logged_at"],
    )
    return _numeric(df, ["entry", "sl", "tp", "atr", "prob", "weight", "exit_price", "pips", "meta_p"])


def load_predictions(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM prediction_log WHERE pair=? AND tf=? ORDER BY bar_time", con, params=(pair, tf),
        parse_dates=["bar_time", "logged_at"],
    )
    return _numeric(df, ["p", "y"])


def load_registry(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM model_registry WHERE pair=? AND tf=? ORDER BY trained_at", con, params=(pair, tf),
        parse_dates=["trained_at", "trained_bar", "trained_through", "eval_from", "eval_to", "train_from"],
    )
    return _numeric(df, ["acc", "majority_acc", "logloss", "base_logloss", "trade_pips", "beat", "threshold", "n_trials"])


def load_alerts(con: sqlite3.Connection, pair: str | None = None, tf: str | None = None,
                active_only: bool = False) -> pd.DataFrame:
    q = "SELECT * FROM alerts WHERE 1=1"
    params: list = []
    if pair:
        q += " AND pair=?"
        params.append(pair)
    if tf:
        q += " AND tf=?"
        params.append(tf)
    if active_only:
        q += " AND resolved_at IS NULL"
    return pd.read_sql_query(q + " ORDER BY created_at DESC", con, params=params,
                             parse_dates=["created_at", "resolved_at"])


def load_weight_history(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM weight_history WHERE pair=? AND tf=? ORDER BY bar_time", con, params=(pair, tf),
        parse_dates=["bar_time"],
    )
    return _numeric(df, ["weight", "recent_pips", "recent_pf", "z"])


def load_verdicts(con: sqlite3.Connection, pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM backtest_verdicts WHERE pair=? AND tf=?", con, params=(pair, tf),
                           parse_dates=["computed_at", "bar_time"])
    return _numeric(df, ["wf_total", "wf_beat", "dsr", "pbo", "n_trials", "wf_n", "wf_win_rate", "wf_payoff"])


@dataclass
class MLStatus:
    model: str
    name: str
    horizon: int
    p: float
    decision: str
    ok: bool
    bar_time: pd.Timestamp


def latest_ml_status(con: sqlite3.Connection, pair: str, tf: str) -> list[MLStatus]:
    """各モデルの最新の予測と、そのとき使ったモデルが条件を満たしていたか。"""
    rows = con.execute(
        """SELECT p.model, p.horizon, p.p, p.decision, r.ok, p.bar_time FROM prediction_log p
           JOIN model_registry r ON r.version = p.version
           WHERE p.pair=? AND p.tf=? AND p.bar_time = (
               SELECT MAX(bar_time) FROM prediction_log q
               WHERE q.pair=p.pair AND q.tf=p.tf AND q.model=p.model AND q.horizon=p.horizon)
           ORDER BY p.model""", (pair, tf),
    ).fetchall()
    return [MLStatus(m, MODEL_NAMES.get(m, m), int(h), float(pv), d, bool(ok), pd.Timestamp(t))
            for m, h, pv, d, ok, t in rows]


def load_job_runs(con: sqlite3.Connection, limit: int = 50) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM job_runs ORDER BY id DESC LIMIT {int(limit)}", con,
                             parse_dates=["started_at", "finished_at"])
