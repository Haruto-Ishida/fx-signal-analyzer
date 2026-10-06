"""MLOps：モデルのバージョン管理・巻き戻し・チャンピオン vs チャレンジャー（要件定義書 9章・ユーザーの決定 2）。

ユーザーの決定
- 設定が同じでデータが増えただけの再学習は、今まで通り自動で切り替える。ただし毎回バージョンを記録し（model_registry と
  models/ のファイル）、画面から前のバージョンに戻せるようにする（戻すと、そのバージョンに固定し、自動の再学習は止める）。
- モデルの種類・設定・特徴量・戦略が変わるものは「チャレンジャー」として、シャドーモード（裏で判定だけ記録し、
  シグナルには使わない）で4週間以上、本番（チャンピオン）と並べて成績を記録し、条件を満たしたらユーザーに提案する。
  入れ替えはユーザーの承認後だけ。入れ替えは月1回まで。

入れ替えの提案の条件（すべて満たすこと）
- シャドーモードを始めてから28日以上
- 決済したシグナルが30件以上、チャンピオン・チャレンジャーとも「乗る」が10件以上（足りなければ延長）
- 同じシグナルに対する「乗る／見送り」の判断で、チャレンジャーのシャープレシオ（1シグナルあたり）がチャンピオンより高く、
  その差がブートストラップ検定で偶然とは言いにくい（p < 0.05）
- 最大ドローダウンが、チャンピオンの1.2倍（＋10 pips）より悪くない
- 開発用データでの検証（6-3 の合格基準にあたる機械学習の条件）を満たす
- 前回の入れ替えから30日以上
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import holdout, ml, ml_eval, tracking, trials
from .config import ML_META_FOLDS, ML_META_THRESHOLD, MODELS_DIR, RiskParams, ROOT, TRACKING, VERTICAL_BARRIER
from .tracking import ts_str

SHADOW_DAYS = 28
MIN_SIGNALS = 30
MIN_TAKEN = 10
P_VALUE = 0.05
DD_TOLERANCE = 1.2
DD_SLACK_PIPS = 10.0
SWAP_INTERVAL_DAYS = 30
BOOTSTRAP = 2000


@dataclass(frozen=True)
class MetaSpec:
    name: str
    label: str
    description: str
    fracdiff: bool = False

    @property
    def features(self) -> str:
        return "v2+fd" if self.fracdiff else "v2"


SPECS = {
    "base": MetaSpec("base", "これまでの特徴量", "これまでの本番と同じ特徴量（テクニカル指標・直近の値動き・時間帯など）"),
    "fracdiff": MetaSpec("fracdiff", "分数階差分を追加",
                         "これまでの特徴量に、分数階差分（過去の水準の情報を残した値動き、d=0.4）を1つ加える", fracdiff=True),
}


# ---------------- 本番（チャンピオン）の設定とバージョン ----------------

def current_spec(con, pair, tf) -> MetaSpec:
    return SPECS.get(tracking.get_state(con, pair, tf, "meta_spec") or "base", SPECS["base"])


def pinned_version(con, pair, tf) -> str | None:
    return tracking.get_state(con, pair, tf, "meta_pin") or None


def _registry_row(con, sql: str, params) -> dict | None:
    cur = con.execute(sql, params)
    row = cur.fetchone()
    return dict(zip([c[0] for c in cur.description], row)) if row else None


def active_meta(con, pair, tf) -> dict | None:
    """本番で使うモデル。前のバージョンに戻して固定していればそれ、なければ最新。"""
    pin = pinned_version(con, pair, tf)
    if pin:
        row = _registry_row(con, "SELECT * FROM model_registry WHERE version=? AND model='meta'", (pin,))
        if row is not None:
            return row
    return _registry_row(con, "SELECT * FROM model_registry WHERE pair=? AND tf=? AND model='meta' "
                              "ORDER BY trained_at DESC LIMIT 1", (pair, tf))


def versions(con, pair, tf) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT version, trained_at, trained_through, n_train, kind, features, ok, trade_pips, beat, path
           FROM model_registry WHERE pair=? AND tf=? AND model='meta' ORDER BY trained_at DESC""",
        con, params=(pair, tf))


def rollback(con, pair, tf, version: str, now: pd.Timestamp) -> None:
    """そのバージョンに戻して固定する（自動の再学習は、固定を解除するまで止める）。"""
    row = _registry_row(con, "SELECT version, path FROM model_registry WHERE version=? AND pair=? AND tf=? AND model='meta'",
                        (version, pair, tf))
    if row is None:
        raise ValueError(f"バージョン {version} が見つかりません")
    if not (ROOT / row["path"]).exists():
        raise FileNotFoundError(f"モデルのファイル {row['path']} が見つかりません")
    tracking.set_state(con, pair, tf, "meta_pin", version)
    tracking.set_state(con, pair, tf, "meta_pin_at", ts_str(now))
    con.commit()


def unpin(con, pair, tf) -> None:
    tracking.set_state(con, pair, tf, "meta_pin", "")
    con.commit()


def model_features(reg: dict) -> bool:
    """そのモデルが分数階差分の特徴量を使うか。"""
    return (reg.get("features") or "v2") == "v2+fd"


# ---------------- チャレンジャー ----------------

def challengers(con, pair, tf) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM challengers WHERE pair=? AND tf=? ORDER BY id DESC", con, params=(pair, tf))


def register(con, pair, tf, spec_name: str, now: pd.Timestamp, note: str = "") -> int:
    """チャレンジャーをシャドーモードで始める。モデルの種類は、いまの本番と同じにする（比べるのは特徴量だけ）。
    試行として1つ記録する（本番と違う設定を試すことになるため）。"""
    spec = SPECS[spec_name]
    champ = active_meta(con, pair, tf)
    if champ is None:
        raise ValueError("本番のモデルがまだないため、比べる相手がいません")
    if spec.features == (champ.get("features") or "v2"):
        raise ValueError("いまの本番と同じ設定です")
    if not challengers(con, pair, tf).query("status in ('shadow', 'ready') and spec == @spec_name").empty:
        raise ValueError("同じチャレンジャーがすでにシャドーモード中です")
    kind = champ["kind"]
    cid = con.execute("""INSERT INTO challengers (pair, tf, spec, kind, status, created_at, note)
                         VALUES (?, ?, ?, ?, 'shadow', ?, ?)""", (pair, tf, spec_name, kind, ts_str(now), note)).lastrowid
    plan = ml_eval.planned_trials("meta", ML_META_THRESHOLD, VERTICAL_BARRIER[tf], ML_META_FOLDS, spec.features, (kind,))
    # 期間は、入れ替えの前に検証する開発用データの期間（dev_check と同じ試行として数える）
    from . import data
    dev = data.load_dev(con, pair, tf)
    trials.record(con, [{**r, "purpose": "challenger_shadow"} for r in plan], pair=pair, tf=tf, origin="dashboard",
                  now=now, data_from=dev.index[0], data_to=dev.index[-1], n_bars=len(dev),
                  note=f"チャレンジャー {cid}（{spec.label}）をシャドーモードで開始")
    con.commit()
    return cid


def _challenger_model(con, cid: int) -> dict | None:
    return _registry_row(con, "SELECT * FROM model_registry WHERE model=? ORDER BY trained_at DESC LIMIT 1",
                         (f"challenger_{cid}",))


def shadow_cycle(con, pair, tf, ind: pd.DataFrame, risk: RiskParams, p, costs, pip, now: pd.Timestamp) -> int:
    """シャドーモードのチャレンジャーを（必要なら）再学習し、新しいシグナルを判定して記録する。シグナルには使わない。
    戻り値：判定したシグナルの数。"""
    ch = challengers(con, pair, tf)
    ch = ch[ch["status"].isin(["shadow", "ready"]) & (ch["kind"] != "strategy")]   # 機械学習のチャレンジャーだけ
    judged = 0
    every = TRACKING[tf].retrain_every
    for c in ch.itertuples():
        spec = SPECS[c.spec]
        reg = _challenger_model(con, c.id)
        due = reg is None or len(ind) - int(ind.index.searchsorted(pd.Timestamp(reg["trained_bar"]), side="right")) >= every
        if due:
            smp = ml_eval.build_meta(ind, tf, p, risk, VERTICAL_BARRIER[tf], costs.round_trip, pip, fracdiff=spec.fracdiff)
            model = ml_eval.fit_production(smp, c.kind)
            version = f"{pair.replace('/', '')}_{tf}_challenger{c.id}_{c.kind}_{now:%Y%m%d%H%M%S}"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"{version}.joblib"
            joblib.dump(model, path)
            reg = dict(version=version, pair=pair, tf=tf, model=f"challenger_{c.id}", horizon=VERTICAL_BARRIER[tf],
                       trained_at=ts_str(now), trained_bar=ts_str(ind.index[-1]),
                       trained_through=ts_str(ind.index[int(smp.t0.max())]), n_train=len(smp.y), ok=0,
                       path=path.relative_to(ROOT).as_posix(), kind=c.kind, mode="meta", threshold=ML_META_THRESHOLD,
                       features=spec.features, train_from=ts_str(ind.index[int(smp.t0.min())]))
            con.execute(f"INSERT OR REPLACE INTO model_registry ({', '.join(reg)}) VALUES ({', '.join('?' * len(reg))})",
                        tuple(reg.values()))
        judged += _judge_shadow(con, c.id, pair, tf, reg, ind, spec, now)
    con.commit()
    return judged


def _judge_shadow(con, cid, pair, tf, reg, ind, spec, now) -> int:
    """シャドーモードを始めた後、かつモデルを学習した時点の最新の足以降の、まだ判定していないシグナルを判定する。"""
    start = max(reg["trained_bar"], con.execute("SELECT created_at FROM challengers WHERE id=?", (cid,)).fetchone()[0])
    rows = con.execute(
        """SELECT id, source, signal_time, direction FROM signal_log s
           WHERE pair=? AND tf=? AND entry_basis='next_open' AND source NOT LIKE 'ml_%' AND shadow=0 AND signal_time >= ?
             AND NOT EXISTS (SELECT 1 FROM shadow_judgments j WHERE j.challenger_id=? AND j.signal_id=s.id)""",
        (pair, tf, start, cid)).fetchall()
    pos = {t: i for i, t in enumerate(ind.index)}
    rows = [(rid, src, pos[pd.Timestamp(t)], d) for rid, src, t, d in rows if pd.Timestamp(t) in pos]
    if not rows:
        return 0
    F = ml.make_features(ind, tf, fracdiff=spec.fracdiff)
    X = ml_eval.meta_features(F, np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                              np.array([r[1] for r in rows]))
    ok = ~X.isna().any(axis=1).to_numpy()
    if not ok.any():
        return 0
    model = joblib.load(ROOT / reg["path"])
    p = model.predict_proba(X.to_numpy()[ok])[:, 1]
    th = reg.get("threshold") or ML_META_THRESHOLD
    con.executemany(
        "INSERT OR IGNORE INTO shadow_judgments (challenger_id, signal_id, p, decision, version, judged_at) VALUES (?, ?, ?, ?, ?, ?)",
        [(cid, rid, float(pi), "乗る" if pi >= th else "見送り", reg["version"], ts_str(now))
         for (rid, *_), pi in zip([r for r, k in zip(rows, ok) if k], p)])
    return int(ok.sum())


# ---------------- 比較 ----------------

@dataclass
class Comparison:
    days: float
    n: int                    # チャンピオン・チャレンジャーの両方が判定し、決済したシグナルの数
    taken_champ: int
    taken_chal: int
    pips_champ: float
    pips_chal: float
    sr_champ: float           # 1シグナルあたりのシャープレシオ（見送りは 0 pips）
    sr_chal: float
    dd_champ: float           # 最大ドローダウン（pips、0以下）
    dd_chal: float
    p_value: float            # 「チャレンジャーの方が良い」が偶然である確率（ブートストラップ）
    checks: list[tuple[str, bool, str]]

    @property
    def ready(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def _sr(x: np.ndarray) -> float:
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / s) if s > 0 else 0.0


def _max_dd(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    eq = np.cumsum(x)
    return float(min(0.0, (eq - np.maximum.accumulate(np.maximum(eq, 0))).min()))


def bootstrap_p(diff: np.ndarray, n_boot: int = BOOTSTRAP, seed: int = 0) -> float:
    """差の平均が 0 以下になる割合（片側のブートストラップ検定）。"""
    if len(diff) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float((means <= 0).mean())


def compare(con, cid: int, now: pd.Timestamp, dev_ok: bool | None = None, last_swap: pd.Timestamp | None = None) -> Comparison:
    c = con.execute("SELECT created_at FROM challengers WHERE id=?", (cid,)).fetchone()
    days = (now - pd.Timestamp(c[0])).total_seconds() / 86400
    df = pd.read_sql_query(
        f"""SELECT s.exit_time, s.pips, s.meta_decision AS champ, j.decision AS chal
            FROM shadow_judgments j JOIN signal_log s ON s.id = j.signal_id
            WHERE j.challenger_id=? AND s.status IN ({",".join("?" * len(tracking.DONE))})
              AND s.meta_decision IN ('乗る', '見送り') AND s.pips IS NOT NULL
            ORDER BY s.exit_time""", con, params=(cid, *tracking.DONE))
    pips = df["pips"].to_numpy(dtype=float)
    a = np.where(df["champ"] == "乗る", pips, 0.0)
    b = np.where(df["chal"] == "乗る", pips, 0.0)
    ta, tb = int((df["champ"] == "乗る").sum()), int((df["chal"] == "乗る").sum())
    sra, srb = _sr(a), _sr(b)
    dda, ddb = _max_dd(a), _max_dd(b)
    p = bootstrap_p(b - a)
    checks = [
        ("シャドーモードの期間", days >= SHADOW_DAYS, f"{days:.0f}日（{SHADOW_DAYS}日以上）"),
        ("決済したシグナルの数", len(df) >= MIN_SIGNALS and min(ta, tb) >= MIN_TAKEN,
         f"{len(df)}件（{MIN_SIGNALS}件以上）・乗った数 本番 {ta}件／チャレンジャー {tb}件（各{MIN_TAKEN}件以上）"),
        ("シャープレシオ", srb > sra, f"チャレンジャー {srb:+.2f} ／ 本番 {sra:+.2f}"),
        ("偶然ではないか（ブートストラップ）", p < P_VALUE, f"p = {p:.2f}（{P_VALUE}未満）"),
        ("最大ドローダウン", -ddb <= DD_TOLERANCE * -dda + DD_SLACK_PIPS,
         f"チャレンジャー {ddb:,.0f} ／ 本番 {dda:,.0f} pips（本番の{DD_TOLERANCE}倍＋{DD_SLACK_PIPS:g} pips まで）"),
    ]
    if dev_ok is not None:
        checks.append(("開発用データでの検証", bool(dev_ok), "機械学習の条件（DSR など）を満たす" if dev_ok else "条件を満たさない"))
    if last_swap is not None:
        gap = (now - last_swap).days
        checks.append(("前回の入れ替えから", gap >= SWAP_INTERVAL_DAYS, f"{gap}日（{SWAP_INTERVAL_DAYS}日以上）"))
    return Comparison(days, len(df), ta, tb, float(a.sum()), float(b.sum()), sra, srb, dda, ddb, p, checks)


def dev_check(con, pair, tf, cid: int, ind_dev: pd.DataFrame, b, p, risk, costs, pip, now, n_trials: int) -> bool:
    """チャレンジャーの設定を、開発用データで本番と同じ方法で検証する（同じ設定なので試行は増えない）。"""
    holdout.assert_dev_only(ind_dev, b)
    row = con.execute("SELECT spec, kind FROM challengers WHERE id=?", (cid,)).fetchone()
    spec = SPECS[row[0]]
    smp = ml_eval.build_meta(ind_dev, tf, p, risk, VERTICAL_BARRIER[tf], costs.round_trip, pip, fracdiff=spec.fracdiff)
    rep = ml_eval.evaluate_all(smp, ML_META_THRESHOLD, ML_META_FOLDS, runs=200, kinds=(row[1],))
    trials.record(con, rep.trials, pair=pair, tf=tf, origin="job", now=now, data_from=ind_dev.index[0],
                  data_to=ind_dev.index[-1], n_bars=len(ind_dev), note=f"チャレンジャー {cid} の開発用データでの検証")
    rep = ml_eval.finalize(rep, smp, n_trials)
    return bool(rep.results[row[1]].ok)


def last_swap(con, pair, tf) -> pd.Timestamp | None:
    v = con.execute("SELECT MAX(decided_at) FROM challengers WHERE pair=? AND tf=? AND status='approved'",
                    (pair, tf)).fetchone()[0]
    return pd.Timestamp(v) if v else None


def update_status(con, pair, tf, cid: int, cmp: Comparison, now: pd.Timestamp) -> bool:
    """条件をすべて満たしたら「提案中（ready）」にする。満たさなくなったら shadow に戻す。戻り値：新しく提案中になったか。"""
    status = con.execute("SELECT status FROM challengers WHERE id=?", (cid,)).fetchone()[0]
    new = "ready" if cmp.ready else "shadow"
    con.execute("UPDATE challengers SET status=?, last_check=? WHERE id=? AND status IN ('shadow', 'ready')",
                (new, ts_str(now), cid))
    con.commit()
    return status == "shadow" and new == "ready"


def approve(con, pair, tf, cid: int, now: pd.Timestamp) -> None:
    """ユーザーの承認：本番の設定をチャレンジャーの設定に切り替える。次の定期処理で、新しい設定の本番モデルを学習する。
    それまでは今の本番モデルを使う。前のバージョンには画面から戻せる。"""
    row = con.execute("SELECT spec, status, kind FROM challengers WHERE id=? AND pair=? AND tf=?", (cid, pair, tf)).fetchone()
    if row is None or row[1] != "ready":
        raise ValueError("入れ替えの提案中のチャレンジャーではありません")
    if row[2] == "strategy":
        # 戦略の追加は、本番の画面・口座・機械学習の特徴量など広い範囲に関わるので、承認の記録だけ残し、
        # 本番への組み込みは、あなたと確認しながら行う（自動では組み込まない）
        con.execute("UPDATE challengers SET status='approved', decided_at=?, note=? WHERE id=?",
                    (ts_str(now), "承認済み。本番への組み込みは Claude と確認しながら行う", cid))
        con.commit()
        return
    tracking.set_state(con, pair, tf, "meta_spec", row[0])
    tracking.set_state(con, pair, tf, "meta_pin", "")
    con.execute("UPDATE challengers SET status='approved', decided_at=? WHERE id=?", (ts_str(now), cid))
    con.commit()


def reject(con, cid: int, now: pd.Timestamp, note: str = "ユーザーが見送り") -> None:
    con.execute("UPDATE challengers SET status='rejected', decided_at=?, note=? WHERE id=?", (ts_str(now), note, cid))
    con.commit()


def review(con, pair, tf, ind_dev, b, p, risk, costs, pip, now: pd.Timestamp, n_trials: int) -> list[str]:
    """シャドーモードのチャレンジャーの成績を本番と比べ、条件を満たしたら提案中にする。戻り値：新しく提案中になったものの説明。
    開発用データでの検証は重いので、ほかの条件をすべて満たしたときだけ、1日1回まで行う。"""
    out = []
    ch = challengers(con, pair, tf)
    ch = ch[ch["kind"] != "strategy"]   # 戦略のチャレンジャーは review_strategies
    for c in ch[ch["status"].isin(["shadow", "ready"])].itertuples():
        ls = last_swap(con, pair, tf)
        cmp = compare(con, c.id, now, last_swap=ls)
        if cmp.ready:
            key = f"challenger_dev_{c.id}"   # 前回の開発用データでの検証（時刻|結果）
            prev = tracking.get_state(con, pair, tf, key)
            if prev and now - pd.Timestamp(prev.split("|")[0]) < pd.Timedelta(days=1):
                dev_ok = prev.split("|")[1] == "1"
            else:
                dev_ok = dev_check(con, pair, tf, c.id, ind_dev, b, p, risk, costs, pip, now, n_trials)
                tracking.set_state(con, pair, tf, key, f"{ts_str(now)}|{int(dev_ok)}")
            cmp = compare(con, c.id, now, dev_ok=dev_ok, last_swap=ls)
        if update_status(con, pair, tf, c.id, cmp, now):
            out.append(f"チャレンジャー「{SPECS[c.spec].label}」が、入れ替えの条件を満たしました（シャープレシオ "
                       f"{cmp.sr_chal:+.2f} ＞ 本番 {cmp.sr_champ:+.2f}）。「機械学習予測」タブで確認して、承認するか決めてください")
    return out


# ---------------- 戦略のチャレンジャー（2026-10-07） ----------------
# 全通りの評価などで見つかった戦略の候補を、本番には使わずに「出ていたはずのシグナル」として記録し（signal_log の shadow=1）、
# 本番と同じ方法で答え合わせをする。本番の画面・ペーパー口座・キルスイッチ・通知・重み・機械学習には使わない。

STRATEGY_BASELINE = "ma_cross"   # 比べる相手：最初からある「移動平均クロス」（いちばん近い形の本番の戦略）


def register_strategy(con, pair, tf, genes: dict, now: pd.Timestamp, note: str = "") -> int:
    """戦略の候補をシャドーモードで始める。全通りの評価で試した設定そのものなので、試行数は増えない。"""
    import json
    from . import ga_jobs
    genome = ga_jobs.genome_from_genes(genes)         # 作れない組み合わせなら ValueError
    from . import ga
    key = ga.key_of(genome)
    ch = challengers(con, pair, tf)
    if len(ch) and ((ch["spec"] == key) & ch["status"].isin(["shadow", "ready"])).any():
        raise ValueError("同じ戦略がすでにシャドーモード中です")
    cid = con.execute("""INSERT INTO challengers (pair, tf, spec, kind, status, created_at, note, genes)
                         VALUES (?, ?, ?, 'strategy', 'shadow', ?, ?, ?)""",
                      (pair, tf, key, ts_str(now), note or ga.describe(genome), json.dumps(genes, ensure_ascii=False))
                      ).lastrowid
    con.commit()
    return cid


def strategy_label(c) -> str:
    import json
    from . import ga, ga_jobs
    try:
        return "戦略：" + ga.describe(ga_jobs.genome_from_genes(json.loads(c.genes)))
    except Exception:
        return f"戦略 {c.spec}"


def shadow_strategies(con, pair, tf, ind: pd.DataFrame, p, now: pd.Timestamp) -> int:
    """シャドーモードの戦略のシグナルを、前回処理した足より後に確定した足の分だけ記録する（初回は最新の確定足だけ。
    過去にさかのぼって作らない）。エントリーは本番と同じく次の足の始値（fill_entries・resolve_signals が答え合わせをする）。"""
    import json
    from . import ga, ga_jobs
    ch = challengers(con, pair, tf)
    ch = ch[(ch["kind"] == "strategy") & ch["status"].isin(["shadow", "ready"])] if len(ch) else ch
    added = 0
    n = len(ind)
    atr = ind["atr"].to_numpy()
    max_bars = TRACKING[tf].log_max_bars
    for c in ch.itertuples():
        genome = ga_jobs.genome_from_genes(json.loads(c.genes))
        sig = ga.make_func(genome)(ind, p).where(ind["atr"].notna(), 0).astype(int).to_numpy()
        state = f"challenger_{c.id}_last_bar"
        last = tracking.get_state(con, pair, tf, state)
        start = n - 1 if last is None else int(ind.index.searchsorted(pd.Timestamp(last), side="right"))
        rows = [(pair, tf, c.spec, ts_str(ind.index[i]), int(sig[i]), float(atr[i]), max_bars, ts_str(now))
                for i in range(start, n) if sig[i]]
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO signal_log (pair, tf, source, signal_time, direction, atr, max_bars, logged_at,
               suppressed, shadow, status, entry_basis) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 1, '待機', 'next_open')""", rows)
        added += con.total_changes - before
        tracking.set_state(con, pair, tf, state, ts_str(ind.index[-1]))
    con.commit()
    return added


@dataclass
class StrategyComparison:
    days: float
    n: int
    pips: float
    mean: float
    p_value: float           # 「1回あたりの損益がプラス」が偶然である確率（ブートストラップ）
    dd: float
    base_n: int
    base_mean: float
    base_dd: float
    checks: list[tuple[str, bool, str]]

    @property
    def ready(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def compare_strategy(con, cid: int, now: pd.Timestamp, last_swap_at: pd.Timestamp | None = None) -> StrategyComparison:
    c = con.execute("SELECT pair, tf, spec, created_at FROM challengers WHERE id=?", (cid,)).fetchone()
    pair, tf, key, created = c
    days = (now - pd.Timestamp(created)).total_seconds() / 86400
    done = tuple(tracking.DONE)

    def closed(source, shadow):
        return pd.read_sql_query(
            f"""SELECT exit_time, pips FROM signal_log WHERE pair=? AND tf=? AND source=? AND shadow=? AND signal_time >= ?
                AND status IN ({",".join("?" * len(done))}) AND pips IS NOT NULL ORDER BY exit_time""",
            con, params=(pair, tf, source, shadow, created, *done))

    me, base = closed(key, 1), closed(STRATEGY_BASELINE, 0)
    x, y = me["pips"].to_numpy(dtype=float), base["pips"].to_numpy(dtype=float)
    mean = float(x.mean()) if len(x) else float("nan")
    bmean = float(y.mean()) if len(y) else float("nan")
    p = bootstrap_p(x)
    dd, bdd = _max_dd(x), _max_dd(y)
    checks = [
        ("シャドーモードの期間", days >= SHADOW_DAYS, f"{days:.0f}日（{SHADOW_DAYS}日以上）"),
        ("決済したシグナルの数", len(x) >= MIN_SIGNALS, f"{len(x)}件（{MIN_SIGNALS}件以上）"),
        ("コスト後の損益がプラスで、偶然ではない", len(x) > 0 and mean > 0 and p < P_VALUE,
         f"1回あたり {mean:+.1f} pips・p = {p:.2f}（{P_VALUE}未満）" if len(x) else "まだ記録がありません"),
        ("本番の「移動平均クロス」より良い", len(y) >= MIN_TAKEN and len(x) > 0 and mean > bmean,
         f"1回あたり {mean:+.1f} ／ 本番 {bmean:+.1f} pips（本番 {len(y)}件。{MIN_TAKEN}件以上）" if len(y) else "本番の記録がまだありません"),
        ("最大ドローダウン", -dd <= DD_TOLERANCE * -bdd + DD_SLACK_PIPS,
         f"{dd:,.0f} ／ 本番 {bdd:,.0f} pips（本番の{DD_TOLERANCE}倍＋{DD_SLACK_PIPS:g} pips まで）"),
    ]
    if last_swap_at is not None:
        gap = (now - last_swap_at).days
        checks.append(("前回の入れ替えから", gap >= SWAP_INTERVAL_DAYS, f"{gap}日（{SWAP_INTERVAL_DAYS}日以上）"))
    return StrategyComparison(days, len(x), float(x.sum()), mean, p, dd, len(y), bmean, bdd, checks)


def review_strategies(con, pair, tf, now: pd.Timestamp) -> list[str]:
    out = []
    ch = challengers(con, pair, tf)
    if ch.empty:
        return out
    for c in ch[(ch["kind"] == "strategy") & ch["status"].isin(["shadow", "ready"])].itertuples():
        cmp = compare_strategy(con, c.id, now, last_swap(con, pair, tf))
        if update_status(con, pair, tf, c.id, cmp, now):
            out.append(f"{strategy_label(c)} が、本番への追加の条件を満たしました")
    return out
