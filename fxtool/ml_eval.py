"""機械学習の検証（フェーズ3改修。研究ノート 2-2・3・4）。

- ラベルはトリプルバリア（labeling.triple_barrier）。
- メタラベリング：5つの戦略のシグナル（方向は戦略が決める）について、「コスト後に利益が出る確率」を予測し、
  確信度に届かないシグナルは見送る。方向予測：上下同じ幅のトリプルバリアで「上に先に届く確率」を予測する。
- 検証の中心はウォークフォワード（学習期間を時間順に広げる）。ラベルの期間が検証期間に入るサンプルは学習から外す（パージ）。
  補助として Purged K-Fold（パージ＋エンバーゴ）でも確かめる。
- 学習では、ラベルの期間の重なりに応じてサンプルの重みを下げる（独自性）。
- モデルは ロジスティック回帰 → ランダムフォレスト → 勾配ブースティング の順に比べ、
  複雑なモデルは、単純なモデルにコスト後の成績で勝ったときだけ採用する。ハイパーパラメータは固定。
- この評価は開発用データ（最終テスト期間より前）だけで行う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import backtest as bt
from . import labeling, overfit, strategies
from .config import IndicatorParams, RiskParams
from .ml import feature_label as _base_label
from .ml import make_features

LADDER = {
    "logistic": ("ロジスティック回帰", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
    "rf": ("ランダムフォレスト", RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=50, max_features="sqrt", random_state=0, n_jobs=-1)),
    "gbm": ("勾配ブースティング", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=False, random_state=0)),
}
MODES = {"meta": "メタラベリング（おすすめ）", "direction": "方向予測（トリプルバリア）"}
# 確信度の初期値と、DSR のばらつきの計算に使う代表的な確信度。メタラベリングは「勝つ確率」なので、
# 利確が損切りの2倍なら約33%が損益の分かれ目になり、方向予測（50%が分かれ目）とは水準が違う
DEFAULT_THRESHOLD = {"meta": 0.40, "direction": 0.55}
THRESHOLDS = {"meta": (0.35, 0.40, 0.45, 0.50), "direction": (0.50, 0.55, 0.60, 0.65)}

# 方向に意味がある特徴量（メタラベリングでは「シグナルの方向を掛けたもの」も加える）
_DIRECTIONAL = ("dist_sma_s", "dist_sma_l", "sma_spread", "macd", "macd_hist", "dc_hi_dist", "dc_lo_dist")


def feature_label(name: str) -> str:
    if name.endswith("_x_dir"):
        return f"{feature_label(name[:-6])}×方向"
    if name.startswith("from_"):
        return f"シグナル元：{strategies.STRATEGY_BY_KEY[name[5:]].name}"
    if name == "dir":
        return "シグナルの方向"
    return _base_label(name)


@dataclass
class Samples:
    mode: str
    X: pd.DataFrame            # 特徴量（行＝サンプル、t0 の順）
    y: np.ndarray              # 1 = コスト後に利益（メタ）／上に先に届いた（方向）
    t0: np.ndarray             # 特徴量を計算した足（シグナルの足）の行番号
    t1: np.ndarray             # ラベルが決まった足（決済の足）の行番号
    pips_long: np.ndarray      # メタ：そのシグナルに乗ったときのコスト後の損益／方向：買ったときの損益
    pips_short: np.ndarray     # 方向：売ったときの損益（メタでは使わない）
    source: np.ndarray         # メタ：シグナル元の戦略のキー
    index: pd.DatetimeIndex    # 足の時刻
    max_bars: int
    bars: bt.Bars | None = None            # 方向：ポジション1つずつの売買シミュレーション用
    risk: RiskParams | None = None         # 方向：上下同じ幅のバリア
    costs: bt.Costs | None = None
    pip: float = 0.01


def _fit(template, kind: str, X, y, w):
    m = clone(template)
    if isinstance(m, Pipeline):
        return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    return m.fit(X, y, sample_weight=w)


def build_meta(ind: pd.DataFrame, tf_key: str, p: IndicatorParams, risk: RiskParams, max_bars: int,
               cost_pips: float, pip: float) -> Samples:
    F = make_features(ind, tf_key)
    ev = []
    for s in strategies.STRATEGIES:
        sig = strategies.signals(ind, s, p).to_numpy()
        for i in np.flatnonzero(sig):
            ev.append((int(i), int(sig[i]), s.key))
    events = pd.DataFrame(ev, columns=["i", "d", "source"]).sort_values(["i", "source"], kind="stable").reset_index(drop=True)
    tb = labeling.triple_barrier(ind, events, risk, max_bars, cost_pips, pip)
    tb["source"] = events["source"].to_numpy()[tb["ev"].to_numpy()]
    X = F.iloc[tb["i"].to_numpy()].reset_index(drop=True)
    d = tb["d"].to_numpy()
    for col in [c for c in F.columns if c.startswith("ret_")] + list(_DIRECTIONAL):
        X[f"{col}_x_dir"] = X[col].to_numpy() * d
    X["bb_pos_x_dir"] = (X["bb_pos"].to_numpy() - 0.5) * d
    X["rsi_x_dir"] = (X["rsi"].to_numpy() - 50) * d
    X["dir"] = d
    for s in strategies.STRATEGIES:
        X[f"from_{s.key}"] = (tb["source"].to_numpy() == s.key).astype(float)
    ok = ~X.isna().any(axis=1).to_numpy()
    tb, X = tb[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    return Samples("meta", X, (tb["pips"].to_numpy() > 0).astype(float), tb["i"].to_numpy(), tb["t1"].to_numpy(),
                   tb["pips"].to_numpy(), np.full(len(tb), np.nan), tb["source"].to_numpy(), ind.index, max_bars)


def build_direction(ind: pd.DataFrame, tf_key: str, width: float, max_bars: int, costs: bt.Costs, pip: float) -> Samples:
    """上下同じ幅（ATR × width）のトリプルバリア。買いと売りは同じ足で決済され、コスト前の損益は正負が逆になる。
    売買の成績は、予測から作ったシグナルを、ポジション1つずつの売買エンジン（フェーズ2と同じ）で測る。"""
    F = make_features(ind, tf_key)
    valid = np.flatnonzero(~F.isna().any(axis=1).to_numpy())
    events = pd.DataFrame({"i": valid, "d": 1})
    risk = RiskParams(sl_atr=width, tp_atr=width)
    tb = labeling.triple_barrier(ind, events, risk, max_bars, costs.round_trip, pip)
    X = F.iloc[tb["i"].to_numpy()].reset_index(drop=True)
    g = tb["gross"].to_numpy()
    return Samples("direction", X, (g > 0).astype(float), tb["i"].to_numpy(), tb["t1"].to_numpy(),
                   g - costs.round_trip, -g - costs.round_trip, np.array([""] * len(tb)), ind.index, max_bars,
                   bt.Bars(ind), risk, costs, pip)


# ---------------- 評価 ----------------

@dataclass
class Result:
    kind: str
    name: str
    mode: str
    threshold: float
    pred: pd.DataFrame                  # 検証期間の予測（行＝サンプル番号 s、fold, p, y, base_rate, w）
    folds: list[tuple[int, int]]        # 検証期間（足の行番号）
    n: int
    n_eff: float                        # 重複度の重みの合計（実質的に独立なサンプルの数）
    acc: float
    acc_ci: tuple[float, float]
    majority_acc: float
    logloss: float
    base_logloss: float
    auc: float
    trades: pd.DataFrame                # 確信度を超えて売買したもの（列 s, side, pips, t1）
    trade_metrics: dict
    all_metrics: dict                   # メタ：全シグナルに乗った場合
    random_totals: np.ndarray           # 同じ数だけランダムに選んだ（方向はランダム）場合の損益合計
    beat: float
    daily: pd.Series                    # 売買した分の日ごとの損益
    purged: dict = field(default_factory=dict)      # Purged K-Fold の結果
    importance: pd.Series = None
    dsr: dict = field(default_factory=dict)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    last_fold: tuple = None             # （モデル, 最後の検証区間の X, y）：重要度の計算用

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c[1] for c in self.checks)


def walk_forward_predict(smp: Samples, kind: str, k: int = 5, test_frac: float = 0.5):
    """学習期間を広げていくウォークフォワード。検証期間の始まりより後に決済するラベルは学習に使わない（パージ）。"""
    template = LADDER[kind][1]
    t0, t1, y, X = smp.t0, smp.t1, smp.y, smp.X.to_numpy()
    w_all = labeling.uniqueness(t0, t1)  # 検証側の「実質の件数」を数えるためだけに使う（学習の重みには使わない）
    start, end = int(t0.min()), int(t0.max()) + 1
    test_start = start + int((end - start) * (1 - test_frac))
    unit = (end - test_start) // k
    parts, folds, model, last = [], [], None, None
    for f in range(k):
        a = test_start + f * unit
        b = end if f == k - 1 else a + unit
        tr = t1 < a
        te = (t0 >= a) & (t0 < b)
        if tr.sum() < 100 or te.sum() < 10:
            raise ValueError("データが少なすぎて機械学習の検証ができません")
        # 学習の重みは、学習データ（検証期間の前に決着したラベル）だけから計算する。未決着のラベルの決済時刻は未来の情報なので使わない
        w_tr = labeling.uniqueness(t0[tr], t1[tr])
        model = _fit(template, kind, X[tr], y[tr], w_tr / w_tr.mean())
        p = model.predict_proba(X[te])[:, 1]
        idx = np.flatnonzero(te)
        parts.append(pd.DataFrame({"s": idx, "fold": f, "p": p, "y": y[te], "base_rate": float(np.average(y[tr])),
                                   "w": w_all[te]}))
        folds.append((a, b))
        last = (X[te], y[te])
    return pd.concat(parts, ignore_index=True), folds, model, last


def purged_kfold(smp: Samples, kind: str, k: int = 5, embargo: int | None = None) -> dict:
    """Purged K-Fold：期間を k 個に分け、それぞれを検証に使う。学習データからは、ラベルの期間が検証期間と重なるもの（パージ）と、
    検証期間の直後 embargo 本以内に始まるもの（エンバーゴ）を除く。未来側のデータでも学習するので、本番に近いのはウォークフォワードの方。"""
    embargo = smp.max_bars if embargo is None else embargo
    template = LADDER[kind][1]
    t0, t1, y, X = smp.t0, smp.t1, smp.y, smp.X.to_numpy()
    edges = np.linspace(int(t0.min()), int(t0.max()) + 1, k + 1).astype(int)
    p_all = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        te = (t0 >= a) & (t0 < b)
        tr = labeling.purged_train_mask(t0, t1, a, b, embargo)
        if tr.sum() < 100 or te.sum() == 0:
            continue
        w_tr = labeling.uniqueness(t0[tr], t1[tr])
        m = _fit(template, kind, X[tr], y[tr], w_tr / w_tr.mean())
        p_all[te] = m.predict_proba(X[te])[:, 1]
        base[te] = np.average(y[tr])
    ok = ~np.isnan(p_all)
    yt, pt, bt_ = y[ok], p_all[ok], np.clip(base[ok], 1e-6, 1 - 1e-6)
    out = {"n": int(ok.sum()), "logloss": float(log_loss(yt, pt, labels=[0, 1])),
           "base_logloss": float(log_loss(yt, bt_, labels=[0, 1])),
           "auc": float(roc_auc_score(yt, pt)) if len(set(yt)) == 2 else math.nan, "embargo": embargo}
    if smp.mode == "meta":
        pips = smp.pips_long[ok]
        out["avg_all"] = float(pips.mean())
        out["avg_taken_55"] = float(pips[pt >= 0.55].mean()) if (pt >= 0.55).any() else math.nan
    return out


def _direction_signals(smp: Samples, pred: pd.DataFrame, threshold: float) -> list[int]:
    sig = np.zeros(len(smp.index), dtype=int)
    p, bars_i = pred["p"].to_numpy(), smp.t0[pred["s"].to_numpy()]
    sig[bars_i[p >= threshold]] = 1
    sig[bars_i[p <= 1 - threshold]] = -1
    return sig.tolist()


def _direction_rules(smp: Samples) -> bt.ExitRules:
    return bt.ExitRules(opposite_exit=False, max_bars=smp.max_bars)


def _trades(smp: Samples, pred: pd.DataFrame, threshold: float, folds: list[tuple[int, int]]) -> pd.DataFrame:
    """売買した取引（列 side, pips, t1, fold）。
    メタ：確信度を超えたシグナルごとに独立して売買（全シグナル・ランダムな見送りと同じ扱いで比べる）。
    方向：予測から作ったシグナルを、ポジション1つずつの売買エンジンで売買（区間の終わりで決済）。"""
    if smp.mode == "meta":
        p, s = pred["p"].to_numpy(), pred["s"].to_numpy()
        take = p >= threshold
        return pd.DataFrame({"side": 1, "pips": smp.pips_long[s[take]], "t1": smp.t1[s[take]],
                             "fold": pred["fold"].to_numpy()[take]})
    sig = _direction_signals(smp, pred, threshold)
    parts = []
    for f, (a, b) in enumerate(folds):
        t = bt.simulate(smp.bars, sig, smp.risk, _direction_rules(smp), smp.costs, smp.pip, a, b)
        parts.append(pd.DataFrame({"side": t["dir"], "pips": t["pips"], "t1": t["exit_i"], "fold": f}))
    return pd.concat(parts, ignore_index=True)


def _random_totals(smp: Samples, pred: pd.DataFrame, trades: pd.DataFrame, threshold: float,
                   folds: list[tuple[int, int]], runs: int, seed: int) -> np.ndarray:
    """メタ：区間ごとに、売買したのと同じ数のシグナルをランダムに選んだ（同じ割合だけランダムに見送った）場合の損益合計。
    方向：同じ頻度・ランダムな方向でエントリーし、同じ売買エンジン・決済ルールで売買した場合の損益合計。"""
    if smp.mode == "direction":
        sig = np.asarray(_direction_signals(smp, pred, threshold))
        windows = [(a, b, float(np.count_nonzero(sig[a:b])) / max(1, b - a)) for a, b in folds]
        r, _ = bt.random_baseline(smp.bars, windows, smp.risk, _direction_rules(smp), smp.costs, smp.pip, runs, seed)
        return r["total"].to_numpy()
    rng = np.random.default_rng(seed)
    by_fold = {f: g["s"].to_numpy() for f, g in pred.groupby("fold")}
    counts = trades.groupby("fold").size().to_dict()
    out = np.zeros(runs)
    for r in range(runs):
        total = 0.0
        for f, pool in by_fold.items():
            m = int(counts.get(f, 0))
            if m:
                total += smp.pips_long[rng.choice(pool, size=min(m, len(pool)), replace=False)].sum()
        out[r] = total
    return out


def _daily(smp: Samples, trades: pd.DataFrame, start_bar: int) -> pd.Series:
    idx = smp.index
    last = int(max(smp.t1.max(), start_bar))
    days = overfit.business_days(idx[start_bar], idx[last])
    return overfit.daily_pnl(idx[trades["t1"].to_numpy(dtype=int)], trades["pips"], days)


def add_importance(res: "Result", smp: Samples) -> None:
    """最後の検証区間で、特徴量をシャッフルしたときに対数損失がどれだけ悪化するか。"""
    model, X, y = res.last_fold
    r = permutation_importance(model, X, y, scoring="neg_log_loss", n_repeats=3, random_state=0)
    res.importance = pd.Series(r.importances_mean, index=smp.X.columns).sort_values(ascending=False)


def evaluate(smp: Samples, kind: str, threshold: float, k: int = 5, test_frac: float = 0.5, runs: int = 200,
             seed: int = 0, importance: bool = True, purged: bool = True) -> Result:
    pred, folds, model, last = walk_forward_predict(smp, kind, k, test_frac)
    yt, pt, w = pred["y"].to_numpy(), pred["p"].to_numpy(), pred["w"].to_numpy()
    acc = float(((pt >= 0.5) == (yt == 1)).mean())
    n_eff = float(w.sum())
    se = math.sqrt(acc * (1 - acc) / max(n_eff, 1))
    base_p = np.clip(pred["base_rate"].to_numpy(), 1e-6, 1 - 1e-6)
    majority = (pred["base_rate"].to_numpy() >= 0.5).astype(float)

    trades = _trades(smp, pred, threshold, folds).sort_values("t1", kind="stable")
    rnd = _random_totals(smp, pred, trades, threshold, folds, runs, seed)
    tm = bt.metrics(trades["pips"])
    all_m = bt.metrics(smp.pips_long[pred["s"].to_numpy()]) if smp.mode == "meta" else {}
    res = Result(
        kind=kind, name=LADDER[kind][0], mode=smp.mode, threshold=threshold, pred=pred, folds=folds,
        n=len(pred), n_eff=n_eff, acc=acc, acc_ci=(acc - 1.96 * se, acc + 1.96 * se),
        majority_acc=float((majority == yt).mean()),
        logloss=float(log_loss(yt, pt, labels=[0, 1])), base_logloss=float(log_loss(yt, base_p, labels=[0, 1])),
        auc=float(roc_auc_score(yt, pt)) if len(set(yt)) == 2 else math.nan,
        trades=trades, trade_metrics=tm, all_metrics=all_m, random_totals=rnd,
        beat=float((rnd < tm["total"]).mean()), daily=_daily(smp, trades, folds[0][0]),
        purged=purged_kfold(smp, kind, k) if purged else {}, last_fold=(model, *last),
    )
    if importance:
        add_importance(res, smp)
    return res


def checks(res: Result) -> list[tuple[str, bool, str]]:
    """シグナルを出す条件（すべて満たしたときだけ）。"""
    tm = res.trade_metrics
    d = res.dsr.get("dsr", math.nan)
    trade = (f"損益 {tm['total']:+,.1f} pips（{tm['n']}回）、DSR {'—' if math.isnan(d) else f'{d:.0%}'}"
             f"（試した {res.dsr.get('n_trials', 0)} 通りで割り引き）")
    trade_ok = tm["total"] > 0 and tm["n"] >= bt.MIN_TRADES_RELIABLE and not math.isnan(d) and d >= 0.95
    quality = (res.logloss < res.base_logloss,
               f"対数損失 {res.logloss:.4f}（基準 {res.base_logloss:.4f}、小さいほど良い）")
    if res.mode == "meta":
        am = res.all_metrics
        better = tm["n"] > 0 and not math.isnan(tm["avg"]) and tm["avg"] > am["avg"]
        return [
            ("確率の質が基準より良い", *quality),
            ("見送ることで成績が良くなる",
             better and res.beat >= 0.95,
             f"1回あたり {tm['avg']:+.2f} pips（全シグナルに乗ると {am['avg']:+.2f}）、同じ数をランダムに選んだ場合の {res.beat:.0%} を上回る"),
            ("コスト込みでプラス・取引100回以上・偶然ではない（DSR 95%以上）", trade_ok, trade),
        ]
    bar = max(res.majority_acc, 0.5)
    return [
        ("正解率が基準を上回る", res.acc_ci[0] > bar,
         f"正解率 {res.acc:.1%}（95%範囲 {res.acc_ci[0]:.1%}〜{res.acc_ci[1]:.1%}、実質 {res.n_eff:,.0f} 件）、基準 {bar:.1%}"),
        ("確率の質が基準より良い", *quality),
        ("コスト込みでプラス・ランダムを上回る・取引100回以上・偶然ではない（DSR 95%以上）",
         trade_ok and res.beat >= 0.95, trade + f"、ランダム売買の {res.beat:.0%} を上回る"),
    ]


@dataclass
class Report:
    mode: str
    threshold: float
    results: dict[str, Result]
    adopted: str
    ladder_notes: list[str]
    n_samples: int
    base_win_rate: float
    n_trials: int = 0
    sr_variance: float = math.nan
    trials: list[dict] = field(default_factory=list)


def choose_model(results: dict[str, Result]) -> tuple[str, list[str]]:
    """単純なモデルから順に、コスト後の検証区間の損益で勝ったときだけ、より複雑なモデルに進む。"""
    order = [k for k in LADDER if k in results]
    adopted, notes = order[0], []
    for kind in order[1:]:
        a, c = results[adopted], results[kind]
        if c.trade_metrics["total"] > a.trade_metrics["total"]:
            notes.append(f"{c.name}は{a.name}に勝った（{c.trade_metrics['total']:+,.0f} > {a.trade_metrics['total']:+,.0f} pips）→ {c.name}を採用")
            adopted = kind
        else:
            notes.append(f"{c.name}は{a.name}に勝てなかった（{c.trade_metrics['total']:+,.0f} ≤ {a.trade_metrics['total']:+,.0f} pips）→ {a.name}のまま")
    return adopted, notes


def evaluate_all(smp: Samples, threshold: float, k: int = 5, runs: int = 200, n_trials: int | None = None,
                 kinds=tuple(LADDER)) -> Report:
    results = {kind: evaluate(smp, kind, threshold, k, runs=runs, seed=i, importance=False, purged=False)
               for i, kind in enumerate(kinds)}
    adopted, notes = choose_model(results)
    # 重要度と Purged K-Fold は、採用したモデルだけ計算する（時間がかかるため）
    add_importance(results[adopted], smp)
    results[adopted].purged = purged_kfold(smp, adopted, k)
    rep = Report(smp.mode, threshold, results, adopted, notes, len(smp.y), float(smp.y.mean()))
    from .trials import ml_config
    label = "triple_barrier_meta" if smp.mode == "meta" else "triple_barrier_direction"
    rep.trials = [{"family": "ml", "name": r.kind, "purpose": "ml_eval_v2",
                   "config": ml_config(r.kind, smp.max_bars, threshold, label=label, features="v2", k=k),
                   "metrics": {"acc": r.acc, "logloss": r.logloss, "base_logloss": r.base_logloss, "auc": r.auc,
                               "trade": r.trade_metrics, "beat_random": r.beat}} for r in results.values()]
    return finalize(rep, smp, n_trials if n_trials is not None else len(results) * len(THRESHOLDS[smp.mode]))


def finalize(rep: Report, smp: Samples, n_trials: int) -> Report:
    """試した数で DSR を割り引き、条件を判定する。SR のばらつきは、モデル × 代表的な確信度 の組み合わせから求める。"""
    srs = []
    for r in rep.results.values():
        for th in THRESHOLDS[smp.mode]:
            t = _trades(smp, r.pred, th, r.folds)
            srs.append(overfit.sharpe(_daily(smp, t, r.folds[0][0])))
    rep.n_trials = int(n_trials)
    rep.sr_variance = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    for r in rep.results.values():
        r.dsr = overfit.deflated_sharpe(r.daily.to_numpy(), rep.n_trials, rep.sr_variance)
        r.checks = checks(r)
    return rep
