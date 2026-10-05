"""機械学習による方向予測（フェーズ3）。

「最新の確定足の終値から h 本後の終値が上がっているか」を確率で予測する。

未来のデータを混ぜないための決まりごと
- 特徴量はその足までのデータだけで計算する（指標はすべて過去方向のみ。テストで確認）。
- 学習期間を時間順に広げていくウォークフォワードで検証する。
- 学習データの最後の h 本は正解ラベルが検証期間の値動きを含むため、学習から外す（パージ）。
- ハイパーパラメータは固定。検証結果を見てチューニングすることはしない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import backtest as bt
from .config import RiskParams

HORIZONS = {"1h": (1, 4, 24), "1d": (1, 5, 20)}
LOOKBACKS = {"1h": (1, 4, 12, 24), "1d": (1, 5, 20, 60)}

MODELS = {
    "logistic": ("ロジスティック回帰", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
    "gbm": ("勾配ブースティング", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=False, random_state=0,
    )),
}

_LABELS = {
    "dist_sma_s": "短期SMAからの乖離", "dist_sma_l": "長期SMAからの乖離", "sma_spread": "短期SMAと長期SMAの差",
    "rsi": "RSI", "macd": "MACD", "macd_hist": "MACDヒストグラム",
    "bb_pos": "ボリンジャーバンド内の位置", "bb_width": "ボリンジャーバンドの幅",
    "atr_pct": "ATR（価格に対する%）", "atr_ratio": "ATRの変化（直近 ÷ 過去100本平均）",
    "dc_hi_dist": "直前高値までの距離", "dc_lo_dist": "直前安値までの距離",
    "fd_close": "分数階差分（過去の水準の情報を残した値動き）", "hour_sin": "時間帯(sin)", "hour_cos": "時間帯(cos)", "dow_sin": "曜日(sin)", "dow_cos": "曜日(cos)",
}


def feature_label(name: str) -> str:
    if name.startswith("ret_"):
        return f"直近{name[4:]}本の値動き"
    return _LABELS.get(name, name)


def make_features(ind: pd.DataFrame, tf_key: str, fracdiff: bool = False) -> pd.DataFrame:
    """値幅に関わる特徴量は ATR で割って、時期や通貨ペアが違っても同じ尺度になるようにする。

    fracdiff=True で分数階差分の特徴量（fd_close）を加える。本番のモデルは使っていない
    （特徴量を変えるのはチャレンジャー扱い：シャドーモード4週間とユーザーの承認が必要）。"""
    c, a = ind["close"], ind["atr"]
    f = pd.DataFrame(index=ind.index)
    for k in LOOKBACKS[tf_key]:
        f[f"ret_{k}"] = (c - c.shift(k)) / a
    f["dist_sma_s"] = (c - ind["sma_s"]) / a
    f["dist_sma_l"] = (c - ind["sma_l"]) / a
    f["sma_spread"] = (ind["sma_s"] - ind["sma_l"]) / a
    f["rsi"] = ind["rsi"]
    f["macd"] = ind["macd"] / a
    f["macd_hist"] = ind["macd_hist"] / a
    width = ind["bb_up"] - ind["bb_lo"]
    f["bb_pos"] = (c - ind["bb_lo"]) / width
    f["bb_width"] = width / a
    f["atr_pct"] = a / c * 100
    f["atr_ratio"] = a / a.rolling(100).mean()
    f["dc_hi_dist"] = (ind["dc_hi"] - c) / a
    f["dc_lo_dist"] = (c - ind["dc_lo"]) / a
    if tf_key == "1h":
        hour = ind.index.hour.to_numpy()
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = ind.index.dayofweek.to_numpy()
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    if fracdiff:
        from .fracdiff import fd_feature
        f["fd_close"] = fd_feature(c)
    return f.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, h: int) -> pd.Series:
    """h 本後の終値が今の終値より高ければ 1、そうでなければ 0。最後の h 本は答えがまだないので NaN。"""
    future = close.shift(-h)
    return (future > close).astype(float).where(future.notna())


@dataclass
class FoldInfo:
    train: tuple[int, int]
    test: tuple[int, int]
    acc: float
    majority_acc: float
    n: int


@dataclass
class ModelEval:
    kind: str
    name: str
    horizon: int
    threshold: float
    pred: pd.DataFrame               # 検証期間の予測（行番号 i、確率 p、正解 y、各基準の予測）
    folds: list[FoldInfo]
    n: int
    n_eff: float                     # ラベルの重なりを考慮した実質的な件数（n ÷ h）
    acc: float
    acc_ci: tuple[float, float]
    majority_acc: float
    momentum_acc: float
    logloss: float
    base_logloss: float
    auc: float
    trades: pd.DataFrame
    trade_metrics: dict
    random: pd.DataFrame
    beat: float
    signal_rate: float
    importance: pd.Series
    current_p: float
    current_time: pd.Timestamp
    model: object = None             # 答えのある全期間で学習した最終モデル
    trained_through: pd.Timestamp | None = None  # 最終モデルの学習に使った最後の足（これより後の足の予測は学習データ外）
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)


def make_folds(start: int, end: int, h: int, k: int, test_frac: float) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """学習期間を広げていくウォークフォワード。学習の終わりと検証の始まりの間に h 本の空白を置く。"""
    test_start = start + int((end - start) * (1 - test_frac))
    unit = (end - test_start) // k
    if unit < 20 or test_start - h - start < 200:
        raise ValueError("データが少なすぎて機械学習の検証ができません")
    out = []
    for i in range(k):
        ts = test_start + i * unit
        te = end if i == k - 1 else ts + unit
        out.append(((start, ts - h), (ts, te)))
    return out


def directional_signals(p: np.ndarray, rows: np.ndarray, n: int, threshold: float) -> list[int]:
    sig = np.zeros(n, dtype=int)
    sig[rows[p >= threshold]] = 1
    sig[rows[p <= 1 - threshold]] = -1
    return sig.tolist()


def fit_production(ind: pd.DataFrame, tf_key: str, kind: str, h: int) -> tuple[object, pd.Timestamp, pd.Timestamp]:
    """本番用モデル：設定は固定したまま、答えのある最新の足まで使って学習する（研究ノートの例外として、
    選ぶ判断には使わず、本番の予測だけに使う）。戻り値：(モデル, 学習に使った最初の足, 最後の足)。"""
    X = make_features(ind, tf_key).to_numpy()
    y = make_target(ind["close"], h).to_numpy()
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    rows = np.flatnonzero(valid)
    model = clone(MODELS[kind][1]).fit(X[rows], y[rows])
    return model, ind.index[rows[0]], ind.index[rows[-1]]


def trial_row(ev: "ModelEval", k: int, test_frac: float) -> dict:
    from .trials import ml_config

    return {"family": "ml", "name": ev.kind, "purpose": "ml_eval",
            "config": ml_config(ev.kind, ev.horizon, ev.threshold, k=k, test_frac=test_frac),
            "metrics": {"acc": ev.acc, "majority_acc": ev.majority_acc, "logloss": ev.logloss,
                        "base_logloss": ev.base_logloss, "auc": ev.auc, "trade": ev.trade_metrics,
                        "beat_random": ev.beat, "ok": ev.ok}}


def evaluate_model(
    ind: pd.DataFrame, tf_key: str, kind: str, h: int, threshold: float, risk: RiskParams, costs: bt.Costs,
    pip: float, k: int = 5, test_frac: float = 0.5, runs: int = 200, holdout_start: pd.Timestamp | None = None,
) -> ModelEval:
    """ind は確定足だけの指標付きデータで、開発用（最終テスト期間より前）であること。
    holdout_start を渡すと、それ以降の足が混ざっていたら止める。"""
    if holdout_start is not None and len(ind) and ind.index[-1] >= holdout_start:
        from .holdout import FinalTestAccessError
        raise FinalTestAccessError(f"最終テスト期間（{holdout_start} 以降）のデータが機械学習の検証に渡されています")
    name, template = MODELS[kind]
    X_df = make_features(ind, tf_key)
    y_s = make_target(ind["close"], h)
    X, y = X_df.to_numpy(), y_s.to_numpy()
    n = len(ind)
    valid = ~np.isnan(X).any(axis=1)
    start = int(np.argmax(valid))
    end = n - h  # ここから先は答えがない
    close = ind["close"].to_numpy()

    folds_idx = make_folds(start, end, h, k, test_frac)
    parts, folds, model = [], [], None
    for (a, b), (ts, te) in folds_idx:
        tr = np.arange(a, b)[valid[a:b]]
        te_rows = np.arange(ts, te)[valid[ts:te]]
        model = clone(template).fit(X[tr], y[tr])
        p = model.predict_proba(X[te_rows])[:, 1]
        base_rate = float(y[tr].mean())
        momentum = close[te_rows] > close[te_rows - h]
        part = pd.DataFrame({
            "i": te_rows, "fold": len(folds), "p": p, "y": y[te_rows], "base_rate": base_rate,
            "majority": float(base_rate >= 0.5), "momentum": momentum.astype(float),
        })
        parts.append(part)
        folds.append(FoldInfo((a, b), (ts, te), float(((p >= 0.5) == part["y"]).mean()),
                              float((part["majority"] == part["y"]).mean()), len(part)))
    pred = pd.concat(parts, ignore_index=True)
    last_test = (X[parts[-1]["i"].to_numpy()], parts[-1]["y"].to_numpy())

    yt, pt = pred["y"].to_numpy(), pred["p"].to_numpy()
    m = len(pred)
    acc = float(((pt >= 0.5) == yt).mean())
    n_eff = m / h
    se = math.sqrt(acc * (1 - acc) / n_eff)
    base_p = np.clip(pred["base_rate"].to_numpy(), 1e-6, 1 - 1e-6)

    # 確信度が高いときだけ売買（フェーズ2と同じエンジン・コスト。保有は最大 h 本）
    bars = bt.Bars(ind)
    rules = bt.ExitRules(opposite_exit=True, max_bars=h)
    sig = directional_signals(pt, pred["i"].to_numpy(), n, threshold)
    trades = pd.concat(
        [bt.simulate(bars, sig, risk, rules, costs, pip, f.test[0], f.test[1]) for f in folds], ignore_index=True
    )
    tm = bt.metrics(trades["pips"])
    windows = [(f.test[0], f.test[1], bt.signal_rate(sig, f.test[0], f.test[1])) for f in folds]
    random, _ = bt.random_baseline(bars, windows, risk, rules, costs, pip, runs, seed=7)
    beat = bt.beat_ratio(tm["total"], random["total"])

    imp = permutation_importance(model, *last_test, scoring="neg_log_loss", n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=X_df.columns).sort_values(ascending=False)

    # 現在の予測：答えのある全期間で学習し直して、最新の確定足について予測する
    tr_all = np.arange(start, end)[valid[start:end]]
    final = clone(template).fit(X[tr_all], y[tr_all])
    current_p = float(final.predict_proba(X[[n - 1]])[:, 1][0]) if valid[n - 1] else math.nan

    ev = ModelEval(
        kind=kind, name=name, horizon=h, threshold=threshold, pred=pred, folds=folds,
        n=m, n_eff=n_eff, acc=acc, acc_ci=(acc - 1.96 * se, acc + 1.96 * se),
        majority_acc=float((pred["majority"] == yt).mean()),
        momentum_acc=float((pred["momentum"] == yt).mean()),
        logloss=float(log_loss(yt, pt, labels=[0, 1])),
        base_logloss=float(log_loss(yt, base_p, labels=[0, 1])),
        auc=float(roc_auc_score(yt, pt)) if len(set(yt)) == 2 else math.nan,
        trades=trades, trade_metrics=tm, random=random, beat=beat,
        signal_rate=float(np.count_nonzero(sig)) / max(1, m),
        importance=importance, current_p=current_p, current_time=ind.index[n - 1],
        model=final, trained_through=ind.index[tr_all[-1]],
    )
    ev.checks = quality_checks(ev)
    return ev


def quality_checks(ev: ModelEval) -> list[tuple[str, bool, str]]:
    bar = max(ev.majority_acc, 0.5)
    tm = ev.trade_metrics
    return [
        ("正解率が基準を上回る", ev.acc_ci[0] > bar,
         f"正解率 {ev.acc:.1%}（95%範囲 {ev.acc_ci[0]:.1%}〜{ev.acc_ci[1]:.1%}）、基準 {bar:.1%}"),
        ("確率の質が基準より良い", ev.logloss < ev.base_logloss,
         f"対数損失 {ev.logloss:.4f}（基準 {ev.base_logloss:.4f}、小さいほど良い）"),
        ("コスト込みの売買でランダムを上回る", tm["total"] > 0 and ev.beat >= 0.95,
         f"損益 {tm['total']:+,.1f} pips（{tm['n']}回）、ランダム売買の {ev.beat:.0%} を上回る"),
    ]


def decide(p: float, ok: bool, threshold: float) -> str:
    """予測力が確認できない場合はシグナルを出さない。"""
    if not ok or math.isnan(p):
        return "シグナルなし"
    if p >= threshold:
        return "買い"
    if p <= 1 - threshold:
        return "売り"
    return "様子見"


def decision(ev: ModelEval) -> str:
    return decide(ev.current_p, ev.ok, ev.threshold)


def calibration(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """予測確率の帯ごとに、実際に上がった割合を集計する。"""
    q = pd.qcut(pred["p"], bins, duplicates="drop")
    g = pred.groupby(q, observed=True)
    return pd.DataFrame({"予測確率": g["p"].mean(), "実際に上がった割合": g["y"].mean(), "件数": g.size()})
