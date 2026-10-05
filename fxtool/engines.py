"""予測エンジンの共通の形（要件定義書 4章の3）。

どのモデルも Engine(kind).train(X, y, sample_weight) → .predict(X)（1 になる確率）の同じ形で使う。
新しいモデルを足すときは LADDER に追加する（ハイパーパラメータは固定。検証結果を見て調整しない）。
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class QuantileBinner(BaseEstimator, TransformerMixin):
    """勾配ブースティングの前処理：各特徴量を、学習データの分位点（重みなし）で最大 max_bins 段階に区切る。

    scikit-learn 1.9 では、サンプルの重みを渡すと勾配ブースティングの「区切りの計算」が約30倍遅くなる。
    特徴量の値が max_bins 種類以下なら勾配ブースティングは速い計算方法を使うので、区切りをここで先に計算しておく。
    木の学習にはサンプルの重みをそのまま使う。区切りは学習データだけから計算するので、未来の情報は混ざらない。"""

    def __init__(self, max_bins: int = 255):
        self.max_bins = max_bins

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.edges_ = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= self.max_bins:
                edges = (u[:-1] + u[1:]) / 2
            else:  # scikit-learn（重みなし）と同じ分位点の求め方
                qs = np.linspace(0, 100, self.max_bins + 1)[1:-1]
                edges = np.percentile(X[:, j], qs, method="averaged_inverted_cdf")
            self.edges_.append(edges)
        return self

    def transform(self, X):
        # 区切り e[i-1] < x <= e[i] を i 番目の段階にする（scikit-learn と同じ向き）
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.searchsorted(e, X[:, j], side="left") for j, e in enumerate(self.edges_)]).astype(float)


# モデルの段階（単純 → 複雑）。研究ノート 4-1 の順番
LADDER = {
    "logistic": ("ロジスティック回帰", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
    # n_jobs=1：データが小さいので並列化しない方が速い（Windows では並列処理の起動・終了に毎回数秒かかっていた）
    "rf": ("ランダムフォレスト", RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=50, max_features="sqrt", random_state=0, n_jobs=1)),
    "gbm": ("勾配ブースティング", make_pipeline(QuantileBinner(), HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=False, random_state=0))),
}


class Engine:
    """予測エンジンの共通の形。"""

    def __init__(self, kind: str):
        if kind not in LADDER:
            raise ValueError(f"知らないモデルの種類です：{kind}")
        self.kind = kind
        self.name = LADDER[kind][0]
        self.model = None

    def train(self, X, y, sample_weight=None) -> "Engine":
        m = clone(LADDER[self.kind][1])
        if sample_weight is None:
            self.model = m.fit(X, y)
        elif isinstance(m, Pipeline):
            self.model = m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": sample_weight})
        else:
            self.model = m.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        """1（勝つ・上に先に届く）になる確率。"""
        if self.model is None:
            raise RuntimeError("train() の前に predict() は使えません")
        return self.model.predict_proba(X)[:, 1]
