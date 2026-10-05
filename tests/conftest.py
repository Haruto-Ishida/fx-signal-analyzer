import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_holdout_file(tmp_path, monkeypatch):
    """テストが本物の holdout.json（最終テスト期間の境界）を書き換えないよう、必ず一時ファイルを使う。"""
    from fxtool import holdout

    monkeypatch.setattr(holdout, "HOLDOUT_FILE", tmp_path / "holdout.json")


def make_ohlc(close, start="2025-01-01", freq="h", spread=0.05) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq=freq)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
    }, index=idx)


def make_path_ohlc(n: int, sub: int = 50, sd: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """細かいランダムウォークを足にまとめた、高値・安値が本物の値動きと矛盾しないデータ。"""
    rng = np.random.default_rng(seed)
    p = (150 + np.cumsum(rng.normal(0, sd / np.sqrt(sub), n * sub))).reshape(n, sub)
    prev = np.r_[150, p[:-1, -1]]
    return pd.DataFrame({
        "open": prev,
        "high": np.maximum(p.max(axis=1), prev),
        "low": np.minimum(p.min(axis=1), prev),
        "close": p[:, -1],
    }, index=pd.date_range("2020-01-01", periods=n, freq="h"))


@pytest.fixture
def random_walk() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return make_ohlc(150 + np.cumsum(rng.normal(0, 0.2, 1500)))
