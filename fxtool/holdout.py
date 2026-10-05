"""最終テスト期間（ホールドアウト）の管理。研究ノート 2-3 の「最後の一定期間は開発中に一切使わない」。

- 通貨ペア・時間足ごとに、固定した時点で保存済みだった確定足の直近 20% を最終テスト期間とする。
- 境界の日時は holdout.json（git で管理）に保存し、あとから動かさない。データが増えても境界は同じ。
- 境界より後の足（固定時点の直近 20% と、それ以降に入ってくる新しいデータ）は、
  戦略・設定・モデルを選ぶ判断（バックテスト・検証・重み）には使わない。
- 本番の表示と、選んだ設定を固定したままの本番モデルの学習には、最新データまで使ってよい。
- 最終テスト期間のデータを取り出すには、確認の文言を付けて final_part() を呼ぶ必要がある（呼んだ記録が残る）。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ROOT

HOLDOUT_FILE = ROOT / "holdout.json"
FRACTION = 0.2
FINAL_TEST_CONFIRM = "最終テストを実行する"

# 固定した時点で、すでにこの期間を見てしまっていた評価（正直に記録しておく）
TOUCHED_USDJPY = (
    "固定より前に、既存の5戦略（初期パラメータ）とウォークフォワードのパラメータ候補、"
    "機械学習（2モデル × 予測先3種類）を全期間のデータで評価しており、この最終テスト期間の結果もすでに見ている。"
    "これらの戦略・設定については、この期間は「未使用の最終テスト」にはならない。"
)
TOUCHED_UNKNOWN = (
    "固定より前に、画面の検証タブ（バックテスト・機械学習）で全期間のデータを評価した可能性がある（記録なし）。"
)


class FinalTestAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Boundary:
    pair: str
    tf: str
    start: pd.Timestamp        # 最終テスト期間の最初の足（UTC）。これより前が開発用データ
    fixed_at: pd.Timestamp     # 境界を固定した日時（UTC）
    n_bars_at_fix: int         # 固定した時点の確定足の本数
    touched: str               # 固定前にこの期間を見てしまっていた評価についてのメモ


def _key(pair: str, tf: str) -> str:
    return f"{pair}|{tf}"


def _read(path: Path) -> dict:
    if not path.exists():
        return {"fraction": FRACTION, "boundaries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get(pair: str, tf: str, path: Path | None = None) -> Boundary | None:
    b = _read(path or HOLDOUT_FILE)["boundaries"].get(_key(pair, tf))
    if b is None:
        return None
    return Boundary(pair, tf, pd.Timestamp(b["start"]), pd.Timestamp(b["fixed_at"]), int(b["n_bars_at_fix"]),
                    b.get("touched", ""))


def fix(pair: str, tf: str, confirmed: pd.DataFrame, now: pd.Timestamp, touched: str,
        path: Path | None = None) -> Boundary:
    """まだ境界がなければ、確定足の直近 FRACTION を最終テスト期間として固定する。すでにあれば変えない。"""
    path = path or HOLDOUT_FILE
    existing = get(pair, tf, path)
    if existing is not None:
        return existing
    if len(confirmed) < 100:
        raise ValueError(f"{pair} {tf}：データが少なすぎて最終テスト期間を決められません（{len(confirmed)}本）")
    k = int(len(confirmed) * (1 - FRACTION))
    start = confirmed.index[k]
    doc = _read(path)
    doc.setdefault("note", "最終テスト期間（開発中は使わない）。境界は固定後に動かさないこと。")
    doc["boundaries"][_key(pair, tf)] = {
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "fixed_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "n_bars_at_fix": len(confirmed),
        "touched": touched,
    }
    _write(path, doc)
    return Boundary(pair, tf, start, now, len(confirmed), touched)


def dev_part(df: pd.DataFrame, b: Boundary) -> pd.DataFrame:
    """開発用データ（最終テスト期間より前の足だけ）。"""
    return df[df.index < b.start]


def assert_dev_only(df: pd.DataFrame, b: Boundary) -> None:
    if len(df) and df.index[-1] >= b.start:
        raise FinalTestAccessError(
            f"{b.pair} {b.tf}：最終テスト期間（{b.start} 以降）のデータが、開発用の処理に渡されています"
        )


def final_part(df: pd.DataFrame, b: Boundary, *, confirm: str, con: sqlite3.Connection, reason: str,
               now: pd.Timestamp) -> pd.DataFrame:
    """最終テスト期間のデータ。確認の文言が必要で、取り出した記録を DB に残す。"""
    if confirm != FINAL_TEST_CONFIRM:
        raise FinalTestAccessError(f"最終テスト期間のデータを使うには confirm='{FINAL_TEST_CONFIRM}' が必要です")
    con.execute(
        "INSERT INTO final_test_access (pair, tf, accessed_at, reason) VALUES (?, ?, ?, ?)",
        (b.pair, b.tf, now.strftime("%Y-%m-%d %H:%M:%S"), reason),
    )
    con.commit()
    return df[df.index >= b.start]
