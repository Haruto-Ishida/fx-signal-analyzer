"""年1回の見直しの時期（2026-10-07 のユーザーの決定：「毎日の見張り＋年1回の見直し」）。

見直しとは：最終テスト期間の境界（holdout.json）を新しくして、新しい開発用データで戦略・設定を選び直すこと。
自動では行わない。時期が近づいたら（settings.toml の [review]）画面と Discord で知らせ、行うかどうかはユーザーが決める。

見直しの手順（そのときにユーザーと一緒に行う）
1. 区切りを動かす前に、いまの戦略・モデルを、いまの最終テスト期間で **1回だけ** 確かめて記録する（最も正直な成績の確認）
2. 境界を新しくする（使い終わった最終テスト期間は、開発用データに入る）
3. 新しい開発用データで、全通りの評価・最初からある戦略の判定・機械学習の検証をやり直す
   （データの期間が変わるので、試行は別に数える＝試行数が増え、合格の基準は少し厳しくなる）
4. 良い候補が見つかっても、すぐには使わない（チャレンジャーとしてシャドーモード → ユーザーの承認）
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import holdout
from .settings import Review
from .tracking import JST


@dataclass
class ReviewStatus:
    pair: str
    tf: str
    last: pd.Timestamp     # 前回、境界を決めた日（UTC）
    due: pd.Timestamp      # 見直しの時期（UTC）
    days_left: int         # 見直しの時期までの日数（マイナスなら過ぎている）
    state: str             # ok / soon（近づいた）/ overdue（過ぎた）

    def message(self) -> str:
        when = f"{(self.due + JST):%Y/%m/%d}"
        if self.state == "overdue":
            head = f"{self.pair} {'1時間足' if self.tf == '1h' else '日足'}：年1回の見直しの時期（{when}）を {-self.days_left} 日過ぎています。"
        else:
            head = f"{self.pair} {'1時間足' if self.tf == '1h' else '日足'}：年1回の見直しの時期（{when}）まで、あと {self.days_left} 日です。"
        return head + "検証に使う期間を新しくして選び直すかどうか、決めてください（自動では行いません）。"


def status(pair: str, tf: str, now: pd.Timestamp, r: Review | None = None, path=None) -> ReviewStatus | None:
    r = r or Review()
    b = holdout.get(pair, tf, path)
    if b is None:
        return None
    last = pd.Timestamp(b.fixed_at)
    due = last + pd.DateOffset(months=r.interval_months)
    days_left = int((due.normalize() - now.normalize()).days)
    state = "overdue" if days_left < 0 else ("soon" if days_left <= r.notice_days else "ok")
    return ReviewStatus(pair, tf, last, due, days_left, state)
