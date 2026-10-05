"""資金管理：分数ケリー（要件定義書 7章）。

ケリー基準：勝率 p、平均利益 ÷ 平均損失 b のとき、1回に「損切りで失う額」を残高の f* = p − (1 − p) / b にすると、
長い目で見て資産の増え方が最大になる、という理論上の値。ただし勝率などの見積もりの誤差に弱く、
f* をそのまま使うと資産が大きく上下するので、このツールでは次のように控えめに使う。
- f* の 4分の1（設定で変更可。上限は 2分の1）を使う
- 勝率・損益比は、最終テスト期間を除いたウォークフォワードの検証区間の成績から見積もる
- 取引が100回未満ならケリーは使わず、固定比率（1回の損失が残高の1%）にする
- どの場合でも、1回の損失は残高の2%まで
- f* がマイナス（期待値がマイナス）なら、取引しない（0%）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .backtest import MIN_TRADES_RELIABLE
from .settings import Sizing


@dataclass
class RiskDecision:
    risk_pct: float          # 1回の取引で、損切りになったときに失う額（残高に対する%）
    method: str              # "kelly" / "fixed" / "none"
    kelly_full_pct: float    # ケリー基準そのもの（f*、%）。計算できなければ nan
    explanation: str


def kelly_full(win_rate: float, payoff: float) -> float:
    """f* = p − (1 − p) / b。p：勝率、b：平均利益 ÷ 平均損失。"""
    if any(math.isnan(x) for x in (win_rate, payoff)) or payoff <= 0:
        return math.nan
    return win_rate - (1 - win_rate) / payoff


def decide_risk(n_trades: int, win_rate: float, payoff: float, s: Sizing) -> RiskDecision:
    f = kelly_full(win_rate, payoff)
    if n_trades < MIN_TRADES_RELIABLE or math.isnan(f):
        return RiskDecision(s.fixed_risk_pct, "fixed", f * 100 if not math.isnan(f) else math.nan,
                            f"検証の取引が {n_trades} 回（{MIN_TRADES_RELIABLE}回未満）なので、ケリー基準は使わず、"
                            f"固定比率（1回の損失が残高の {s.fixed_risk_pct:g}%）にしています。")
    if f <= 0:
        return RiskDecision(0.0, "none", f * 100,
                            f"ケリー基準の値がマイナス（{f:.1%}）で、検証の成績では期待値がマイナスです。取引しない（0%）のが目安です。")
    frac = min(s.kelly_fraction, 0.5)
    pct = min(f * frac * 100, s.max_risk_pct)
    capped = f * frac * 100 > s.max_risk_pct
    return RiskDecision(pct, "kelly", f * 100,
                        f"ケリー基準は {f:.1%}。その {frac:g} 倍（{f * frac:.2%}）"
                        + (f"が上限の {s.max_risk_pct:g}% を超えるので、{s.max_risk_pct:g}% にしています。" if capped
                           else "を、1回の損失の目安にしています。"))
