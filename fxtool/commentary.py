"""現在の相場の解説（フェーズ5。研究ノートを反映した改修版）。

ルールにもとづいて、計算済みの数値だけから日本語の文章を組み立てる（AI による文章生成は使わない）。
そのため、根拠のない理由を書くことはないが、ニュースや経済指標の結果など数値にない情報は扱えない。

根拠の強さは、過去の検証の判定に加えて「試した組み合わせの数」「DSR（偶然ではない確率）」「PBO（パラメータ選びの
過学習の確率）」で示す。機械学習（メタラベリング）の判定、最大損失による停止、決まった時刻の予定も書く。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import events as ev
from .config import IndicatorParams, RiskParams
from .strategies import StrategyView
from .tracking import score_label, weighted_score

JST = pd.Timedelta(hours=9)
WEAK_VERDICTS = {"優位性なし", "偶然の範囲", "不安定", "判断材料不足", "参考値", "偶然の可能性", "過学習の疑い"}
MAX_CAUTIONS = 7
NEWS_CAVEAT = ("経済指標の結果やニュース（要人発言・金融政策・地政学リスクなど）の中身は考慮していません。"
               "発表の前後は値動きが急変し、損切りの価格を飛び越えることもあります。")


@dataclass
class Commentary:
    stance: str                 # 買い寄り / 売り寄り / 中立 など
    headline: str
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


def _signal_text(key: str, direction: int, p: IndicatorParams) -> str:
    buy = direction > 0
    return {
        "ma_cross": f"短期線（SMA{p.sma_short}）が長期線（SMA{p.sma_long}）を{'上抜け（ゴールデンクロス）' if buy else '下抜け（デッドクロス）'}",
        "macd_cross": f"MACDがシグナル線を{'上抜け' if buy else '下抜け'}",
        "breakout": f"終値が直前{p.donchian_period}本の{'高値を上抜け' if buy else '安値を下抜け'}",
        "rsi_reversal": (f"RSIが売られすぎ（{p.rsi_low:g}未満）から回復" if buy
                         else f"RSIが買われすぎ（{p.rsi_high:g}超）から低下"),
        "bb_reversion": f"終値が{'−' if buy else '+'}{p.bb_std:g}σの外からバンド内に戻った",
    }[key]


def _when(bars_ago: int) -> str:
    return "最新の確定足" if bars_ago == 0 else f"{bars_ago}本前"


def _num(row: dict, key: str) -> float:
    v = row.get(key)
    return math.nan if v is None or pd.isna(v) else float(v)


def _evidence(row: dict) -> str:
    """過去の検証の判定と、試した数・DSR・PBO。"""
    parts = []
    n, dsr, pbo = _num(row, "n_trials"), _num(row, "dsr"), _num(row, "pbo")
    if not math.isnan(dsr):
        parts.append(f"偶然ではない確率（DSR）{dsr:.0%}")
    if not math.isnan(pbo):
        parts.append(f"パラメータ選びの過学習の確率（PBO）{pbo:.0%}")
    detail = "・".join(parts)
    if not math.isnan(n):
        detail = f"{int(n)}通り試した中で" + (f"、{detail}" if detail else "")
    return f"過去の検証では「{row['verdict']}」" + (f"（{detail}）" if detail else "") + "。"


def build(
    ind: pd.DataFrame, n_conf: int, views: list[StrategyView], weights: dict[str, float],
    alerts: pd.DataFrame, verdicts: pd.DataFrame, tf_key: str, pip: float, digits: int,
    p: IndicatorParams, risk: RiskParams, now: pd.Timestamp, label: Callable[[pd.Timestamp], str], *,
    meta: dict[str, tuple[float, str]] | None = None, meta_ok: bool = False, halt=None,
    upcoming: list[ev.Event] | None = None,
) -> Commentary:
    """meta：戦略のキー → その戦略の直近シグナルについての（勝つ確率, 乗る/見送り）。
    upcoming：その通貨ペアに関係する予定（events.upcoming）。"""
    meta, upcoming = meta or {}, upcoming or []
    last = ind.iloc[n_conf - 1]
    fmt = lambda x: f"{x:.{digits}f}"  # noqa: E731
    vrow = {r["source"]: r for r in verdicts.to_dict("records")} if not verdicts.empty else {}
    n_trials = max((_num(r, "n_trials") for r in vrow.values()), default=math.nan)
    alerts_of: dict[str, list[str]] = {}
    for a in alerts.itertuples():
        alerts_of.setdefault(a.source, []).append(a.message)
    halted = halt is not None and getattr(halt, "active", False)

    active = [v for v in views if v.decision in ("買い", "売り")]
    buys = [v for v in active if v.decision == "買い"]
    sells = [v for v in active if v.decision == "売り"]
    score = weighted_score({v.strategy.key: v.decision for v in views}, weights)  # 重みがなければ全て1
    stance = score_label(score)
    promising = [v for v in active if vrow.get(v.strategy.key, {}).get("verdict") == "有望（要継続監視）"]
    meta_take = [v for v in active if meta_ok and meta.get(v.strategy.key, (0, ""))[1] == "乗る"]
    meta_skip = [v for v in active if meta_ok and meta.get(v.strategy.key, (0, ""))[1] == "見送り"]

    # ---------------- 結論 ----------------
    if not active:
        headline = "様子見：いまはどの戦略からも有効なシグナルが出ていません。"
    else:
        parts = []
        if buys:
            parts.append(f"{'・'.join(v.strategy.name for v in buys)}が買い")
        if sells:
            parts.append(f"{'・'.join(v.strategy.name for v in sells)}が売り")
        headline = f"{stance}（重み付き総合スコア {score:+.2f}）：{'、'.join(parts)}のシグナルを出しています。"
        if not promising and not meta_take:
            n_txt = "" if math.isnan(n_trials) else f"（{int(n_trials)}通り試した中での結果）"
            headline += f"ただし、過去の検証で優位性が確認できた手法ではないため{n_txt}、根拠としては弱いシグナルです。"
        if meta_skip:
            headline += f"機械学習は{'・'.join(v.strategy.name for v in meta_skip)}のシグナルを「見送り」と判定しています。"
    if halted:
        headline = "⛔ 停止中：新しいシグナルは出していません（理由は注意点を参照）。" + headline

    # ---------------- なぜこのシグナルか ----------------
    reasons = []
    for v in active:
        r = (f"**{v.strategy.name}**（{v.strategy.kind}）：{_signal_text(v.strategy.key, v.direction, p)}"
             f"（{_when(v.bars_ago)}）→ {v.decision}。エントリー目安 {fmt(v.entry)}、"
             f"損切り {fmt(v.sl)}（{-risk.sl_atr * v.atr / pip:.1f} pips）、利確 {fmt(v.tp)}（+{risk.tp_atr * v.atr / pip:.1f} pips）。")
        if v.strategy.key in vrow:
            r += _evidence(vrow[v.strategy.key])
        reasons.append(r)

    up_trend = last["sma_s"] > last["sma_l"]
    above_long = last["close"] > last["sma_l"]
    if up_trend and above_long:
        trend = "価格が長期線の上にあり、短期線も長期線の上にあるので、**上昇トレンド**の形です。"
    elif not up_trend and not above_long:
        trend = "価格が長期線の下にあり、短期線も長期線の下にあるので、**下降トレンド**の形です。"
    else:
        trend = "価格と移動平均線の位置関係がそろっておらず、**トレンドがはっきりしない（転換点かもみ合い）**状態です。"
    rsi = last["rsi"]
    rsi_txt = "買われすぎ" if rsi > p.rsi_high else "売られすぎ" if rsi < p.rsi_low else "過熱感なし"
    hist, prev_hist = last["macd_hist"], ind["macd_hist"].iat[n_conf - 2]
    if hist > 0:
        mom = "上向きの勢いが" + ("強まっています" if hist > prev_hist else "弱まりつつあります")
    else:
        mom = "下向きの勢いが" + ("強まっています" if hist < prev_hist else "弱まりつつあります")
    reasons.append(f"{trend}RSIは {rsi:.0f}（{rsi_txt}）、MACDは{mom}。")

    if active:
        judged = [(v, meta[v.strategy.key]) for v in active if v.strategy.key in meta]
        if judged:
            txt = "、".join(f"{v.strategy.name}は「{d}」（コスト後に勝つ確率 {pr:.0%}）" for v, (pr, d) in judged)
            tail = ("予測力の条件を満たしたモデルなので、判断に含めています。" if meta_ok
                    else "予測力の条件を満たしていないため、参考です（判断には使っていません）。")
            reasons.append(f"機械学習（メタラベリング：シグナルに乗るか見送るか）：{txt}。{tail}")
        else:
            reasons.append("機械学習（メタラベリング）：このシグナルの判定はまだありません（モデルの学習より前に出たシグナルは判定しません）。")

    # ---------------- 注意点 ----------------
    cautions = []
    if halted:
        cautions.append(f"停止中：{halt.reason}。")
    for e in [x for x in upcoming if x.importance == "high"][:2]:
        state = ev.near(e, now)
        when = f"日本時間 {e.jst_label()}"
        if state == "直前":
            cautions.append(f"まもなく{e.name}です（{when}）。発表の前後は値動きが急変し、損切りの価格を飛び越えることもあります。")
        elif state == "直後":
            cautions.append(f"{e.name}の直後です（{when}）。値動きが荒く、スプレッドも広がりやすい状態です。")
        else:
            cautions.append(f"{when} に{e.name}の予定があります。" + (e.note + "。" if e.note else ""))
    if buys and sells:
        cautions.append("戦略によって買いと売りに判断が分かれています。方向感がはっきりしない相場の可能性があります。")
    for v in active:
        counter = (v.direction > 0 and not up_trend and not above_long) or (v.direction < 0 and up_trend and above_long)
        if counter:
            cautions.append(f"{v.strategy.name}の{v.decision}は、いまのトレンドと逆向きです。トレンドが続くと損切りになりやすい点に注意してください。")
    for v in active:
        for msg in alerts_of.get(v.strategy.key, []):
            cautions.append(f"警告中：{msg}")
        w = weights.get(v.strategy.key)
        if w is not None and w < 0.8:
            cautions.append(f"{v.strategy.name}の重みは {w:.2f} と低めです（本番の記録で成績が振るわない）。")

    atr_ratio = last["atr"] / ind["atr"].iloc[max(0, n_conf - 100):n_conf].mean()
    if atr_ratio >= 1.3:
        cautions.append(f"値動きが普段より大きくなっています（ATRが過去100本平均の {atr_ratio:.1f} 倍）。損切り幅も広くなるので、取引量に注意してください。")
    elif atr_ratio <= 0.7:
        cautions.append(f"値動きが普段より小さくなっています（ATRが過去100本平均の {atr_ratio:.1f} 倍）。小さな値動きが続いた後は、急に大きく動くことがあります。")
    if rsi > 80 or rsi < 20:
        cautions.append(f"RSIが {rsi:.0f} と極端な水準です。行き過ぎた値動きの反動が起きやすい状態です。")

    if tf_key == "1h":
        jst_now = now + JST
        if 5 <= jst_now.hour < 8:
            cautions.append("日本時間の早朝は取引量が少なく、スプレッドが広がりやすい時間帯です。")
        lows = [x for x in upcoming if x.importance == "low" and ev.near(x, now)]
        if lows:
            e = lows[0]
            cautions.append(f"{e.name}（日本時間 {e.jst_label()}）の前後です。値動きが一時的に大きくなりやすい時刻です。")
        if (jst_now.dayofweek == 4 and jst_now.hour >= 20) or (jst_now.dayofweek == 5 and jst_now.hour < 7):
            cautions.append("週末をまたいでポジションを持つと、月曜の取引開始時に窓を開けて大きく動くことがあります。")
    for msg in alerts_of.get("data", []):
        cautions.append(f"警告中：{msg}")

    # 重複を除き、件数を絞って、最後に必ずニュースの注意を付ける
    seen, uniq = set(), []
    for c in cautions:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    cautions = uniq[: MAX_CAUTIONS - 1] + [NEWS_CAVEAT]
    return Commentary(stance, headline, reasons, cautions)
