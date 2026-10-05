"""有名手法ライブラリ（要件定義書 12章）。

公開されていて、数値で再現できる売買手法を、元のルールにできるだけ近い形で戦略の共通の形（Strategy）にする。
- 出典とルールの要点は自分の言葉でまとめ、書籍や記事の文章はそのまま保存しない。
- 元のルールを完全には再現できない部分は、spec["simplifications"] に必ず書く。
- まずは元のルールのままの版だけを作る（パラメータを変えた版は作らない。ユーザーの決定）。
- 元のルールは日足（1日ごと）で考えられているので、日足で検証する。
- すべて、その足までのデータだけで計算する（テストで確認）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators
from .config import IndicatorParams
from .strategies import Strategy, _signal


def _hh(s: pd.Series, n: int) -> pd.Series:
    """直前 n 本の最高値（その足を含まない）。"""
    return s.rolling(n).max().shift()


def _ll(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).min().shift()


def _first(cond: pd.Series) -> pd.Series:
    """条件が成り立った最初の足だけ True。"""
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(fill_value=False).astype(bool)


def _exit_series(exit_long: pd.Series, exit_short: pd.Series) -> pd.Series:
    """決済シグナル：+1 = 買いを決済、−1 = 売りを決済、2 = どちらも。"""
    el, es = exit_long.fillna(False).astype(bool), exit_short.fillna(False).astype(bool)
    return pd.Series(np.select([el & es, el, es], [2, 1, -1], 0), index=el.index, dtype=int)


# ---------------- タートルズ ----------------

def _turtle_entries(d: pd.DataFrame, entry_n: int) -> tuple[pd.Series, pd.Series]:
    return _first(d["close"] > _hh(d["high"], entry_n)), _first(d["close"] < _ll(d["low"], entry_n))


def turtle_s2(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    buy, sell = _turtle_entries(d, 55)
    return _signal(buy, sell)


def turtle_s2_exit(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    return _exit_series(d["close"] < _ll(d["low"], 20), d["close"] > _hh(d["high"], 20))


def turtle_s1_exit(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    return _exit_series(d["close"] < _ll(d["low"], 10), d["close"] > _hh(d["high"], 10))


def turtle_s1(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    """20日ブレイクアウト。ただし「直前のブレイクアウトが（仮に取引していたら）勝ちだった」場合は見送る。
    直前のブレイクアウトの仮の取引は、次の足の始値で入り、2N の損切りか、10日の反対側のブレイクで終わるとみなす。
    仮の取引が続いている間のブレイクは、新しいブレイクアウトとして数えない（すでにポジションを持っている状態のため）。
    すべてその足までのデータで判断する。"""
    buy, sell = _turtle_entries(d, 20)
    o, h, l, c = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n_atr = d["atr"].to_numpy(dtype=float)
    lo10, hi10 = _ll(d["low"], 10).to_numpy(), _hh(d["high"], 10).to_numpy()
    raw = np.where(buy.to_numpy(), 1, np.where(sell.to_numpy(), -1, 0))
    out = np.zeros(len(d), dtype=int)
    pend = None          # 直前のブレイクアウトの仮の取引：[方向, シグナルの足, エントリー価格, 損切り]
    last_win = None      # 直前のブレイクアウトの仮の取引の結果（決まっていなければ None）
    for i in range(len(d)):
        if pend is not None and i > pend[1]:
            dr, si = pend[0], pend[1]
            if pend[2] is None:  # シグナルの次の足：始値で入る
                pend[2] = o[i]
                pend[3] = o[i] - dr * 2 * n_atr[si]
            entry, stop = pend[2], pend[3]
            if (dr > 0 and l[i] <= stop) or (dr < 0 and h[i] >= stop):
                last_win, pend = False, None
            elif (dr > 0 and c[i] < lo10[i]) or (dr < 0 and c[i] > hi10[i]):
                # bool() にする：numpy の真偽値は「is True」で比べると常に False になるため
                last_win, pend = bool(dr * (c[i] - entry) > 0), None
        # 仮の取引が続いている間のブレイク（同じトレンドの中での更新）は、新しいブレイクアウトとして数えない
        if raw[i] != 0 and pend is None:
            out[i] = 0 if last_win is True else raw[i]
            pend, last_win = [int(raw[i]), i, None, None], None
    return pd.Series(out, index=d.index, dtype=int)


# ---------------- ドンチャンの4週ルール ----------------

def donchian_4week(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    return _signal(_first(d["close"] > _hh(d["high"], 20)), _first(d["close"] < _ll(d["low"], 20)))


# ---------------- ワイルダーのパラボリック SAR ----------------

def parabolic_sar(h: np.ndarray, l: np.ndarray, af0: float = 0.02, step: float = 0.02, af_max: float = 0.2):
    """パラボリック SAR と、その足での向き（+1 上昇・−1 下降）。その足までのデータだけで計算する。"""
    n = len(h)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    if n < 2:
        return sar, trend
    up = h[1] >= h[0]
    s = l[0] if up else h[0]
    ep = h[0] if up else l[0]
    af = af0
    trend[0] = 1 if up else -1
    sar[0] = s
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < s:                      # 反転して下降へ
                up, s, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + step, af_max)
        else:
            s = max(s, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > s:                      # 反転して上昇へ
                up, s, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + step, af_max)
        sar[i] = s
        trend[i] = 1 if up else -1
    return sar, trend


def sar_reversal(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    _, trend = parabolic_sar(d["high"].to_numpy(dtype=float), d["low"].to_numpy(dtype=float))
    t = pd.Series(trend, index=d.index)
    flip = t != t.shift()
    flip.iloc[0] = False
    return _signal(flip & (t > 0), flip & (t < 0))


# ---------------- ボリンジャーのスクイーズ（Method I） ----------------

def bollinger_squeeze(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    """バンドの幅が直近125日（約半年）で最も狭くなった（スクイーズ）あと20日以内に、終値がバンドの外に出たら、その方向へ。"""
    width = (d["bb_up"] - d["bb_lo"]) / d["bb_mid"]
    squeeze = width <= width.rolling(125).min()
    recent = squeeze.fillna(False).astype(float).rolling(20, min_periods=1).max().astype(bool)
    buy = _first(d["close"] > d["bb_up"]) & recent
    sell = _first(d["close"] < d["bb_lo"]) & recent
    return _signal(buy, sell)


def bollinger_squeeze_exit(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    """反対側のバンドに届いたら決済。"""
    return _exit_series(d["close"] < d["bb_lo"], d["close"] > d["bb_up"])


# ---------------- ワイルダーの DMI（方向性指数） ----------------

def dmi(d: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    h, l, c = d["high"], d["low"], d["close"]
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=d.index)
    tr = indicators.true_range(d)
    atr = indicators.wilder(tr, n)
    pdi = 100 * indicators.wilder(plus_dm, n) / atr
    mdi = 100 * indicators.wilder(minus_dm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = indicators.wilder(dx, n)
    adxr = (adx + adx.shift(n)) / 2
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": adx, "adxr": adxr})


def dmi_cross(d: pd.DataFrame, p: IndicatorParams) -> pd.Series:
    """+DI が −DI を上抜けで買い、下抜けで売り。ただし ADXR が25より大きい（トレンドがある）ときだけ。"""
    m = dmi(d)
    strong = m["adxr"] > 25
    up = (m["pdi"] > m["mdi"]) & (m["pdi"].shift() <= m["mdi"].shift())
    down = (m["pdi"] < m["mdi"]) & (m["pdi"].shift() >= m["mdi"].shift())
    return _signal(up & strong, down & strong)


# ---------------- 一覧 ----------------

FAMOUS = [
    Strategy(
        "turtle_s1", "タートルズ（システム1）", "順張り",
        "終値が直前20日の最高値を超えたら買い、最安値を割ったら売り。直前のブレイクアウトが勝ちだった場合は見送る。"
        "直前10日の反対側（買いなら最安値）を割ったら決済。損切りは 2N（N＝20日のATR）。利確はしない。",
        turtle_s1, family="famous", defaults={"atr_period": 20}, exit_func=turtle_s1_exit,
        exits={"sl_atr": 2.0, "tp_atr": None, "max_bars": 0, "opposite_exit": True},
        source="Richard Dennis と William Eckhardt が1983年に始めた「タートルズ」の売買ルール（のちに元参加者が書籍などで公開）",
        spec={"rule": "turtle_system1_original", "entry": 20, "exit": 10, "stop_n": 2, "atr": 20, "skip_after_winner": True,
              "simplifications": ["ブレイクは足の途中の価格ではなく終値で判定し、次の足の始値で入る",
                                  "ユニットを買い増す（ピラミッディング）ルールは使わない",
                                  "見送ったときの55日ブレイクアウトでの入り直しは使わない",
                                  "数量（1N＝残高の1%）はこのツールの資金管理を使う"]}),
    Strategy(
        "turtle_s2", "タートルズ（システム2）", "順張り",
        "終値が直前55日の最高値を超えたら買い、最安値を割ったら売り。直前20日の反対側を割ったら決済。"
        "損切りは 2N（N＝20日のATR）。利確はしない。",
        turtle_s2, family="famous", defaults={"atr_period": 20}, exit_func=turtle_s2_exit,
        exits={"sl_atr": 2.0, "tp_atr": None, "max_bars": 0, "opposite_exit": True},
        source="同上（タートルズの長期のシステム）",
        spec={"rule": "turtle_system2_original", "entry": 55, "exit": 20, "stop_n": 2, "atr": 20,
              "simplifications": ["ブレイクは終値で判定し、次の足の始値で入る", "ピラミッディングは使わない",
                                  "数量はこのツールの資金管理を使う"]}),
    Strategy(
        "donchian_4week", "ドンチャンの4週ルール", "順張り",
        "終値が直前4週間（20日）の最高値を超えたら買い、最安値を割ったら売り。反対のシグナルで決済して乗り換える（ドテン）。"
        "損切り・利確は使わない。",
        donchian_4week, family="famous", exits={"sl_atr": None, "tp_atr": None, "max_bars": 0, "opposite_exit": True},
        source="Richard Donchian（トレンドフォローの先駆者）の「4週ルール」。チャネルの考え方は1960年代から知られる",
        spec={"rule": "donchian_4week_original", "period": 20,
              "simplifications": ["ブレイクは終値で判定し、次の足の始値で入る"]}),
    Strategy(
        "parabolic_sar", "パラボリックSAR（ワイルダー）", "順張り",
        "パラボリックSARの向きが上向きに変わったら買い、下向きに変わったら売り。常にどちらかのポジションを持つ（ドテン）。"
        "加速係数は 0.02 から 0.02 ずつ、最大 0.2。",
        sar_reversal, family="famous", exits={"sl_atr": None, "tp_atr": None, "max_bars": 0, "opposite_exit": True},
        source="J. Welles Wilder Jr.『New Concepts in Technical Trading Systems』（1978年）",
        spec={"rule": "parabolic_sar_original", "af0": 0.02, "step": 0.02, "af_max": 0.2,
              "simplifications": ["本来は足の途中でSARの価格に届いた瞬間に乗り換えるが、ここでは次の足の始値で乗り換える"]}),
    Strategy(
        "bollinger_squeeze", "ボリンジャーのスクイーズ（Method I）", "順張り",
        "バンドの幅が直近半年（125日）で最も狭くなった（スクイーズ）あと、終値がバンドの外に出たらその方向へ。"
        "反対側のバンドに届いたら決済。損切り・利確は使わない。バンドは20日・±2σ。",
        bollinger_squeeze, family="famous", exit_func=bollinger_squeeze_exit,
        exits={"sl_atr": None, "tp_atr": None, "max_bars": 0, "opposite_exit": True},
        source="John Bollinger『Bollinger on Bollinger Bands』（2001年）で紹介された使い方",
        spec={"rule": "bollinger_method1_original", "period": 20, "std": 2.0, "squeeze_lookback": 125, "after_squeeze": 20,
              "simplifications": ["「スクイーズのあと」は20日以内と決めた（元の説明では数値は決まっていない）",
                                  "決済は「反対側のバンド」を使った（元の説明ではパラボリックSARなどの候補もある）"]}),
    Strategy(
        "dmi_cross", "DMIのクロス（ワイルダー）", "順張り",
        "+DI が −DI を上抜けで買い、下抜けで売り。ADXR が25より大きい（トレンドがある）ときだけ入る。反対のクロスで乗り換える。",
        dmi_cross, family="famous", exits={"sl_atr": None, "tp_atr": None, "max_bars": 0, "opposite_exit": True},
        source="J. Welles Wilder Jr.『New Concepts in Technical Trading Systems』（1978年）",
        spec={"rule": "dmi_original", "period": 14, "adxr_min": 25,
              "simplifications": ["クロスした日の高値・安値を損切りに使う「エクストリーム・ポイント・ルール」は使わず、反対のクロスで乗り換える"]}),
]
FAMOUS_BY_KEY = {s.key: s for s in FAMOUS}
