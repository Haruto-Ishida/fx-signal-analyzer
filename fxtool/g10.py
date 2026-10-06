"""G10 通貨の仮説の検証（事前登録：preregistration/g10_hypotheses.json・計画：preregistration/g10_plan.md）。

研究で効果が報告されている5つの仮説を、ルールを1つずつ事前に固定して、月1回の入れ替えで確かめる。
- 10通貨（USD と9通貨）。損益は「9通貨それぞれを米ドルに対して持つ」ポジションの合計で、月ごとに計算する。
- 月末（FRED の H.10 為替レート＝ニューヨークの正午）に、その時点で使える情報だけでポジションを決め、翌月末まで持つ。
- 損益 ＝ 為替の動き ＋ 金利差（スワップ。個人向け FX 会社の不利な分を割り引く）− 入れ替えのコスト。
- 金利・物価は発表の遅れを入れて使う（金利：前の月、物価：月ごとは2か月前、3か月ごと（AUD・NZD）は四半期が終わって3か月後の月末から）。
- 開発用の期間（2012-01〜2020-09）のあいだは、2020-10-14 より後のデータを使わない（DEV_CUTOFF。最終テスト期間の共通の境界）。

この中の計算は、事前登録の指紋（fingerprint）で固定している。検証が終わるまで変更しない。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import overfit

CURRENCIES = ["EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"]   # 米ドルに対して持つ9通貨
ALL = ["USD"] + CURRENCIES
DEV_CUTOFF = pd.Timestamp("2020-10-14")           # 最終テスト期間の共通の境界（2020-10-15〜）の前日
DEV_FIRST_RETURN, DEV_LAST_RETURN = pd.Period("2012-01", "M"), pd.Period("2020-09", "M")

# FRED の系列（為替：USD per 通貨 にそろえる。invert=True は「通貨 per USD」で発表されているもの）
FX_SERIES = {"EUR": ("DEXUSEU", False), "JPY": ("DEXJPUS", True), "GBP": ("DEXUSUK", False), "CHF": ("DEXSZUS", True),
             "AUD": ("DEXUSAL", False), "NZD": ("DEXUSNZ", False), "CAD": ("DEXCAUS", True), "SEK": ("DEXSDUS", True),
             "NOK": ("DEXNOUS", True)}
RATE_SERIES = {c: f"IR3TIB01{cc}M156N" for c, cc in
               {"USD": "US", "EUR": "EZ", "JPY": "JP", "GBP": "GB", "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA",
                "SEK": "SE", "NOK": "NO"}.items()}
CPI_SERIES = {"USD": ("CPIAUCNS", "M"), "EUR": ("CP0000EZ19M086NEST", "M"), "JPY": ("JPNCPIALLMINMEI", "M"),
              "GBP": ("GBRCPIALLMINMEI", "M"), "CHF": ("CHECPIALLMINMEI", "M"), "AUD": ("AUSCPIALLQINMEI", "Q"),
              "NZD": ("NZLCPIALLQINMEI", "Q"), "CAD": ("CANCPIALLMINMEI", "M"), "SEK": ("SWECPIALLMINMEI", "M"),
              "NOK": ("NORCPIALLMINMEI", "M")}
RATE_LAG = 1          # 金利：ポジションを決める月の、前の月の平均を使う
CPI_LAG_MONTHLY = 2   # 物価（月ごと）：2か月前の値を使う
CPI_LAG_QUARTERLY = 3  # 物価（3か月ごと）：四半期の最後の月から3か月たった月末から使う

HYPOTHESES = ["H1", "H2", "H3", "H4", "H5"]


# ---------------- データの形をそろえる ----------------

def usd_per_unit(fx_daily: pd.DataFrame) -> pd.DataFrame:
    """FRED の為替（列＝系列の ID）を、「1通貨あたり何米ドルか」（列＝通貨）にそろえる。"""
    out = {}
    for cur, (sid, inv) in FX_SERIES.items():
        s = fx_daily[sid].astype(float)
        out[cur] = 1.0 / s if inv else s
    return pd.DataFrame(out).sort_index()


def month_end(daily: pd.DataFrame) -> pd.DataFrame:
    """各月の最後の値（その月に値がなければ欠損）。index は月（Period）。"""
    d = daily.dropna(how="all")
    return d.groupby(d.index.to_period("M")).last()


def monthly(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(float)
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    return s.groupby(level=0).last()


def quarterly_available(q: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """3か月ごとの値（FRED は四半期の最初の日付で表す）を、「四半期の最後の月から CPI_LAG_QUARTERLY か月たった月末」から使えるとして、
    月ごとの値（次が使えるようになるまで同じ値）にする。"""
    q = q.dropna().astype(float)
    avail = {pd.Timestamp(t).to_period("M") + 2 + CPI_LAG_QUARTERLY: v for t, v in q.items()}
    s = pd.Series(avail).sort_index()
    return s.reindex(months.union(s.index)).ffill().reindex(months)


@dataclass
class Panel:
    months: pd.PeriodIndex            # ポジションを決める月（月末）
    price: pd.DataFrame               # 月末の為替（USD per 通貨）
    spot_next: pd.DataFrame           # その月末から翌月末までの為替の動き（米ドルに対してその通貨を持ったとき）
    rate_signal: pd.DataFrame         # ポジションを決めるときに使える金利（%、前の月の平均）。列は ALL
    rate_next: pd.DataFrame           # 翌月の実際の金利の平均（%）。金利差の受け取り・支払いの計算用。列は ALL
    cpi: pd.DataFrame                 # ポジションを決めるときに使える物価（発表の遅れ込み）。列は ALL
    vol: pd.DataFrame                 # 月末までの直近63営業日の為替の値動きの大きさ（年率）
    excess_past: pd.DataFrame         # その月までの1か月ごとの「為替＋金利差（銀行間）」の損益（シグナル用）


def build_panel(fx_daily: pd.DataFrame, rates: dict[str, pd.Series], cpi: dict[str, pd.Series],
                cutoff: pd.Timestamp | None = DEV_CUTOFF) -> Panel:
    """FRED のデータから、月ごとの表を作る。cutoff より後のデータは使わない（開発用の期間の検証では DEV_CUTOFF）。"""
    if cutoff is not None:
        fx_daily = fx_daily[fx_daily.index <= cutoff]
        rates = {k: v[v.index <= cutoff] for k, v in rates.items()}
        cpi = {k: v[v.index <= cutoff] for k, v in cpi.items()}
    p_daily = usd_per_unit(fx_daily)
    price = month_end(p_daily)
    months = price.index
    spot_next = price.shift(-1) / price - 1
    r = pd.DataFrame({c: monthly(rates[c]) for c in ALL}).reindex(months)
    rate_signal = r.shift(RATE_LAG)
    rate_next = r.shift(-1)
    cols = {}
    for c in ALL:
        sid, freq = CPI_SERIES[c]
        if freq == "Q":
            cols[c] = quarterly_available(cpi[c], months)
        else:
            cols[c] = monthly(cpi[c]).reindex(months).shift(CPI_LAG_MONTHLY)
    cpi_m = pd.DataFrame(cols)
    logret = np.log(p_daily).diff()
    vol = logret.rolling(63, min_periods=50).std().mul(math.sqrt(252))
    vol = vol.groupby(vol.index.to_period("M")).last().reindex(months)
    carry_past = (r[CURRENCIES].sub(r["USD"], axis=0) / 100 / 12).shift(1)   # 前の月に持っていたときに受け取った金利差
    excess_past = (price / price.shift(1) - 1) + carry_past
    return Panel(months, price, spot_next, rate_signal, rate_next, cpi_m, vol, excess_past)


# ---------------- 仮説ごとのポジション（9通貨の、米ドルに対する重み） ----------------

def _rank_weights(score: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """10通貨（USD を含む）を score の高い順に並べ、上位 n を買い・下位 n を売り（片側の合計 0.5 ずつ）。
    USD が選ばれた場合、その分は米ドルを持つこと（9通貨の重みには現れない）。score に欠損がある月は持たない。"""
    w = pd.DataFrame(0.0, index=score.index, columns=CURRENCIES)
    for m, row in score.iterrows():
        if row.isna().any():
            continue
        order = row.sort_values(ascending=False, kind="stable").index
        for c in order[:n]:
            if c != "USD":
                w.at[m, c] += 0.5 / n
        for c in order[-n:]:
            if c != "USD":
                w.at[m, c] -= 0.5 / n
    return w


def weights_h1(pn: Panel) -> pd.DataFrame:
    """時系列モメンタム：過去12か月の「為替＋金利差」がプラスなら買い、マイナスなら売り（9通貨 同じ重み 1/9）。"""
    past12 = (1 + pn.excess_past).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1
    return np.sign(past12).fillna(0.0) / len(CURRENCIES)


def weights_h2(pn: Panel) -> pd.DataFrame:
    """キャリー：金利（前の月の平均）の高い3通貨を買い、低い3通貨を売り。"""
    return _rank_weights(pn.rate_signal[ALL])


def weights_h3(pn: Panel) -> pd.DataFrame:
    """クロスセクションのモメンタム：過去3か月の「為替＋金利差」が大きい3通貨を買い、小さい3通貨を売り（USD は 0）。"""
    past3 = (1 + pn.excess_past).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1
    score = past3.copy()
    score.insert(0, "USD", 0.0)
    score.loc[past3.isna().any(axis=1), "USD"] = np.nan
    return _rank_weights(score[ALL])


def weights_h4(pn: Panel) -> pd.DataFrame:
    """バリュー：実質為替レート（USD per 通貨 × 通貨の物価 ÷ 米国の物価）の5年前（60か月前）からの変化が小さい（割安になった）
    3通貨を買い、大きい3通貨を売り（USD は 0）。物価は発表の遅れ込み。"""
    rer = np.log(pn.price[CURRENCIES]) + np.log(pn.cpi[CURRENCIES]).sub(np.log(pn.cpi["USD"]), axis=0)
    score = -(rer - rer.shift(60))
    score.insert(0, "USD", 0.0)
    score.loc[score[CURRENCIES].isna().any(axis=1), "USD"] = np.nan
    return _rank_weights(score[ALL])


VOL_TARGET = 0.10    # 1通貨あたりの値動きの目安（年率）
VOL_CAP = 2.0        # 同じ重みの何倍まで


def weights_h5(pn: Panel) -> pd.DataFrame:
    """H1 と同じ方向で、量を値動きの大きさの逆数に合わせる（VOL_TARGET ÷ 値動き × 1/9、同じ重みの2倍まで）。"""
    s = np.sign(weights_h1(pn))
    scale = (VOL_TARGET / pn.vol[CURRENCIES]).clip(upper=VOL_CAP)
    return (s * scale / len(CURRENCIES)).fillna(0.0)


WEIGHT_FUNCS = {"H1": weights_h1, "H2": weights_h2, "H3": weights_h3, "H4": weights_h4, "H5": weights_h5}


# ---------------- 損益 ----------------

@dataclass(frozen=True)
class CostModel:
    one_way: dict                 # 1単位を売買したときのコスト（取引額に対する割合）
    swap_haircut: float = 0.005   # 年率。受け取る側は金利差から差し引き、支払う側は上乗せ


def monthly_returns(w: pd.DataFrame, pn: Panel, cm: CostModel) -> pd.DataFrame:
    """月末 m に決めた重み w で、翌月の損益を計算する。戻り値の index は損益が出た月（m+1）。
    列：spot（為替）・carry（金利差、割り引き込み）・cost（入れ替えのコスト）・total・gross（重みの絶対値の合計）。"""
    diff = pn.rate_next[CURRENCIES].sub(pn.rate_next["USD"], axis=0) / 100
    spot = (w * pn.spot_next[CURRENCIES]).sum(axis=1)
    carry = (w * diff / 12).sum(axis=1) - w.abs().sum(axis=1) * cm.swap_haircut / 12
    turnover = (w - w.shift(1).fillna(0.0)).abs()
    cost = (turnover * pd.Series(cm.one_way)).sum(axis=1)
    out = pd.DataFrame({"spot": spot, "carry": carry, "cost": -cost, "gross": w.abs().sum(axis=1)})
    out["total"] = out["spot"] + out["carry"] + out["cost"]
    out.index = out.index + 1
    return out


def dev_window(ret: pd.DataFrame) -> pd.DataFrame:
    return ret[(ret.index >= DEV_FIRST_RETURN) & (ret.index <= DEV_LAST_RETURN)]


# ---------------- 比べる基準（ランダム）と成績 ----------------

def random_totals(w: pd.DataFrame, pn: Panel, cm: CostModel, months: pd.PeriodIndex, runs: int = 1000,
                  seed: int = 0, ranked: bool = False) -> np.ndarray:
    """同じ形のでたらめなポジションの損益合計（runs 通り）。
    ranked=False（H1・H5）：各通貨の重みの大きさはそのままで、方向（買い・売り）をでたらめにする。
    ranked=True（H2〜H4）：毎月、10通貨からでたらめに3つ買い・3つ売りを選ぶ。"""
    rng = np.random.default_rng(seed)
    out = np.empty(runs)
    dec = w.index
    for k in range(runs):
        if ranked:
            score = pd.DataFrame(rng.random((len(dec), len(ALL))), index=dec, columns=ALL)
            score[w.abs().sum(axis=1) == 0] = np.nan
            wr = _rank_weights(score)
        else:
            wr = w.abs() * rng.choice([-1.0, 1.0], size=w.shape)
        r = monthly_returns(wr, pn, cm)
        out[k] = r.loc[r.index.isin(months), "total"].sum()
    return out


def max_drawdown(x: np.ndarray) -> float:
    eq = np.cumsum(x)
    return float(min(0.0, (eq - np.maximum.accumulate(np.maximum(eq, 0))).min())) if len(x) else 0.0


@dataclass
class Result:
    name: str
    returns: pd.DataFrame
    n_months_active: int
    total: float
    ann_return: float
    ann_vol: float
    sharpe: float                  # 年率（月ごとの平均 ÷ ばらつき × √12）
    max_dd: float
    yearly: pd.Series
    random_beat: float = math.nan
    psr_single: float = math.nan
    dsr_plan: float = math.nan
    dsr_tool: float = math.nan
    checks: list = field(default_factory=list)
    verdict: str = ""


def evaluate(name: str, ret: pd.DataFrame) -> Result:
    x = ret["total"].to_numpy(dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    sr = float(x.mean() / sd * math.sqrt(12)) if sd > 0 else 0.0
    yearly = ret["total"].groupby(ret.index.year).sum()
    return Result(name, ret, int((ret["gross"] > 0).sum()), float(x.sum()), float(x.mean() * 12), float(sd * math.sqrt(12)),
                  sr, max_drawdown(x), yearly)


def judge(r: Result, prob: float, criteria: dict) -> tuple[str, list[str]]:
    """事前登録の合格基準（H1〜H4）。prob は偶然ではない確率（主な判定は、この計画の5つで割り引いた DSR）。"""
    worst = float(r.yearly.min()) if len(r.yearly) else 0.0
    extreme = r.total > 0 and -worst > criteria["extreme_year_share"] * r.total
    checks = [
        (r.n_months_active >= criteria["min_months"], f"ポジションのあった月 {r.n_months_active}（{criteria['min_months']}以上）"),
        (r.total > 0, f"コスト後の損益 {r.total:+.2%}（プラス）"),
        (r.random_beat >= criteria["min_random_beat"], f"でたらめなポジションを上回った割合 {r.random_beat:.0%}（{criteria['min_random_beat']:.0%}以上）"),
        ((r.yearly > 0).mean() >= criteria["min_positive_year_ratio"],
         f"黒字の年 {(r.yearly > 0).sum()}/{len(r.yearly)}（{criteria['min_positive_year_ratio']:.0%}以上）"),
        (not extreme, f"いちばん悪い年 {worst:+.2%}（利益合計の{criteria['extreme_year_share']:.0%}を超える損失がない）"),
        (-r.max_dd <= criteria["max_dd_vol_multiple"] * r.ann_vol,
         f"最大ドローダウン {r.max_dd:.2%}（年間の値動き {r.ann_vol:.2%} の{criteria['max_dd_vol_multiple']}倍以内）"),
        (not math.isnan(prob) and prob >= criteria["min_probability"], f"偶然ではない確率 {prob:.0%}（{criteria['min_probability']:.0%}以上）"),
    ]
    return ("合格" if all(ok for ok, _ in checks) else "不合格"), [("✅ " if ok else "❌ ") + m for ok, m in checks]


def non_dsr_ok(checks: list[str]) -> bool:
    """DSR（最後の条件）以外をすべて満たしたか（最終テストに進めるか）。"""
    return all(c.startswith("✅") for c in checks[:-1])


def sharpe_diff_p(a: np.ndarray, b: np.ndarray, runs: int = 5000, seed: int = 0) -> float:
    """H5 用：月をまとめて復元抽出し、シャープレシオの差（a − b）が 0 以下になる割合（片側のブートストラップ）。"""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(runs, n))
    def sr(x):
        sd = x.std(axis=1, ddof=1)
        return np.where(sd > 0, x.mean(axis=1) / sd, 0.0)
    d = sr(a[idx]) - sr(b[idx])
    return float((d <= 0).mean())


def run_all(pn: Panel, cm: CostModel, criteria: dict, n_trials_tool: int, months_filter=dev_window,
            random_runs: int = 1000) -> dict[str, Result]:
    """5つの仮説を、それぞれ1回だけ計算する。判定：主な判定は「この計画の5つで割り引いた DSR」、参考に「1つだけ」と「ツール全体」。"""
    res = {}
    for h in HYPOTHESES:
        w = WEIGHT_FUNCS[h](pn)
        ret = months_filter(monthly_returns(w, pn, cm))
        r = evaluate(h, ret)
        r.random_beat = float((random_totals(w, pn, cm, ret.index, random_runs, seed=HYPOTHESES.index(h),
                                             ranked=h in ("H2", "H3", "H4")) < r.total).mean())
        res[h] = r
    srs = np.array([res[h].sharpe / math.sqrt(12) for h in HYPOTHESES])   # 月ごとのシャープレシオ
    var = float(srs.var(ddof=1))
    for h in HYPOTHESES:
        x = res[h].returns["total"].to_numpy(dtype=float)
        res[h].psr_single = overfit.probabilistic_sharpe(overfit.sharpe(x), 0.0, x)
        res[h].dsr_plan = overfit.deflated_sharpe(x, len(HYPOTHESES), var)["dsr"]
        res[h].dsr_tool = overfit.deflated_sharpe(x, n_trials_tool, var)["dsr"]
        if h != "H5":
            res[h].verdict, res[h].checks = judge(res[h], res[h].dsr_plan, criteria)
    # H5：H1 より良くなったか（シャープレシオの差が偶然とは言いにくく、最大ドローダウンも小さい）
    a, b = res["H5"].returns["total"].to_numpy(dtype=float), res["H1"].returns["total"].to_numpy(dtype=float)
    p = sharpe_diff_p(a, b)
    ok1 = res["H5"].sharpe > res["H1"].sharpe and p < criteria["h5_p_value"]
    ok2 = res["H5"].max_dd > res["H1"].max_dd
    res["H5"].checks = [("✅ " if ok1 else "❌ ") + f"シャープレシオ {res['H5'].sharpe:.2f}（H1 {res['H1'].sharpe:.2f}）・p = {p:.2f}（{criteria['h5_p_value']}未満）",
                        ("✅ " if ok2 else "❌ ") + f"最大ドローダウン {res['H5'].max_dd:.2%}（H1 {res['H1'].max_dd:.2%} より小さい）"]
    res["H5"].verdict = "合格" if ok1 and ok2 else "不合格"
    return res


# ---------------- 計算の指紋 ----------------

def synthetic_inputs(seed: int = 20261007):
    """指紋とテスト用の合成データ（FRED と同じ形）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2005-01-03", "2020-10-14")
    fx = {}
    for cur, (sid, inv) in FX_SERIES.items():
        lvl = np.exp(np.cumsum(rng.normal(0, 0.006, len(days))))
        fx[sid] = (1 / lvl) if inv else lvl
    fx_daily = pd.DataFrame(fx, index=days)
    fx_daily.iloc[::37] = np.nan                         # 祝日などの欠け
    mi = pd.date_range("2004-01-01", "2020-10-01", freq="MS")
    rates = {c: pd.Series(2 + np.cumsum(rng.normal(0, 0.1, len(mi))), index=mi) for c in ALL}
    qi = pd.date_range("2004-01-01", "2020-07-01", freq="QS")
    cpi = {}
    for c in ALL:
        idx = qi if CPI_SERIES[c][1] == "Q" else mi
        cpi[c] = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.002, 0.003, len(idx)))), index=idx)
    return fx_daily, rates, cpi


def fingerprint(cm: CostModel) -> str:
    fx, rates, cpi = synthetic_inputs()
    pn = build_panel(fx, rates, cpi)
    payload = []
    for h in HYPOTHESES:
        r = dev_window(monthly_returns(WEIGHT_FUNCS[h](pn), pn, cm))
        payload.append([h, [round(v, 10) for v in r["total"].tolist()]])
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
