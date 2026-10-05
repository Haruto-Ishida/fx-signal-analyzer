"""口座全体のシミュレーション（ペーパートレード・積立のシミュレーション。要件定義書 5-6・7章）。

取引の一覧を時間順に、仮想の口座（円）で売買したとみなして残高を計算する。
- 数量は、エントリーした時点の残高（決済済みの損益だけを反映）から、1回の損失の目安（risk_pct）で決める
- 実効レバレッジ（持っているポジションの合計 ÷ 残高）の上限を超える分は減らす。上限で1単位も持てなければ見送る
- 毎月の追加入金（積立）は、月が変わった最初の時点で残高に加える
- 成績は2つに分ける：
  - 運用成績：入金の影響を除いた時間加重リターン（TWR）、シャープレシオ、最大ドローダウン → 戦略の評価に使う
  - 資産推移：入金の累計と、運用による増減（色分けして表示する）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import overfit, sizing

JST = pd.Timedelta(hours=9)


@dataclass
class AccountResult:
    trades: pd.DataFrame      # 取引ごとの数量・損益（円）・見送った理由
    daily: pd.DataFrame       # 日ごと（日本時間の日付）：balance, deposits_cum, pnl_cum, twr_index
    metrics: dict             # initial, final, deposits, profit, twr, twr_annual, sharpe, max_dd, n, n_skipped, bankrupt


def simulate(trades: pd.DataFrame, ticker: str, pip: float, risk_pct: float, initial: float,
             monthly_deposit: float = 0.0, max_leverage: float | None = 5.0, usdjpy: float | None = None) -> AccountResult:
    """trades の列：entry_time, exit_time, dir, entry, sl, pips（コスト後）。時刻は UTC。
    複数の通貨ペアをまとめるときは、trades に ticker・pip の列を入れる（その行の値を使う）。"""
    t = trades.sort_values("entry_time", kind="stable").reset_index(drop=True)
    tickers = t["ticker"].tolist() if "ticker" in t.columns else [ticker] * len(t)
    pips_ = t["pip"].to_numpy(dtype=float) if "pip" in t.columns else np.full(len(t), pip)
    for tk in dict.fromkeys(tickers or [ticker]):
        if sizing.quote_to_jpy(tk, usdjpy) is None:
            raise ValueError(f"{tk} の損益を円に換算できません（USD/JPY のレートが必要です）")
    convs = np.array([sizing.quote_to_jpy(tk, usdjpy) for tk in tickers], dtype=float)
    if t.empty:
        return AccountResult(t, pd.DataFrame(), _empty_metrics(initial))

    # 出来事を時間順に並べる（同じ時刻なら、入金 → 決済 → エントリー の順）
    start = pd.Timestamp(t["entry_time"].min())
    end = pd.Timestamp(t["exit_time"].max())
    months = pd.date_range((start + JST).normalize().replace(day=1) + pd.offsets.MonthBegin(1), end + JST, freq="MS") - JST
    ev = [(m, 0, "deposit", -1) for m in months if monthly_deposit > 0]
    ev += [(row.exit_time, 1, "exit", i) for i, row in t.iterrows()]
    ev += [(row.entry_time, 2, "entry", i) for i, row in t.iterrows()]
    ev.sort(key=lambda x: (x[0], x[1], x[3]))

    balance, deposits = float(initial), 0.0
    units = np.zeros(len(t))
    pnl = np.full(len(t), np.nan)
    risk_jpy = np.full(len(t), np.nan)
    note = [""] * len(t)
    open_notional: dict[int, float] = {}
    flows = []          # (時刻, 残高, その時点までの入金の累計)
    bankrupt = False
    for time, _, kind, i in ev:
        if kind == "deposit":
            balance += monthly_deposit
            deposits += monthly_deposit
        elif kind == "entry":
            if bankrupt:
                note[i] = "残高がなくなったため見送り"
                continue
            row = t.iloc[i]
            sz = sizing.position_size(balance, risk_pct, row.entry, row.sl, tickers[i], usdjpy, max_leverage,
                                      sum(open_notional.values()))
            if sz is None or sz.units == 0:
                note[i] = "見送り：" + (sz.note if sz else "数量を計算できない")
                continue
            units[i], risk_jpy[i], note[i] = sz.units, sz.loss_at_stop_jpy, sz.note
            open_notional[i] = sizing.notional_jpy(sz.units, row.entry, tickers[i], usdjpy)
        else:  # exit
            if units[i] == 0:
                continue
            pnl[i] = units[i] * t["pips"].iat[i] * pips_[i] * convs[i]
            balance += pnl[i]
            open_notional.pop(i, None)
            if balance <= 0:
                bankrupt = True
        flows.append((time, balance, deposits))

    out = t.assign(units=units, pnl_jpy=pnl, loss_at_stop_jpy=risk_jpy, note=note)
    daily = _daily(flows, initial, start)
    return AccountResult(out, daily, _metrics(daily, initial, deposits, out, bankrupt))


def _daily(flows, initial: float, start: pd.Timestamp) -> pd.DataFrame:
    """日本時間の日付ごとの、その日の終わりの残高と入金の累計。"""
    f = pd.DataFrame(flows, columns=["time", "balance", "deposits_cum"])
    f["day"] = (pd.DatetimeIndex(f["time"]) + JST).normalize()
    d = f.groupby("day")[["balance", "deposits_cum"]].last()
    first = (start + JST).normalize() - pd.Timedelta(days=1)
    d = pd.concat([pd.DataFrame({"balance": [initial], "deposits_cum": [0.0]}, index=[first]), d])
    d.index.name = "day"
    d = d[~d.index.duplicated(keep="last")]
    d["pnl_cum"] = d["balance"] - initial - d["deposits_cum"]
    # 時間加重リターン：その日の入金を除いた増え方を、日ごとにかけ合わせる
    prev = d["balance"].shift()
    dep = d["deposits_cum"].diff().fillna(0.0)
    r = ((d["balance"] - dep) / prev - 1).fillna(0.0)
    d["ret"] = r
    d["twr_index"] = (1 + r).cumprod()
    return d


def _empty_metrics(initial: float) -> dict:
    return dict(initial=initial, final=initial, deposits=0.0, profit=0.0, twr=0.0, twr_annual=math.nan,
                sharpe=math.nan, max_dd=0.0, n=0, n_skipped=0, bankrupt=False)


def _metrics(d: pd.DataFrame, initial: float, deposits: float, trades: pd.DataFrame, bankrupt: bool) -> dict:
    twr = float(d["twr_index"].iat[-1] - 1)
    years = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    idx = d["twr_index"]
    max_dd = float((idx / idx.cummax() - 1).min())
    rets = d["ret"].reindex(overfit.business_days(d.index[0], d.index[-1]), fill_value=0.0)
    final = float(d["balance"].iat[-1])
    return dict(
        initial=initial, final=final, deposits=deposits, profit=final - initial - deposits,
        twr=twr, twr_annual=(1 + twr) ** (1 / years) - 1 if twr > -1 else -1.0,
        sharpe=overfit.annualized(overfit.sharpe(rets.to_numpy())), max_dd=max_dd,
        n=int((trades["units"] > 0).sum()), n_skipped=int((trades["units"] == 0).sum()), bankrupt=bankrupt,
    )
