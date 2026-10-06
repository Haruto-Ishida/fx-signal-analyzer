"""Discord への通知（要件定義書 10章・ユーザーの決定 8）。

ユーザーの決定：送るのは「シグナル」と「動作状況」だけ。個人情報や口座に関する情報（残高・数量・損失額（円）・
ペーパー口座の損益など）は送らない。
- 送る前に必ず check_safe() で確かめ、金額（円）・残高・口座・数量・メールアドレスなどが含まれていれば送らない。
- Webhook の URL は .env から読む（git には入れない）。.env の NOTIFY_ENABLED=true のときだけ送る。
- すべての通知に「参考情報であり、利益を保証しない」旨を付ける。
- 同じ内容（同じキー）は二度送らない（notifications に記録する）。
"""
from __future__ import annotations

import json
import re
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PAIR_NAMES, ROOT, TIMEFRAMES
from .tracking import JST, ts_str

ENV_FILE = ROOT / ".env"
DISCLAIMER = "※参考情報です。利益を保証するものではありません（このツールは注文をしません）。"
MAX_LEN = 1900  # Discord の上限は 2000 文字

SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    key TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL, sent_at TEXT,
    text TEXT NOT NULL, status TEXT NOT NULL, error TEXT
);
"""

# 送ってはいけない内容（口座・個人に関する情報）
FORBIDDEN = [
    (re.compile(r"\d[\d,]*(\.\d+)?\s*円"), "金額（円）"),
    (re.compile(r"残高|口座|数量|ロット|通貨（損切り|損失額"), "口座に関する言葉"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "メールアドレス"),
    (re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|\\Users\\"), "パソコンの中の場所"),
    (re.compile(r"discord(app)?\.com/api/webhooks", re.I), "Webhook の URL"),
]


class UnsafeMessage(ValueError):
    pass


def check_safe(text: str) -> None:
    for pat, what in FORBIDDEN:
        if pat.search(text):
            raise UnsafeMessage(f"送ってはいけない内容（{what}）が含まれているため、送りません")


@dataclass(frozen=True)
class Config:
    enabled: bool
    webhook: str

    @property
    def ready(self) -> bool:
        return self.enabled and self.webhook.startswith("https://")


def load_env(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE の形の簡単な .env を読む（# から始まる行は説明）。"""
    path = path or ENV_FILE
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_config(path: Path | None = None) -> Config:
    env = load_env(path)
    return Config(env.get("NOTIFY_ENABLED", "false").lower() == "true", env.get("DISCORD_WEBHOOK_URL", ""))


# ---------------- 文面 ----------------

def _dir(d: int) -> str:
    return "買い" if d > 0 else "売り"


def signal_message(pair: str, tf: str, strategy_name: str, direction: int, signal_time: pd.Timestamp,
                   entry: float | None, sl: float | None, tp: float | None, digits: int,
                   verdict: str | None = None, dsr: float | None = None, pbo: float | None = None,
                   n_trials: int | None = None, meta: str | None = None) -> str:
    """シグナルの通知。数量・損失額（円）は口座に関する情報なので入れない。"""
    f = (lambda x: "—" if x is None or pd.isna(x) else f"{x:.{digits}f}")
    lines = [
        f"📡 **{pair}（{PAIR_NAMES.get(pair, '')}）{TIMEFRAMES[tf].label}：{strategy_name} が{_dir(direction)}**",
        f"シグナルの足：{(pd.Timestamp(signal_time) + JST):%m/%d %H:%M}（日本時間）",
        f"エントリーの目安：{f(entry)}（次の足の始値）／損切り：{f(sl)}／利確：{f(tp)}",
    ]
    if meta:
        lines.append(f"機械学習の判定：{meta}")
    ev = []
    if verdict:
        ev.append(f"検証の判定「{verdict}」")
    if dsr is not None and not pd.isna(dsr):
        ev.append(f"DSR {dsr:.0%}")
    if pbo is not None and not pd.isna(pbo):
        ev.append(f"PBO {pbo:.0%}")
    if n_trials:
        ev.append(f"試した組み合わせ {n_trials:,} 通り")
    if ev:
        lines.append("根拠の強さ：" + "・".join(ev))
    return "\n".join(lines)


def status_message(kind: str, text: str) -> str:
    icon = {"kill": "⛔", "release": "✅", "retrain": "🤖", "challenger": "🔔", "papers": "📰", "error": "⚠️",
            "alert": "🔔", "review": "🗓️"}.get(kind, "ℹ️")
    return f"{icon} {text}"


def kill_message(kind: str, until: pd.Timestamp) -> str:
    """キルスイッチの作動。ペーパー口座の損益の数字は口座に関する情報なので入れず、種類と解除の時刻だけ。"""
    what = {"daily_loss": "1日の損失の上限", "weekly_loss": "1週間の損失の上限", "consecutive": "連敗の上限",
            "frequency": "1日のシグナルの回数の上限"}.get(kind, kind)
    return status_message("kill", f"キルスイッチ作動：{what}に達したため、新しいシグナルを止めています"
                                  f"（自動で解除：{(until + JST):%m/%d %H:%M} 日本時間）")


def with_disclaimer(text: str) -> str:
    body = text if len(text) <= MAX_LEN - len(DISCLAIMER) - 2 else text[:MAX_LEN - len(DISCLAIMER) - 3] + "…"
    return f"{body}\n{DISCLAIMER}"


# ---------------- 送信 ----------------

def post_discord(webhook: str, text: str, timeout: float = 15.0) -> None:
    body = json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode("utf-8")
    req = urllib.request.Request(webhook, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "fx-signal-tool/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status >= 300:
            raise OSError(f"Discord の応答 {r.status}")


def ensure(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


def send(con: sqlite3.Connection, key: str, kind: str, text: str, now: pd.Timestamp, cfg: Config | None = None,
         sender=post_discord) -> str:
    """1件の通知。戻り値：sent（送った）／duplicate（送り済み）／disabled（設定で止めている）／blocked（安全のため送らない）／error。"""
    ensure(con)
    if con.execute("SELECT 1 FROM notifications WHERE key=? AND status='sent'", (key,)).fetchone():
        return "duplicate"
    cfg = cfg or load_config()
    msg = with_disclaimer(text)
    status, err = "sent", None
    try:
        check_safe(msg)
        if not cfg.ready:
            status = "disabled"
        else:
            sender(cfg.webhook, msg)
    except UnsafeMessage as e:
        status, err = "blocked", str(e)
    except Exception as e:  # 通知に失敗しても、記録の処理は止めない
        status, err = "error", f"{type(e).__name__}: {e}"
    con.execute("""INSERT OR REPLACE INTO notifications (key, kind, created_at, sent_at, text, status, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key, kind, ts_str(now), ts_str(now) if status == "sent" else None, msg, status, err))
    con.commit()
    return status


# ---------------- 定期処理からまとめて送る ----------------

def pending_signals(con: sqlite3.Connection, since: pd.Timestamp) -> pd.DataFrame:
    """since より後に記録した、停止中でない戦略のシグナル（エントリーが決まったもの）。"""
    return pd.read_sql_query(
        """SELECT s.id, s.pair, s.tf, s.source, s.signal_time, s.direction, s.entry, s.sl, s.tp, s.meta_decision,
                  v.verdict, v.dsr, v.pbo, v.n_trials
           FROM signal_log s LEFT JOIN backtest_verdicts v ON v.pair=s.pair AND v.tf=s.tf AND v.source=s.source
           WHERE s.logged_at > ? AND s.suppressed = 0 AND s.entry IS NOT NULL AND s.source NOT LIKE 'ml\\_%' ESCAPE '\\'
           ORDER BY s.signal_time""", con, params=(ts_str(since),))


def notify_cycle(con: sqlite3.Connection, now: pd.Timestamp, cfg: Config | None = None, sender=post_discord,
                 kill=None) -> dict[str, int]:
    """定期処理の最後に呼ぶ：新しいシグナル・新しい警告・キルスイッチの作動を送る。"""
    from . import strategies, tracking
    from .config import PAIRS, price_digits
    ensure(con)
    cfg = cfg or load_config()
    counts: dict[str, int] = {}
    last = tracking.get_state(con, "*", "*", "notify_last") or ts_str(now - pd.Timedelta(hours=2))
    names = {s.key: s.name for s in strategies.STRATEGIES}

    def out(key, kind, text):
        r = send(con, key, kind, text, now, cfg, sender)
        counts[r] = counts.get(r, 0) + 1

    for r in pending_signals(con, pd.Timestamp(last)).itertuples():
        meta = r.meta_decision if isinstance(r.meta_decision, str) else None
        out(f"signal:{r.id}", "signal",
            signal_message(r.pair, r.tf, names.get(r.source, r.source), int(r.direction), pd.Timestamp(r.signal_time),
                           r.entry, r.sl, r.tp, price_digits(PAIRS[r.pair]), r.verdict, r.dsr, r.pbo,
                           None if pd.isna(r.n_trials) else int(r.n_trials), meta))
    alerts = pd.read_sql_query("SELECT id, pair, tf, kind, message FROM alerts WHERE created_at > ? AND kind NOT LIKE 'kill_%'",
                               con, params=(last,))
    for a in alerts.itertuples():
        out(f"alert:{a.id}", "alert", status_message("alert", f"{a.pair} {TIMEFRAMES[a.tf].label if a.tf in TIMEFRAMES else a.tf}："
                                                               f"{_strip_account(a.message)}"))
    if kill is not None:
        for t in kill.trips:
            out(f"kill:{t.key}", "kill", kill_message(t.kind, t.until))
    # 年1回の見直しの時期（近づいたとき・過ぎたときに、それぞれ1回だけ）
    from . import review
    from .config import JOB_PAIRS
    from .settings import load as load_settings
    r_cfg = load_settings().review
    for pair in JOB_PAIRS:
        for tf in TIMEFRAMES:
            rs = review.status(pair, tf, now, r_cfg)
            if rs is not None and rs.state != "ok":
                out(f"review:{pair}:{tf}:{rs.due:%Y-%m-%d}:{rs.state}", "review", status_message("review", rs.message()))
    tracking.set_state(con, "*", "*", "notify_last", ts_str(now))
    con.commit()
    return counts


def _strip_account(msg: str) -> str:
    """警告の文から、口座に関する数字を含む部分を除く（例：最大損失の R の説明）。"""
    return re.sub(r"（[^）]*(口座|残高|円)[^）]*）", "", msg)


def error_message(where: str, err: str) -> str:
    """エラーの通知。パソコンの中の場所などが入らないように、種類と短い説明だけにする。"""
    short = re.sub(r"[A-Za-z]:\\[^\s'\"]+|/[^\s'\"]+/[^\s'\"]+", "（場所）", err)[:200]
    return status_message("error", f"{where}でエラーが起きました：{short}")
