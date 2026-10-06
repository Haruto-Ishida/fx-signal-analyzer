"""Step 10：Discord 通知（実際には送らない。送る関数を偽物にして確認）。"""
import pandas as pd
import pytest

from fxtool import killswitch, notify

NOW = pd.Timestamp("2026-10-07 03:00")
ON = notify.Config(True, "https://example.invalid/webhook")


@pytest.fixture
def con(tmp_path):
    from fxtool import data
    c = data.connect(tmp_path / "n.db")
    yield c
    c.close()


class Sender:
    def __init__(self):
        self.sent = []

    def __call__(self, url, text):
        self.sent.append((url, text))


def test_env_parsing_and_default_off(tmp_path):
    assert not notify.load_config(tmp_path / "none.env").ready
    p = tmp_path / ".env"
    p.write_text("# 説明\nNOTIFY_ENABLED=true\nDISCORD_WEBHOOK_URL=\"https://discord.com/api/webhooks/1/abc\"\n", encoding="utf-8")
    c = notify.load_config(p)
    assert c.ready and c.webhook == "https://discord.com/api/webhooks/1/abc"
    p.write_text("NOTIFY_ENABLED=false\nDISCORD_WEBHOOK_URL=https://x\n", encoding="utf-8")
    assert not notify.load_config(p).ready
    assert notify.load_config(notify.ROOT / ".env.example").ready is False   # 見本は送らない設定


def test_env_is_ignored_by_git():
    text = (notify.ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in [l.strip() for l in text.splitlines()]


@pytest.mark.parametrize("bad", [
    "残高 100,000円", "損失 1,000 円", "口座の損益", "数量 3,000通貨", "連絡先 someone@example.com",
    r"C:\Users\someone\file", "https://discord.com/api/webhooks/1/abc",
    "https://api.stlouisfed.org/fred/series?api_key=0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef",
])
def test_unsafe_messages_are_blocked(con, bad):
    s = Sender()
    assert notify.send(con, f"k:{bad}", "test", f"テスト {bad}", NOW, ON, s) == "blocked"
    assert s.sent == []


def test_signal_message_has_no_account_info_and_has_disclaimer(con):
    msg = notify.signal_message("USD/JPY", "1h", "移動平均クロス", 1, pd.Timestamp("2026-10-07 01:00"), 150.123, 149.9,
                                150.5, 3, verdict="偶然の範囲", dsr=0.6, pbo=0.3, n_trials=86, meta="乗る")
    notify.check_safe(msg)
    assert "買い" in msg and "150.123" in msg and "DSR 60%" in msg and "86 通り" in msg
    s = Sender()
    assert notify.send(con, "signal:1", "signal", msg, NOW, ON, s) == "sent"
    assert s.sent[0][1].endswith(notify.DISCLAIMER)
    assert notify.send(con, "signal:1", "signal", msg, NOW, ON, s) == "duplicate"   # 二度送らない
    assert len(s.sent) == 1


def test_disabled_does_not_send(con):
    s = Sender()
    assert notify.send(con, "x", "test", "テスト", NOW, notify.Config(False, "https://x"), s) == "disabled"
    assert s.sent == []


def test_send_error_is_recorded_not_raised(con):
    def broken(url, text):
        raise OSError("timeout")
    assert notify.send(con, "e", "test", "テスト", NOW, ON, broken) == "error"
    assert con.execute("SELECT error FROM notifications WHERE key='e'").fetchone()[0] == "OSError: timeout"


def test_kill_message_has_no_numbers_about_account():
    m = notify.kill_message("daily_loss", pd.Timestamp("2026-10-07 15:00"))
    notify.check_safe(m)
    assert "1日の損失の上限" in m and "%" not in m


def test_error_message_hides_paths():
    m = notify.error_message("USD/JPY 1時間足の定期処理", r"FileNotFoundError: C:\Users\someone\claude-work\data\fx.db")
    notify.check_safe(m)
    assert "someone" not in m


def test_notify_cycle_sends_new_signals_alerts_and_kill(con):
    from fxtool import tracking
    con.execute("""INSERT INTO signal_log (pair, tf, source, signal_time, direction, entry, sl, tp, atr, max_bars,
                   logged_at, status, entry_basis, suppressed) VALUES
                   ('USD/JPY', '1h', 'ma_cross', '2026-10-07 01:00:00', 1, 150.1, 149.95, 150.4, 0.1, 48,
                    '2026-10-07 02:05:00', '進行中', 'next_open', 0),
                   ('USD/JPY', '1h', 'rsi_reversal', '2026-10-07 01:00:00', -1, 150.1, 150.25, 149.8, 0.1, 48,
                    '2026-10-07 02:05:00', '進行中', 'next_open', 1)""")
    con.execute("""INSERT INTO alerts (pair, tf, source, kind, level, message, created_at) VALUES
                   ('USD/JPY', '1h', 'risk', 'max_loss', 'critical',
                    '今日の損失が -2.0R に達したため、今日は新しいシグナルを出しません（上限 −2R ＝ 口座残高の約2%）', '2026-10-07 02:30:00')""")
    con.commit()
    tracking.set_state(con, "*", "*", "notify_last", "2026-10-07 02:00:00")
    trip = killswitch.Trip("consecutive", "consecutive:x", "ペーパー口座で 5 回…", pd.Timestamp("2026-10-08 01:00"))
    s = Sender()
    counts = notify.notify_cycle(con, NOW, ON, s, kill=killswitch.KillState(trips=[trip]))
    assert counts == {"sent": 3}
    texts = [t for _, t in s.sent]
    assert "移動平均クロス が買い" in texts[0]                  # 停止中のシグナルは送らない
    assert "口座" not in texts[1] and "−2R" not in texts[1]      # 警告から口座に関する説明を除く
    assert "連敗の上限" in texts[2]
    assert notify.notify_cycle(con, NOW + pd.Timedelta(hours=1), ON, s, kill=killswitch.KillState(trips=[trip])) == {"duplicate": 1}
