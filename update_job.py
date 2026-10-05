"""定期実行用スクリプト（Windows のタスクスケジューラから1時間ごとに起動する想定）。

取得 → 記録 → 答え合わせ → 再学習 → 重み → 警告 を、設定した通貨ペアと時間足ごとに行う。
週末は、設定で有効にしていれば GA の再探索と論文の収集も行う。最後に、.env で有効にしていれば Discord に通知する
（シグナルと動作状況だけ。口座・個人に関する情報は送らない）。
表示と記録・通知のみで、注文は一切しない。実行結果は data/job.log に追記する。
"""
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fxtool import data, ga_jobs, jobs, killswitch, notify, papers  # noqa: E402
from fxtool import settings as user_settings  # noqa: E402
from fxtool.config import JOB_PAIRS, TIMEFRAMES  # noqa: E402


def main() -> int:
    (ROOT / "data").mkdir(exist_ok=True)
    logging.basicConfig(
        filename=ROOT / "data" / "job.log", level=logging.INFO, encoding="utf-8",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    t0 = time.monotonic()
    con = data.connect()
    failed = False
    status_msgs: list[tuple[str, str, str]] = []  # （キー, 種類, 文面）
    stamp = f"{data.now_utc():%Y%m%d%H}"
    try:
        for pair in JOB_PAIRS:
            for tf in TIMEFRAMES:
                r = jobs.run_cycle(con, pair, tf, origin="schedule", force=True)
                line = f"{pair} {TIMEFRAMES[tf].label}: {r.summary()}"
                if r.error:
                    failed = True
                    logging.error(f"{line} / エラー: {r.error}")
                    status_msgs.append((f"error:{pair}:{tf}:{stamp}", "error",
                                        notify.error_message(f"{pair} {TIMEFRAMES[tf].label}の定期処理", r.error)))
                else:
                    logging.info(line)
                if r.retrained:
                    status_msgs.append((f"retrain:{pair}:{tf}:{stamp}", "retrain", notify.status_message(
                        "retrain", f"{pair} {TIMEFRAMES[tf].label}：機械学習（メタラベリング）を最新のデータで再学習しました")))
                print(line + (f" / エラー: {r.error}" if r.error else ""))
                # 週末の GA の再探索（settings.toml で有効にしたときだけ。重い処理なので週末に）
                try:
                    if ga_jobs.weekly_due(con, pair, tf, data.now_utc()):
                        run_id = ga_jobs.run(con, pair, tf, "neighborhood", origin="schedule")
                        logging.info(f"{pair} {TIMEFRAMES[tf].label}: GA の週末の再探索（run {run_id}）")
                except Exception as e:  # 再探索に失敗しても、記録の処理は止めない
                    failed = True
                    logging.error(f"{pair} {TIMEFRAMES[tf].label}: GA の週末の再探索でエラー: {e}")
        # 論文ウォッチの週1回の収集（settings.toml の [papers] enabled = true のときだけ。外部に接続する）
        s = user_settings.load()
        try:
            if papers.weekly_due(con, s.papers, data.now_utc()):
                n, new = papers.fetch(con, s.papers, data.now_utc(), origin="schedule")
                logging.info(f"論文ウォッチ：{n}件中、新しい論文 {new}件")
                status_msgs.append((f"papers:{stamp}", "papers",
                                    notify.status_message("papers", f"論文ウォッチ：新しい論文が {new} 件ありました")))
        except Exception as e:
            failed = True
            logging.error(f"論文ウォッチでエラー: {e}")
        # Discord への通知（.env の NOTIFY_ENABLED=true のときだけ送る）
        try:
            cfg = notify.load_config()
            _, kill = killswitch.check(con, data.now_utc(), s, data.latest_close(con, "USD/JPY"))
            counts = notify.notify_cycle(con, data.now_utc(), cfg, kill=kill)
            for key, kind, text in status_msgs:
                notify.send(con, key, kind, text, data.now_utc(), cfg)
            if cfg.ready:
                logging.info(f"通知：{counts}")
        except Exception as e:
            failed = True
            logging.error(f"通知でエラー: {e}")
    finally:
        con.close()
    logging.info(f"所要時間 {time.monotonic() - t0:.0f} 秒")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
