"""定期実行用スクリプト（Windows のタスクスケジューラから1時間ごとに起動する想定）。

取得 → 記録 → 答え合わせ → 再学習 → 重み → 警告 を、設定した通貨ペアと時間足ごとに行う。
表示と記録のみで、注文は一切しない。実行結果は data/job.log に追記する。
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fxtool import data, ga_jobs, jobs, papers  # noqa: E402
from fxtool import settings as user_settings  # noqa: E402
from fxtool.config import JOB_PAIRS, TIMEFRAMES  # noqa: E402


def main() -> int:
    (ROOT / "data").mkdir(exist_ok=True)
    logging.basicConfig(
        filename=ROOT / "data" / "job.log", level=logging.INFO, encoding="utf-8",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    con = data.connect()
    failed = False
    try:
        for pair in JOB_PAIRS:
            for tf in TIMEFRAMES:
                r = jobs.run_cycle(con, pair, tf, origin="schedule", force=True)
                line = f"{pair} {TIMEFRAMES[tf].label}: {r.summary()}"
                if r.error:
                    failed = True
                    logging.error(f"{line} / エラー: {r.error}")
                else:
                    logging.info(line)
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
        try:
            p = user_settings.load().papers
            if papers.weekly_due(con, p, data.now_utc()):
                n, new = papers.fetch(con, p, data.now_utc(), origin="schedule")
                logging.info(f"論文ウォッチ：{n}件中、新しい論文 {new}件")
        except Exception as e:
            failed = True
            logging.error(f"論文ウォッチでエラー: {e}")
    finally:
        con.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
