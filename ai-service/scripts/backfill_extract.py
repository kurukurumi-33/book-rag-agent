"""把库里 `extract_status='PENDING'` 的帖子批量抽取，回填结构化字段。

开发工具，不是 AI 服务本体（和服务一样，这个脚本也只是**开发期**直连数据库；
线上抽取走的是 Spring Boot → AI 服务 `/extract/batch` 那条路）。

## 为什么不直接调 `/extract/batch`

可以，而且线上就该那么走。但**灌 1 万条**的时候有两个麻烦：

1. 主服务那边一次只捞 `batch-size: 50` 条（见 application.yml），
   1 万条要来回触发 200 轮，脚本里得自己循环打接口 —— 不如直连省事
2. 走接口时 `raw_text` 要先从库里查出来再 POST 出去、结果再写回来，
   多两次序列化；直连少绕一圈

**真正重要的是**：这个脚本用的是 `extract_book_info_many`（真批量，
一次请求 20 条），不是逐条那版。1 万条帖子的 token 从 13.4M 降到约 2M。

## 用法

    # 先看看有多少条待抽取
    .venv/Scripts/python.exe scripts/backfill_extract.py --dry-run

    # 真跑（默认一次取 200 条、每批 20 条，循环直到抽完）
    .venv/Scripts/python.exe scripts/backfill_extract.py

    # 只跑一轮就停（调试用）
    .venv/Scripts/python.exe scripts/backfill_extract.py --once
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pymysql
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import BookInfo  # noqa: E402
from app.services.extract import extract_book_info_many  # noqa: E402

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# 一次从库里捞几条。别太大：全捞进内存再逐批抽，中途挂了就白跑。
FETCH_LIMIT = 200

# 一次 LLM 请求塞几条（真批量）。和接口的 batch_size 默认值保持一致。
EXTRACT_BATCH = 20

_UPDATE_SQL = """
UPDATE book_post
   SET book_name = %s, edition = %s, publisher = %s, author = %s,
       condition_desc = %s, price = %s, has_notes = %s,
       extract_status = 'DONE'
 WHERE id = %s
"""


def connect_db():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "book_agent"),
        charset="utf8mb4",
    )


def pending_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM book_post WHERE extract_status = 'PENDING'")
        return int(cur.fetchone()[0])


def fetch_pending(conn, limit: int) -> list[tuple[int, str]]:
    """取一批待抽取的帖子。返回 [(id, raw_text)]。

    ⚠️ 用 `ORDER BY id LIMIT n`，**不要**用 `ORDER BY RAND()`：
    批量抽取是「抽一批 → 标记 DONE → 再抽下一批」，靠的就是「已抽的不再出现」。
    RAND() 会随机取，可能反复取到同一批已完成的（虽然被 WHERE 滤掉了，
    但排序本身要全表扫 + 排序，1 万条时明显变慢）。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, raw_text FROM book_post "
            "WHERE extract_status = 'PENDING' ORDER BY id LIMIT %s",
            (limit,),
        )
        return [(int(r[0]), r[1] or "") for r in cur.fetchall()]


def save(conn, rows: list[tuple[int, BookInfo]]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            _UPDATE_SQL,
            [
                (bi.book_name, bi.edition, bi.publisher, bi.author,
                 bi.condition_desc, bi.price, bi.has_notes, pid)
                for pid, bi in rows
            ],
        )
    conn.commit()
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量抽取并回填 PENDING 帖子")
    parser.add_argument("--dry-run", action="store_true", help="只统计待抽取条数，不调用 LLM")
    parser.add_argument("--once", action="store_true", help="只跑一轮就退出")
    parser.add_argument("--fetch-limit", type=int, default=FETCH_LIMIT)
    parser.add_argument("--batch-size", type=int, default=EXTRACT_BATCH)
    args = parser.parse_args()

    conn = connect_db()
    try:
        total = pending_count(conn)
        print(f"待抽取：{total} 条")
        if args.dry_run or total == 0:
            return 0

        rounds = (total + args.fetch_limit - 1) // args.fetch_limit
        print(f"每轮取 {args.fetch_limit} 条、每批 {args.batch_size} 条发给模型"
              f"（每批 = 1 次请求）→ 约 {rounds} 轮\n")

        done = failed = 0
        started = time.perf_counter()

        for rnd in range(1, rounds + 1):
            batch = fetch_pending(conn, args.fetch_limit)
            if not batch:
                break

            ids = [pid for pid, _ in batch]
            texts = [t for _, t in batch]
            results = extract_book_info_many(texts, batch_size=args.batch_size)

            ok_rows: list[tuple[int, BookInfo]] = []
            for pid, item in zip(ids, results):
                if isinstance(item, BookInfo):
                    ok_rows.append((pid, item))
                else:
                    # 失败的不写库、也不改 extract_status —— 保持 PENDING，
                    # 下次跑这个脚本会自动重试。比标记 FAILED 更适合灌数据场景。
                    failed += 1

            save(conn, ok_rows)
            done += len(ok_rows)

            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0
            left = (total - done - failed) / rate if rate else 0
            print(f"  [{rnd}/{rounds}] 成功 {len(ok_rows)}，失败 {len(ids) - len(ok_rows)}"
                  f"  （累计 {done}/{total}）  剩约 {left / 60:.0f} 分钟")

            if args.once:
                break

        print()
        print("=" * 60)
        print(f"完成：成功 {done}，失败 {failed}（失败的仍是 PENDING，重跑本脚本会重试）")
        print(f"用时 {(time.perf_counter() - started) / 60:.1f} 分钟")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
