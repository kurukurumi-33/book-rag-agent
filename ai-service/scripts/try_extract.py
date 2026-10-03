"""调 prompt 用的辅助脚本：拿真实帖子跑抽取，打印输入输出对比。

不是 AI 服务的一部分，是开发时调 prompt 的工具。
（和 gen_mock_posts.py 一样，属于 scripts/ 下的开发脚本）

用法：
    .venv/Scripts/python.exe scripts/try_extract.py            # 随机 5 条（并发）
    .venv/Scripts/python.exe scripts/try_extract.py -n 20      # 随机 20 条
    .venv/Scripts/python.exe scripts/try_extract.py -c 4       # 限制最多同时发 4 个请求
    .venv/Scripts/python.exe scripts/try_extract.py --serial   # 退回串行，用来对比快慢
    .venv/Scripts/python.exe scripts/try_extract.py -t "高数同济七版 40 有笔记"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Union

import pymysql
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import BookInfo  # noqa: E402
from app.services.extract import (  # noqa: E402
    extract_book_info,
    extract_book_info_batch,
)

load_dotenv()

# 一条帖子要么抽出 BookInfo，要么是一坨异常（并发模式下不中断整批）
Outcome = Union[BookInfo, BaseException]


def fetch_samples(n: int) -> List[str]:
    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "book_agent"),
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT raw_text FROM book_post ORDER BY RAND() LIMIT %s", (n,)
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def show(idx: int, raw_text: str, outcome: Outcome) -> None:
    """打印一条的输入输出。outcome 是异常就打印异常，不往外抛。"""
    print(f"\n--- [{idx}] 输入 ---")
    print(raw_text)
    print("--- 输出 ---")
    if isinstance(outcome, BaseException):
        print(f"  [失败] {type(outcome).__name__}: {outcome}")
        return
    for k, v in outcome.model_dump().items():
        print(f"  {k}: {v!r}")


def run_serial(texts: List[str]) -> List[Outcome]:
    """一条一条跑：第 2 条必须等第 1 条完全返回。"""
    out: List[Outcome] = []
    for t in texts:
        try:
            out.append(extract_book_info(t))
        except Exception as e:  # noqa: BLE001 —— 开发脚本，啥错都打印出来
            out.append(e)
    return out


def run_batch(texts: List[str], concurrency: Union[int, None]) -> List[Outcome]:
    """一起发出去：N 个请求同时在飞，总耗时 ≈ 最慢的那一条。"""
    return extract_book_info_batch(
        texts, max_concurrency=concurrency, return_exceptions=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="试跑信息抽取")
    parser.add_argument("-n", type=int, default=5, help="随机抽几条帖子")
    parser.add_argument("-t", "--text", help="直接指定一段文本测试")
    parser.add_argument(
        "-c", "--concurrency", type=int, help="最大并发数（条数多时防 429 限流）"
    )
    parser.add_argument("--serial", action="store_true", help="串行跑，用来和并发对比")
    args = parser.parse_args()

    # 指定单条文本：只跑一次，不需要并发
    if args.text:
        show(1, args.text, extract_book_info(args.text))
        print()
        return 0

    texts = fetch_samples(args.n)
    if not texts:
        print("数据库里没有帖子，先用 gen_mock_posts.py 造点数据")
        return 1

    start = time.perf_counter()
    if args.serial:
        results = run_serial(texts)
        mode = "串行"
    else:
        results = run_batch(texts, args.concurrency)
        mode = "并发"
        if args.concurrency:
            mode += f"（max_concurrency={args.concurrency}）"
    elapsed = time.perf_counter() - start

    for i, (t, r) in enumerate(zip(texts, results), 1):
        show(i, t, r)

    failed = sum(1 for r in results if isinstance(r, BaseException))
    print(f"\n[{mode}] 共 {len(texts)} 条，成功 {len(texts) - failed}，失败 {failed}")
    print(f"总耗时 {elapsed:.1f}s，平均 {elapsed / len(texts):.2f}s/条")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
