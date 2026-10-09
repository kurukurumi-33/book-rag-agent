"""开发用种子脚本：用 LLM 批量生成模拟二手书帖子，写入 MySQL。

注意：这是开发工具，不是 AI 服务本体。
AI 服务（app/）不直连业务数据库，只通过 Spring Boot 的 API 取数据。

## 数据性质（面试会被问，记住这个说法）

**书目是真的，帖子是编的。**
- 书目来自 `scripts/data/book_catalog.json`（由 fetch_book_catalog.py 抓豆瓣公开列表页得到，
  是真实出版物）
- 帖文 `raw_text` 由 DeepSeek 生成 —— 不含任何真实交易或个人信息

## 为什么不能只靠 30 个手写种子

最早的版本只有 30 本书、随机挑 6 本让模型「自由发挥」。生成 220 条时没问题
（约 37 个书种），但生成 1 万条就退化成「每本 300 多份」的复读机 ——
检索评测会失去意义（任何查询都能命中，top-3 命中率飙到 100%）。

所以现在改成：**每本书一条帖子**，书目从上万本候选里按轮次抽，
保证覆盖均匀，1 万条 ≈ 每本 7 份，接近真实市场密度。

## 断点续跑

`--count` 指的是**库里总共要有多少条**，不是「这次要生成多少条」。
脚本开头会查一次当前条数，只补差额。所以中途挂了（或按 Ctrl-C）直接重跑即可，
不会重复灌数据。

    # 第一次：库里 220 条，补到 10000
    .venv/Scripts/python.exe scripts/gen_mock_posts.py --count 10000

    # 跑到一半断了，再跑一次 —— 会从断点接着补
    .venv/Scripts/python.exe scripts/gen_mock_posts.py --count 10000

    # 只想要 500 条试水
    .venv/Scripts/python.exe scripts/gen_mock_posts.py --count 500
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import pymysql
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 保证能从项目根目录导入 app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import get_llm  # noqa: E402

load_dotenv()

# 不是终端时 stdout 是块缓冲的，跑两个小时一行都不显示，看着像卡死。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

BATCH_SIZE = 20

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "book_catalog.json"

# 抓不到书目时的兜底：最早那 30 本手写种子。
# 有它脚本就不会因为缺一个 json 文件而跑不起来，但覆盖率会退回到「30 本」的水平 ——
# 所以正常情况别删 book_catalog.json。
FALLBACK_BOOKS = [
    {"title": "高等数学", "author": "同济大学数学系", "publisher": "高等教育出版社"},
    {"title": "线性代数", "author": "同济大学数学系", "publisher": "高等教育出版社"},
    {"title": "概率论与数理统计", "author": "盛骤", "publisher": "高等教育出版社"},
    {"title": "离散数学", "author": "屈婉玲", "publisher": "高等教育出版社"},
    {"title": "大学物理", "author": "马文蔚", "publisher": "高等教育出版社"},
    {"title": "新视野大学英语", "author": None, "publisher": "外语教学与研究出版社"},
    {"title": "数据结构", "author": "严蔚敏", "publisher": "清华大学出版社"},
    {"title": "计算机操作系统", "author": "汤小丹", "publisher": "西安电子科技大学出版社"},
    {"title": "计算机网络", "author": "谢希仁", "publisher": "电子工业出版社"},
    {"title": "数据库系统概论", "author": "王珊", "publisher": "高等教育出版社"},
    {"title": "软件工程", "author": "张海藩", "publisher": "人民邮电出版社"},
    {"title": "编译原理", "author": "陈火旺", "publisher": "国防工业出版社"},
    {"title": "Java核心技术", "author": "Cay S. Horstmann", "publisher": "机械工业出版社"},
    {"title": "Python编程从入门到实践", "author": "Eric Matthes", "publisher": "人民邮电出版社"},
    {"title": "C Primer Plus", "author": "Stephen Prata", "publisher": "人民邮电出版社"},
    {"title": "算法导论", "author": "Thomas H. Cormen", "publisher": "机械工业出版社"},
    {"title": "深入理解计算机系统", "author": "Randal E. Bryant", "publisher": "机械工业出版社"},
    {"title": "Effective Java", "author": "Joshua Bloch", "publisher": "机械工业出版社"},
    {"title": "管理学", "author": "罗宾斯", "publisher": "中国人民大学出版社"},
    {"title": "西方经济学", "author": "高鸿业", "publisher": "中国人民大学出版社"},
    {"title": "会计学原理", "author": None, "publisher": "中国人民大学出版社"},
    {"title": "市场营销学", "author": "科特勒", "publisher": "中国人民大学出版社"},
    {"title": "统计学", "author": "贾俊平", "publisher": "中国人民大学出版社"},
    {"title": "大学计算机基础", "author": None, "publisher": "高等教育出版社"},
    {"title": "马克思主义基本原理", "author": None, "publisher": "高等教育出版社"},
    {"title": "毛泽东思想和中国特色社会主义理论体系概论", "author": None, "publisher": "高等教育出版社"},
    {"title": "英语四级词汇", "author": None, "publisher": "世界图书出版公司"},
    {"title": "考研数学复习全书", "author": "李永乐", "publisher": "国家开放大学出版社"},
    {"title": "数据结构与算法分析", "author": "Mark Allen Weiss", "publisher": "机械工业出版社"},
    {"title": "数学分析", "author": "华东师范大学数学系", "publisher": "高等教育出版社"},
]

SYSTEM_PROMPT = """你是大学二手书交易群里的学生，正在发帖卖书。

请为指定的每一本书各写一条卖书帖子。

## ⚠️ 最重要的一条：不要把书单里的信息原样抄下来

书单给你「书名（作者 / 出版社）」只是为了让你知道**是哪本书**，
不是让你照抄。真实的卖家发帖时**很少**把书名、作者、出版社、版次全写全 ——
请照着下面的比例来：

- **书名**：大部分用简称或口语。
  「高等数学」→「高数」；「概率论与数理统计」→「概率论」；
  「计算机操作系统」→「操作系统」；「新视野大学英语」→「新视野」；
  「数据结构与算法分析」→「数据结构」。只有少数帖子写全名。
- **作者**：十有八九**不写**。写的时候多数只写姓（「严蔚敏」→「严奶奶那本」也行）。
- **出版社**：大多数**不写**。写了也常是简称（「清华大学出版社」→「清华」）。
- **版次**：想到才写，不写是常态。

一句话：**看起来要像随手打的，不像填表填出来的。**

## 其他要求

1. **格式必须各不相同**：有的用空格/顿号分隔，有的写成完整句子，有的分行写；
   有的带 emoji 或颜文字，有的全是缩写口语

2. **信息完整度参差不齐**：
   - 少数帖子写清书名+版次+成色+价格
   - 多数只写书名和价格
   - 有的只写「出一本高数」这种极简的
   - 价格写法多样：「20」「二十」「20r」「20块」「20出」

3. **成色描述要多样**：九成新、八成新、几乎全新、有笔记、划过重点、
   书角卷了、有点破损、无笔记、翻过几次

4. 价格在 5-45 元之间（也可以偶尔「免费送」「白送」）

5. 允许出现错别字、口语（出、甩、便宜出、可小刀、自提、包邮）

6. **不要换成别的书** —— 可以用简称，但别写成另一本。

**重点：不要每条都写得很完整很规范。真实群里大部分帖子信息是残缺的。**

user_id 在 1 到 500 之间随机取。"""


class MockPost(BaseModel):
    user_id: int
    # 这条帖子写的是「第几本书」（对应下面给的书单里的序号，从 1 开始）。
    # 有了它，我们就能把「本来想让他写哪本」和「抽取抽出来是哪本」对上 ——
    # 这是**抽取准确率**唯一的测量方式（否则没有金标准，只能靠眼看）。
    book_index: int = Field(description="这条帖子写的是书单里的第几本，从 1 开始")
    raw_text: str


class MockPostBatch(BaseModel):
    posts: List[MockPost]


def load_catalog() -> list[dict]:
    """读书目。没有就退回内置的 30 本，并明确警告。"""
    if not CATALOG_PATH.exists():
        print(f"⚠️  找不到 {CATALOG_PATH}")
        print("   退回内置的 30 本种子 —— 生成上万条会退化成「每本几百份」。")
        print("   建议先跑：scripts/fetch_book_catalog.py")
        return FALLBACK_BOOKS

    books = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not books:
        print(f"⚠️  {CATALOG_PATH} 是空的，退回内置种子")
        return FALLBACK_BOOKS
    print(f"书目已加载：{len(books)} 本（来自 {CATALOG_PATH.name}）")
    return books


def build_user_prompt(books: list[dict], n: int) -> str:
    lines = []
    for i, b in enumerate(books, start=1):
        extra = " / ".join(x for x in (b.get("author"), b.get("publisher")) if x)
        lines.append(f"{i}. {b['title']}" + (f"（{extra}）" if extra else ""))
    return (
        f"请为下面这 {n} 本书**各写一条**卖书帖子（一条对应一本，别漏、别合并）：\n\n"
        + "\n".join(lines)
        + f"\n\n生成 {n} 条，每条都要填 book_index 说明写的是第几本。只输出 JSON。"
    )


_structured_llm = None


def get_structured_llm(llm):
    """拿到「输出必为 MockPostBatch」的包装器，建一次复用。

    注意和 app/services/extract.py 一样要显式写 function_calling ——
    DeepSeek 不认默认的 json_schema 模式，会直接 400。
    """
    global _structured_llm
    if _structured_llm is None:
        _structured_llm = llm.with_structured_output(
            MockPostBatch, method="function_calling"
        )
    return _structured_llm


def build_messages(books: list[dict], n: int) -> list[tuple[str, str]]:
    """把「这一批写哪几本书」拼成一条请求的消息。"""
    return [
        ("system", SYSTEM_PROMPT),
        ("human", build_user_prompt(books, n)),
    ]


def generate_batch(llm, books: list[dict], n: int) -> List[MockPost]:
    """调一次 LLM，生成一批帖子（串行版，留着调试用）。"""
    result = get_structured_llm(llm).invoke(build_messages(books, n))
    return result.posts


def connect_db():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "book_agent"),
        charset="utf8mb4",
    )


def current_count(conn) -> int:
    """库里现有多少条 —— 断点续跑靠它。"""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM book_post")
        return int(cur.fetchone()[0])


def insert_posts(conn, posts: List[MockPost]) -> int:
    if not posts:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO book_post (user_id, raw_text, status, extract_status) "
            "VALUES (%s, %s, 'ON_SALE', 'PENDING')",
            [(p.user_id, p.raw_text) for p in posts],
        )
    conn.commit()
    return len(posts)


class Deck:
    """不放回地抽书目，抽完自动洗牌重来。

    为什么不用 random.choice（有放回）：有放回时 1 万次抽样会有书被抽到十几次、
    有的书一次没抽到（生日问题）。不放回的洗牌能保证**轮次内均匀**，
    1 万条 / 1500 本 ≈ 每本 6~7 次，正是我们要的密度。
    """

    def __init__(self, books: list[dict], rng: random.Random):
        self._books = books
        self._rng = rng
        self._queue: list[dict] = []

    def draw(self, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            if not self._queue:
                self._queue = list(self._books)
                self._rng.shuffle(self._queue)
            out.append(self._queue.pop())
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="生成模拟二手书帖子")
    parser.add_argument("--count", type=int, default=200,
                        help="库里**总共**要多少条（不是这次生成多少条）。只补差额")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"一次请求生成几条（默认 {BATCH_SIZE}）")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="同时发几个请求（默认 5）。调大能缩短墙钟时间，"
                             "但太大容易被对端限流（429）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，固定它可以让两次跑的选书顺序一致")
    args = parser.parse_args()

    if args.batch_size < 1:
        print("--batch-size 必须 >= 1")
        return 2

    rng = random.Random(args.seed)
    books = load_catalog()
    deck = Deck(books, rng)

    llm = get_llm()
    conn = connect_db()

    try:
        have = current_count(conn)
        todo = args.count - have
        print(f"库里现有 {have} 条，目标 {args.count} 条 → 本次生成 {todo} 条")
        if todo <= 0:
            print("已经够了，什么都不做。")
            return 0

        # 先把所有批次**全部规划好**（每批几条、写哪几本书），再拿去并发跑。
        # 这样并发窗口之间互不依赖，断了重跑也只需按 --count 补差额。
        specs: list[tuple[int, list[dict]]] = []
        remaining = todo
        while remaining > 0:
            n = min(args.batch_size, remaining)
            specs.append((n, deck.draw(n)))
            remaining -= n

        batches = len(specs)
        print(f"分 {batches} 批，每批 {args.batch_size} 条，"
              f"并发 {args.concurrency} → 约 {math.ceil(batches / args.concurrency)} 轮")
        print(f"预计 {(batches / args.concurrency) * 15 / 60:.0f} 分钟左右\n")

        structured = get_structured_llm(llm)
        inserted = 0
        failed_batches = 0
        started = time.perf_counter()

        # 一轮并发发 concurrency 个请求，回来后写库、再发下一轮。
        # 为什么不一次性把 500 个请求全丢出去：那样会打爆对端限流（429），
        # 而且中途挂了已经跑完的 400 批结果也拿不回来（不在库里）。
        for start in range(0, batches, args.concurrency):
            window = specs[start : start + args.concurrency]
            messages = [build_messages(books, n) for n, books in window]
            try:
                outs = structured.batch(
                    messages,
                    config={"max_concurrency": args.concurrency},
                    return_exceptions=True,
                )
            except KeyboardInterrupt:
                print("\n手动中断 —— 已写入的都在库里，重跑本脚本会自动接着补。")
                raise

            for (n, _books), out in zip(window, outs):
                if isinstance(out, Exception):
                    failed_batches += 1
                    print(f"    ⚠️ 一批失败（{n} 条）：{type(out).__name__}: {out}")
                    continue
                inserted += insert_posts(conn, out.posts)

            elapsed = time.perf_counter() - started
            rate = inserted / elapsed if elapsed else 0
            left = (todo - inserted) / rate if rate else 0
            done_rounds = start // args.concurrency + 1
            print(f"  [第 {done_rounds}/{math.ceil(batches / args.concurrency)} 轮]"
                  f" 累计 {inserted}/{todo}  剩约 {left / 60:.0f} 分钟")

        print(f"\n完成，本次写入 {inserted} 条，失败批次 {failed_batches}，"
              f"用时 {(time.perf_counter() - started) / 60:.1f} 分钟")
        if failed_batches:
            print("失败的批次没写库 —— 直接重跑本脚本，它会按 --count 补差额。")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
