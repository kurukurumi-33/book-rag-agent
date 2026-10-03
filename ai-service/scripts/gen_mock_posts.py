"""开发用种子脚本：用 LLM 批量生成模拟二手书帖子，写入 MySQL。

注意：这是开发工具，不是 AI 服务本体。
AI 服务（app/）不直连业务数据库，只通过 Spring Boot 的 API 取数据。

运行：
    .venv/Scripts/python.exe scripts/gen_mock_posts.py --count 200
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List

import pymysql
from dotenv import load_dotenv
from pydantic import BaseModel

# 保证能从项目根目录导入 app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import get_llm  # noqa: E402

load_dotenv()

BATCH_SIZE = 20

# 常见大学教材，作为生成的种子，保证覆盖面广
TEXTBOOKS = [
    "高等数学 同济大学 第七版", "线性代数 同济大学 第六版", "概率论与数理统计 浙大版",
    "离散数学 屈婉玲", "大学物理 马文蔚", "大学英语 新视野 第三版",
    "数据结构 C语言版 严蔚敏", "计算机操作系统 汤小丹", "计算机网络 谢希仁",
    "数据库系统概论 王珊", "软件工程 张海藩", "编译原理 龙书",
    "Java核心技术 卷I", "Python编程从入门到实践", "C Primer Plus",
    "算法导论", "深入理解计算机系统", "Effective Java",
    "管理学 罗宾斯", "西方经济学 高鸿业", "会计学原理",
    "市场营销学 科特勒", "统计学 贾俊平", "线性代数 居余马",
    "大学计算机基础", "马克思主义基本原理", "毛泽东思想概论",
    "英语四级词汇书", "考研数学复习全书", "数据结构与算法分析",
]

SYSTEM_PROMPT = """你是大学二手书交易群里的学生，正在发帖卖书。

请生成指定数量的卖书帖子，要求：

1. **格式必须各不相同**：
   - 有的用空格或顿号分隔信息，有的写成完整句子，有的分行写
   - 有的带 emoji 或颜文字，有的全是缩写和口语

2. **信息完整度必须参差不齐**：
   - 有的写清书名+版次+出版社+成色+价格
   - 有的只写书名和价格
   - 有的只写「出一本高数」这种极简的
   - 价格写法多样：「20」「二十」「20r」「20块」「20出」

3. **成色描述要多样**：九成新、八成新、几乎全新、有笔记、划过重点、
   书角卷了、有点破损、无笔记、翻过几次

4. 价格在 5-45 元之间

5. 允许出现错别字、口语（出、甩、便宜出、可小刀、自提、包邮）

**重点：不要每条都写得很完整很规范。真实群里大部分帖子信息是残缺的。**

user_id 在 1 到 500 之间随机取。"""


class MockPost(BaseModel):
    user_id: int
    raw_text: str


class MockPostBatch(BaseModel):
    posts: List[MockPost]


def build_user_prompt(books: List[str], n: int) -> str:
    return (
        f"请生成 {n} 条卖书帖子，围绕这些教材（可自由发挥）：\n"
        + "、".join(books)
        + f"\n\n生成 {n} 条，只输出 JSON。"
    )


def generate_batch(llm, books: List[str], n: int) -> List[MockPost]:
    """调一次 LLM，生成一批帖子。"""
    # DeepSeek 不支持 response_format=json_schema，改用 function calling
    structured = llm.with_structured_output(MockPostBatch, method="function_calling")
    result: MockPostBatch = structured.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", build_user_prompt(books, n)),
        ]
    )
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


def main() -> int:
    parser = argparse.ArgumentParser(description="生成模拟二手书帖子")
    parser.add_argument("--count", type=int, default=200, help="生成条数")
    args = parser.parse_args()

    total = args.count
    batches = math.ceil(total / BATCH_SIZE)
    print(f"目标 {total} 条，分 {batches} 批（每批 {BATCH_SIZE} 条）\n")

    llm = get_llm()
    conn = connect_db()
    inserted = 0

    try:
        for i in range(batches):
            books = random.sample(TEXTBOOKS, k=min(6, len(TEXTBOOKS)))
            n = min(BATCH_SIZE, total - inserted)
            try:
                posts = generate_batch(llm, books, n)
                cnt = insert_posts(conn, posts)
                inserted += cnt
                print(f"  [{i + 1}/{batches}] +{cnt} 条（累计 {inserted}）")
            except Exception as e:
                print(f"  [{i + 1}/{batches}] 失败: {type(e).__name__}: {e}")
    finally:
        conn.close()

    print(f"\n完成，共写入 {inserted} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
