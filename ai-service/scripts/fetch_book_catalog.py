"""从豆瓣图书标签页抓取**真实书目**（书名 / 作者 / 出版社），作为生成模拟帖子的种子。

## 为什么要有这个脚本

原来 `gen_mock_posts.py` 只有 **30 个手写种子**。生成 220 条帖子时够用（约 37 个书种），
但一旦要生成 **1 万条**，30 本书就变成「每本 300 多份」—— 语料退化成 30 本书的复读机，
检索评测会失去意义（任何查询都能命中，top-3 命中率飙到 100%）。

所以先把书目扩到 **上千本真实教材**，让 1 万条帖子大约是「每本 7 份」，接近真实市场密度。

**书是真的**（豆瓣收录的真实出版物），**帖子是编的**（见 gen_mock_posts.py）。
这个区分很重要，面试被问「数据哪来的」要能说清楚。

## ⚠️ 合规提示（别忽略）

豆瓣**没有 robots.txt**，但其用户协议名义上不允许自动化抓取。本脚本的姿态是：

- **一次性、低频率**：每个请求间隔 ≥ 1.2 秒，全跑一遍约 150 次请求、3 分钟
- **只取事实性元数据**：书名 / 作者 / 出版社 —— 不抓评论、不抓用户内容、不抓评分
- **不建持续任务**：跑一次生成静态 JSON 就够，不要挂成定时任务

如果你打算加大规模或做成定期更新，**先去确认合规性**。这里选了保守的姿态。

## 用法

    .venv/Scripts/python.exe scripts/fetch_book_catalog.py
    .venv/Scripts/python.exe scripts/fetch_book_catalog.py --max-per-tag 60
    .venv/Scripts/python.exe scripts/fetch_book_catalog.py --tags 教材,计算机
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "data" / "book_catalog.json"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 每个请求之间歇多久。豆瓣对高频抓取会返回 403，慢一点比被封强。
DELAY_SECONDS = 1.2

# 抓哪些标签。挑的都是大学教材密集的类目 —— 我们要的是「学生会在二手群卖的书」。
DEFAULT_TAGS = [
    "教材", "大学教材", "数学", "计算机", "物理学", "化学",
    "经济学", "管理学", "法学", "英语", "会计学", "统计学",
    "电子工程", "机械工程", "土木工程", "医学", "心理学", "历史",
]

# 出版社的识别特征。豆瓣的 pub 行格式是：作者 / 译者 / 出版社 / 日期 / 价格
# 但有的条目缺译者，所以不能按固定下标取，得靠特征找。
_PUBLISHER_HINT = re.compile(r"出版社|书局|书店|出版公司|Press", re.IGNORECASE)

_SUBJECT_ITEM = re.compile(r'<li class="subject-item">(.*?)</li>', re.DOTALL)
_TITLE = re.compile(r'<h2>\s*<a[^>]*title="([^"]+)"', re.DOTALL)
_PUB = re.compile(r'<div class="pub">(.*?)</div>', re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def fetch(url: str) -> str | None:
    """抓一个页面。失败返回 None（不抛，让调用方继续跑别的标签）。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        # 403 通常是触发了频率限制 —— 停下来比硬刚更明智
        print(f"    ⚠️ HTTP {e.code}，可能被限流了")
        return None
    except Exception as e:
        print(f"    ⚠️ {type(e).__name__}: {e}")
        return None


def parse_page(page: str) -> list[dict]:
    """把一页列表页解析成 [{title, author, publisher}, ...]。"""
    books = []
    for block in _SUBJECT_ITEM.findall(page):
        m = _TITLE.search(block)
        if not m:
            continue
        title = html.unescape(m.group(1)).strip()

        author = publisher = ""
        pm = _PUB.search(block)
        if pm:
            # pub 行形如：
            #   [美] 罗斯、威斯特菲尔德 / 崔方南 / 机械工业出版社 / 2020-6-1 / 99.00元
            #   〔英〕霍金 / 湖南科学技术出版社 / 2010-4 / 45.00元
            parts = [p.strip() for p in _TAG.sub("", html.unescape(pm.group(1))).split("/")]
            parts = [p for p in parts if p]
            # 出版社：第一个带「出版社/书局/Press」字样的片段
            for p in parts:
                if _PUBLISHER_HINT.search(p):
                    publisher = p
                    break
            # 作者：第一段。但要排掉「出版信息」类条目（作者段本身就是出版社/日期）
            if parts and not _PUBLISHER_HINT.search(parts[0]):
                author = parts[0]

        books.append({"title": title, "author": author, "publisher": publisher})
    return books


def crawl(tags: list[str], max_per_tag: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for tag in tags:
        print(f"\n▶ 标签：{tag}")
        for start in range(0, max_per_tag, 20):
            url = (
                "https://book.douban.com/tag/"
                + urllib.parse.quote(tag)
                + f"?start={start}"
            )
            page = fetch(url)
            if page is None:
                break

            books = parse_page(page)
            if not books:
                # 没有条目 = 翻到底了（或页面结构变了）
                print(f"    start={start} 没有解析到条目，换下一个标签")
                break

            fresh = 0
            for b in books:
                key = b["title"]
                if key not in seen:
                    seen[key] = b
                    fresh += 1
            print(f"    start={start:<4} 解析 {len(books):>2} 条，新增 {fresh:>2}，累计 {len(seen)}")
            time.sleep(DELAY_SECONDS)

    return list(seen.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取豆瓣真实书目作为生成种子")
    parser.add_argument("--tags", help="逗号分隔的标签，默认用内置的 18 个")
    parser.add_argument("--max-per-tag", type=int, default=200,
                        help="每个标签最多取多少条（默认 200 = 10 页）")
    args = parser.parse_args()

    tags = args.tags.split(",") if args.tags else DEFAULT_TAGS
    print(f"准备抓取 {len(tags)} 个标签，每标签上限 {args.max_per_tag} 条")
    print(f"每请求间隔 {DELAY_SECONDS}s，预计 {len(tags) * args.max_per_tag / 20 * DELAY_SECONDS / 60:.0f} 分钟左右")

    books = crawl(tags, args.max_per_tag)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(books, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with_pub = sum(1 for b in books if b["publisher"])
    with_author = sum(1 for b in books if b["author"])
    print()
    print("=" * 60)
    print(f"共抓到 {len(books)} 本不重复的书 → {OUT_PATH}")
    print(f"  有出版社: {with_pub} ({with_pub / max(len(books), 1):.0%})")
    print(f"  有作者:   {with_author} ({with_author / max(len(books), 1):.0%})")
    print()
    print("样例：")
    for b in books[:5]:
        print(f"  {b['title']}  |  {b['author']}  |  {b['publisher']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
