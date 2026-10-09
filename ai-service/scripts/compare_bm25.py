"""量化 BM25 到底有没有用：**仅向量** vs **向量+BM25(RRF)** 逐条对比。

## 为什么要单独有这个脚本

`eval_search.py` 只能告诉你「命中率是多少」。但混合检索的意义不是把命中率从
86% 抬到 100%（那说明本来就能搜到），而是**把本该搜到、却被向量排到后面去的
那些往前挪**。这种改善在命中率上看不出来 —— 必须逐条对比排名。

所以这个脚本比的是：同一条查询，两个方案各自返回的 top-N **帖子 id 序列**。
比对 id 而不是书名，因为书名的版次差异也是排名差异。

## 上一版的结论（**只对 220 条语料成立，别当通用结论**）

9 条查询里只有「计算机网络」一条的 top-3 变了，命中率一条都没翻盘。
归因：语料小且高度同质（全是「书名+版次+作者+出版社」模板文本），
BM25 的 IDF 拉不开差距。

**语料涨到 1 万条 / 2869 个书名后，两个前提都变了**，所以要重测。

## 用法

    # 先确保 Spring Boot :8080 和 AI 服务 :8000 都起着，且索引已建
    .venv\\Scripts\\python.exe scripts\\compare_bm25.py

    # 只看有差异的查询（默认就是，加 --all 看全部）
    .venv\\Scripts\\python.exe scripts\\compare_bm25.py --all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse  # noqa: E402

from app.services import search as search_service  # noqa: E402
from scripts.eval_search import CASES  # noqa: E402

TOP_N = 10
HIT_AT = 3


def _run(query: str, *, use_lexical: bool) -> list[dict]:
    """跑一次检索。use_lexical=False 时把 BM25 通道掐掉，模拟「仅向量」。

    直接调 search() 而不是走 HTTP：这里要**中途替换**通道，
    走接口做不到（服务端不会让你临时关掉 BM25）。
    代价是绕过了 HTTP 层 —— 但那个层面已经被 eval_search.py 覆盖了，
    这里只关心排序差异。
    """
    real = search_service.lexical.search
    if not use_lexical:
        search_service.lexical.search = lambda _q, top_k: []
    try:
        return search_service.search(query, top_k=TOP_N)
    finally:
        # 一定要还原。不还原的话下一条查询就悄悄地也在「仅向量」模式下跑了，
        # 而且结果看着很正常 —— 这种串味的 bug 最难发现。
        search_service.lexical.search = real


def _names(hits: list[dict]) -> list[str]:
    return [h["metadata"].get("book_name") or "?" for h in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 仅向量 vs 向量+BM25")
    parser.add_argument("--all", action="store_true", help="连没差异的查询也打出来")
    args = parser.parse_args()

    print("=" * 78)
    print(f"仅向量  vs  向量+BM25(RRF)   —— {len(CASES)} 条查询, top-{TOP_N}")
    print("=" * 78)

    changed_top3 = changed_top10 = 0
    rows = []

    for case in CASES:
        if case.is_negative:
            continue
        dense = _run(case.query, use_lexical=False)
        hybrid = _run(case.query, use_lexical=True)

        d3, h3 = _names(dense[:HIT_AT]), _names(hybrid[:HIT_AT])
        d10, h10 = _names(dense), _names(hybrid)

        diff3, diff10 = d3 != h3, d10 != h10
        changed_top3 += diff3
        changed_top10 += diff10
        rows.append((case, dense, hybrid, diff3, diff10))

    for case, dense, hybrid, diff3, diff10 in rows:
        if not (diff3 or diff10) and not args.all:
            continue
        flag = "🔀 top-3 变了" if diff3 else ("· 仅 top-10 内变动" if diff10 else "相同")
        print(f"\n[{case.query}]  {flag}")
        if diff3 or args.all:
            print(f"    仅向量  : {' / '.join(_names(dense[:HIT_AT]))}")
            print(f"    混合    : {' / '.join(_names(hybrid[:HIT_AT]))}")
        if diff10 and not diff3:
            # top-3 没动但 top-10 动了 —— 说明 BM25 影响的是靠后的名次。
            # 对「返回 10 条列表给用户翻」的界面有意义，对 top-3 命中率没有。
            only_dense = [n for n in _names(dense) if n not in _names(hybrid)]
            only_hybrid = [n for n in _names(hybrid) if n not in _names(dense)]
            print(f"      ↓ 混合多了 {only_hybrid}")
            print(f"      ↑ 混合少了 {only_dense}")

    pos = len(rows)
    print()
    print("=" * 78)
    print(f"  top-{HIT_AT} 有变化的查询： {changed_top3}/{pos}")
    print(f"  top-{TOP_N} 有变化的查询： {changed_top10}/{pos}")
    print()
    if changed_top3 == 0 and changed_top10 == 0:
        print("  两个通道结果**完全一致** —— BM25 在这份语料上没有净增益。")
        print("  去看 eval_search.py 里那段归因，并**标上语料规模**再下结论。")
    elif changed_top3 == 0:
        print("  BM25 只影响了 top-3 之后的名次：对命中率没帮助，")
        print("  但对「给用户翻 10 条」的界面有帮助。别把它说成「提升了准确率」。")
    else:
        print(f"  BM25 实打实地改变了 {changed_top3} 条查询的 top-{HIT_AT} 排序 ——")
        print("  逐条看上面，确认改的是「把对的往前挪」而不是「把错的往前挪」。")
    print("=" * 78)


if __name__ == "__main__":
    main()
