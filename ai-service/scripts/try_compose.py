"""看数据：不同的「检索文本拼法」到底会检索出什么。

**这个脚本不写向量库、不建索引，全程在内存里算。**
因为 220 条数据用不着向量数据库 —— 直接算点积就行，快得多，
也少一层看不见的东西。（向量库是给几百万条用的，这里只是顺手。）

目的：让你在动手写 compose_text() 之前，先看见
  - 真实帖子长什么样（哪些字段是 None）
  - 不同的拼法分别会拼出什么文本
  - 同一个查询，不同拼法下的实际排名差多少

用法（在 D:\\agent-book\\ai-service 下，先激活 venv，且 Spring Boot 要开着）：
    python scripts\\try_compose.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ⚠️ 让 print 立刻吐出来，不要攒在缓冲区里。
#
# Python 的 stdout 在**不是终端**的时候（VS Code 输出面板、重定向到文件、
# 被别的程序调用）是**块缓冲**的 —— 攒够 4KB 才往外写一次。
# 这个脚本要跑 20 多秒，中间几乎没输出，块缓冲会让屏幕上**一行都不显示**，
# 看起来像卡死了。加上这一行，每条 print 都立即生效。
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from app.clients import book_service  # noqa: E402
from app.services import embedding  # noqa: E402

SAMPLE_COUNT = 6
TOP_K = 3


# ============================================================
# 几种候选拼法。都不是标准答案，看完结果你自己判断。
# ============================================================


def only_name(p: dict) -> str:
    """① 只有书名。最干净，也最省，但信息量最少。"""
    return p.get("bookName") or ""


def reference(p: dict) -> str:
    """② docs/需求与接口.md §5.1 的参考拼法。
    书名 + 版次 + 作者 + 出版社 + 成色。不拼价格、不拼原文。"""
    fields = [
        p.get("bookName"),
        p.get("edition"),
        p.get("author"),
        p.get("publisher"),
        p.get("conditionDesc"),
    ]
    return " ".join(x for x in fields if x)


def name_boosted(p: dict) -> str:
    """③ 书名重复两次「加重权重」，去掉成色。
    思路：书名是最重要的信号，重复出现能让它在向量里占更大比重。
    去掉成色是因为「九成新」这种词对「这是哪本书」毫无帮助。"""
    fields = [
        p.get("bookName"),
        p.get("bookName"),
        p.get("edition"),
        p.get("publisher"),
    ]
    return " ".join(x for x in fields if x)


def reference_plus_raw(p: dict) -> str:
    """④ 参考拼法 + 卖家原文。
    原文信息最全（口语说法都在里面），但「40」「包邮」「可小刀」也是噪音。"""
    base = reference(p)
    raw = p.get("rawText") or ""
    return f"{base} {raw}".strip()

def reference_min(p:dict) -> str:
    """只拼身份信息：书名+版次+出版社"""
    fields = [
        p.get("bookName"),
        p.get("edition"),
        p.get("publisher"),
    ]
    return " ".join(x for x in fields if x)


STRATEGIES = [
    ("① 只有书名", only_name),
    ("② §5.1 参考拼法", reference),
    ("③ 书名加强、去掉成色", name_boosted),
    ("④ 参考拼法 + 原文", reference_plus_raw),
    ("⑤ 只拼身份（书名+版次+出版社）", reference_min),  # ← 你写的
]

QUERIES = [
    "同济版高数",        # 正例：简称 + 出版社
    "想买本数据结构",     # 正例：口语，带「想买本」这种废话
    "便宜的线代",        # 正例：带条件
    "量子力学",          # 反例：库里绝对没有
]

# 哪些是「库里根本没有」的反例 —— 汇总时用来算缝宽
NEGATIVE_QUERIES = {"量子力学"}


def preview(text: str, width: int = 58) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    print("=" * 84)
    print("try_compose.py —— 对比不同「检索文本拼法」的检索效果")
    print("=" * 84)
    print("先连 Spring Boot 取数据，再加载 BGE 模型（第一次要十几秒），")
    print("然后向量化 4 种拼法 × 全部帖子。总共约 20~30 秒，中途没输出是正常的。\n")

    # ---------- 第一步：把真实数据摆出来 ----------
    print("[1/3] 正在从 Spring Boot 取帖子 ...")
    try:
        posts = book_service.fetch_posts(extract_status="DONE", limit=500)
    except Exception as e:
        print(f"❌ 拿不到数据：{e}")
        print("\n确认 Spring Boot 起着（IDEA 里跑 BookServiceApplication）。")
        return

    usable = [p for p in posts if p.get("bookName")]
    print("=" * 84)
    print(f"主服务返回 {len(posts)} 条 DONE 帖子，其中 {len(usable)} 条有书名")
    print("=" * 84)
    print("\n先看清楚一条帖子到底长什么样（挑 6 条）：\n")

    for p in usable[:SAMPLE_COUNT]:
        print(f"  id={p['id']}")
        for label, key in [
            ("书名", "bookName"),
            ("版次", "edition"),
            ("作者", "author"),
            ("出版社", "publisher"),
            ("成色", "conditionDesc"),
            ("价格", "price"),
        ]:
            v = p.get(key)
            mark = "  ← 抽不到，是 None" if v is None else ""
            print(f"      {label:<4} {str(v):<28}{mark}")
        print(f"      原文 {preview(p.get('rawText') or '', 60)}")
        print()

    print("=" * 84)
    print("上面这几条，用四种拼法分别会拼成：")
    print("=" * 84)
    for label, fn in STRATEGIES:
        print(f"\n{label}")
        for p in usable[:2]:
            print(f"    id={p['id']:<4} {preview(fn(p), 66)}")

    # ---------- 第二步：四种拼法各自建一套向量 ----------
    print("\n" + "=" * 84)
    print(f"[2/3] 正在向量化（{len(STRATEGIES)} 种拼法 × {len(usable)} 条帖子 "
          f"= {len(STRATEGIES) * len(usable)} 次），加载模型 + 算完约 15~25 秒 ...")
    print("=" * 84)

    docs = [fn(p) for _, fn in STRATEGIES for p in usable]
    # 转成 numpy 数组才能用 @ 算子。
    # embedding.embed_documents() 返回的是 list[list[float]]（为了能直接喂给 Chroma），
    # 而 Python 原生的 list 不支持 @ —— 只有 numpy 数组 / torch 张量支持。
    all_vecs = np.array(embedding.embed_documents(docs), dtype=np.float32)
    n = len(usable)
    # 切成 4 份，每份对应一种拼法
    vec_sets = [all_vecs[i * n : (i + 1) * n] for i in range(len(STRATEGIES))]
    print("完成。\n")

    # ---------- 第三步：同一个查询，看四种拼法的排名 ----------
    # tops[strategy_index][query] = 该查询在该拼法下的最高分
    tops: dict[int, dict[str, float]] = {i: {} for i in range(len(STRATEGIES))}

    print(f"\n[3/3] 开始查询对比（{len(QUERIES)} 个查询）\n")

    for q in QUERIES:
        print("=" * 84)
        print(f"查询：{q!r}")
        print("=" * 84)
        qv = np.array(embedding.embed_query(q), dtype=np.float32)
        for si, ((label, _), vecs) in enumerate(zip(STRATEGIES, vec_sets)):
            # 向量都是归一化过的（normalize_embeddings=True），
            # 所以矩阵乘出来的直接就是余弦相似度，不用再除模长。
            scores = vecs @ qv
            order = np.argsort(-scores)  # 降序的下标
            tops[si][q] = float(scores[order[0]])

            print(f"\n  {label}")
            for rank, idx in enumerate(order[:TOP_K], 1):
                p = usable[int(idx)]
                print(
                    f"      {rank}. {scores[idx]:+.4f}  "
                    f"{str(p.get('bookName')):<12} "
                    f"{str(p.get('edition') or ''):<8} "
                    f"{str(p.get('publisher') or ''):<14} "
                    f"{str(p.get('conditionDesc') or '')[:14]}"
                )
            print(f"      （第 1 名和第 {TOP_K} 名差 "
                  f"{float(scores[order[0]] - scores[order[TOP_K - 1]]):+.4f}）")

    # ---------- 第四步：汇总，看哪种拼法「缝最宽」 ----------
    positives = [q for q in QUERIES if q not in NEGATIVE_QUERIES]
    negatives = [q for q in QUERIES if q in NEGATIVE_QUERIES]

    print("\n" + "=" * 84)
    print("汇总：正例最低分 vs 反例最高分")
    print("=" * 84)
    print("  阈值要在「反例最高分」和「正例最低分」之间找。缝越宽越好定。\n")
    print(f"  {'拼法':<22}{'正例最低':>10}{'反例最高':>10}{'缝宽':>10}")
    for si, (label, _) in enumerate(STRATEGIES):
        pmin = min(tops[si][q] for q in positives)
        nmax = max(tops[si][q] for q in negatives)
        gap = pmin - nmax
        flag = "  ✅" if gap > 0 else "  ❌ 重叠了，单阈值分不开"
        print(f"  {label:<22}{pmin:>10.4f}{nmax:>10.4f}{gap:>10.4f}{flag}")

    print("\n" + "=" * 84)
    print("看完了，你自己判断：")
    print("  - 哪种拼法让正例的最低分尽量高、反例的最高分尽量低？")
    print("  - 「③ 书名重复」真的比「② 参考拼法」强吗？")
    print("  - 「④ 加原文」的 top-3 里混进了什么？变好还是变差？")
    print("  - 有些拼法下多条帖子分数完全相同（比如都是 +0.6258），为什么？")
    print("    —— 想想什么情况下两条不同的帖子会拼出**一模一样**的文本。")
    print("=" * 84)


if __name__ == "__main__":
    main()
