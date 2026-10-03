"""M2 试跑 + 调阈值：建索引、跑正反例、把原始分数打出来。

用法（在 D:\\agent-book\\ai-service 下，先激活 venv）：

    python scripts\\try_search.py                 # 重建索引 + 跑全部正反例
    python scripts\\try_search.py --no-build      # 不重建，直接查（改阈值时用这个，快）
    python scripts\\try_search.py -q 同济版高数    # 只查一条，看 top10 明细

## 为什么这个脚本要绕过阈值直接查向量库

调阈值的前提是**看见没过滤的原始分数**。
如果走 /search 接口，低于阈值的都被丢掉了，你根本不知道「反例的最高分是多少」，
也就没法判断阈值定得合不合适。

所以这里直接调 vector_store.query，自己打印分布。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import embedding, search, vector_store  # noqa: E402

# ------------------------------------------------------------
# 正例：应该命中。覆盖「字面完全不同但意思相同」的几种说法
POSITIVE = [
    "同济版高数",          # 简称 + 出版社
    "高等数学",            # 标准名
    "高数同济第七版",       # 简称 + 版次
    "线性代数 同济",        # 另一门课
    "计算机网络 谢希仁",     # 有明确作者的
    "大英 新视野",          # 简称
    "考研数学复习书",       # 连书名都没说，只说用途 —— 这条最难
]

# 反例：库里绝对没有，必须返回空
NEGATIVE = [
    "量子力学",
    "有机化学",
    "民法典注释本",
    "今天天气怎么样",       # 完全无关的日常对话
]

TOP_K = 10


def show(label: str, query: str) -> float | None:
    """打印一条 query 的完整 top-k，返回最高分。"""
    qv = embedding.embed_query(query)
    hits = vector_store.query(qv, top_k=TOP_K)

    top = hits[0]["score"] if hits else None
    print(f"\n【{label}】{query!r}　→ 最高分 {top:.4f}" if top is not None else f"\n【{label}】{query!r}　→ 库是空的")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        star = "  ←" if i == 1 else ""
        print(
            f"   {i:2d}. {h['score']:+.4f}  "
            f"{m.get('book_name',''):<12} {m.get('edition','') or '':<8} "
            f"{(m.get('publisher') or '')[:12]:<14} id={h['id']}{star}"
        )
    return top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-build", action="store_true", help="跳过建索引，直接查")
    parser.add_argument("--rebuild", action="store_true", help="清空向量库后重建")
    parser.add_argument("-q", "--query", help="只查这一条，看 top10 明细")
    args = parser.parse_args()

    if not args.no_build:
        print("=" * 78)
        print("建索引中（要调主服务的 API，确认 Spring Boot 起着）...")
        try:
            stats = search.build_index(limit=500, rebuild=args.rebuild)
        except NotImplementedError as e:
            print(f"\n❌ {e}")
            print("\n先去 app/services/search.py 把 compose_text() 实现了再跑。")
            return
        except Exception as e:
            print(f"\n❌ {type(e).__name__}: {e}")
            print("\n如果报「连不上主服务」：确认 Spring Boot 已启动、MySQL 已启动。")
            return
        print(f"✅ {stats}")

    if args.query:
        show("单查", args.query)
        return

    print("\n" + "=" * 78)
    print(f"向量库共 {vector_store.count()} 条\n")
    print("=" * 78)
    print("正例（应该命中）")
    print("=" * 78)
    pos_tops = [show("正例", q) for q in POSITIVE]

    print("\n" + "=" * 78)
    print("反例（库里没有，分数应该明显更低）")
    print("=" * 78)
    neg_tops = [show("反例", q) for q in NEGATIVE]

    # ---- 汇总：这一屏就是用来定阈值的 ----
    pos_tops = [t for t in pos_tops if t is not None]
    neg_tops = [t for t in neg_tops if t is not None]

    print("\n" + "=" * 78)
    print("分数分布汇总 —— 阈值就定在两组之间的那条缝上")
    print("=" * 78)
    if pos_tops and neg_tops:
        print(f"  正例最高分：min={min(pos_tops):.4f}  max={max(pos_tops):.4f}")
        print(f"  反例最高分：min={min(neg_tops):.4f}  max={max(neg_tops):.4f}")
        print(f"  缝隙：{max(neg_tops):.4f}  ~  {min(pos_tops):.4f}")
        if max(neg_tops) >= min(pos_tops):
            print("  ⚠️ 两组有重叠 —— 单一阈值分不干净。")
            print("     要么是某些 query 本身歧义太大，要么是 compose_text 的拼法要调。")
            print("     先看上面哪条正例拖后腿，再决定是改拼法还是放弃那条 query。")
        else:
            print(f"  → 建议阈值取中间值：{(max(neg_tops) + min(pos_tops)) / 2:.2f}")
            print("     偏保守（宁可漏也别错）就往高了取，比如 "
                  f"{min(pos_tops) - 0.02:.2f}")


if __name__ == "__main__":
    main()
