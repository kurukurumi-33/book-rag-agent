"""标定 rerank 的阈值 —— 顺便验证「换了打分器，原来分不开的样本就分开了」。

## 对照着 analyze_threshold.py 看

`analyze_threshold.py` 证明了**余弦**这一路已经到顶：正例和越界查询的
top-1 分数区间重叠，6 种候选统计量没有一种能切开。最扎眼的一条是
「越界查询的 top1−均值 比正例还高」。

这个脚本跑同样一批查询，但看的是 **`rerank_score`**（cross-encoder 的输出，
sigmoid 之后的 0~1 值）。如果 cross-encoder 真的解决了问题，那么：

    正例的 rerank_score 最小值  >  越界的 rerank_score 最大值

—— 也就是**区间不再重叠**。这就是「换打分器」和「调参」的区别：
调参是在分不开的空间里换刀刃角度，换打分器是换一个分得开的空间。

## 怎么读结果

1. 先看第一张表，逐条对比同一批查询在两种分下的表现
2. 再看「能不能分开」那一段 —— 会直接给出 ✅ / ❌
3. 最后一张表是扫描：每个候选阈值拦下几条越界、误杀几条正例，
   自己挑一个（本项目目前取 rerank_min_score 见 config.py）

用法（两个服务都起着、索引已建）：

    .venv\\Scripts\\python.exe scripts\\tune_rerank.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import httpx  # noqa: E402

from scripts.eval_search import CASES, KNOWN_LEAKS  # noqa: E402

AI_SERVICE = "http://127.0.0.1:8000"
TOP_K = 10

# 和 analyze_threshold.py 一样，必须显式传 0.0。
# 用默认阈值的话，我们看到的分布已经被**余弦闸门**切过一刀了，
# 那测的是「余弦闸门 + 精排」的联合效果，分不清是谁的功劳。
NO_THRESHOLD = 0.0


def top1_of(client: httpx.Client, query: str) -> dict | None:
    """返回精排后的 top-1 命中（没有就 None）。"""
    resp = client.post(
        f"{AI_SERVICE}/search",
        json={"query": query, "top_k": TOP_K, "min_score": NO_THRESHOLD, "rerank": True},
        timeout=180,
    )
    resp.raise_for_status()
    hits = resp.json()["hits"]
    return hits[0] if hits else None


def main() -> None:
    client = httpx.Client()

    positives = [c.query for c in CASES if not c.is_negative]
    negatives = [c.query for c in CASES if c.is_negative]
    leaks = [(q, name) for q, name, _ in KNOWN_LEAKS]

    print("=" * 88)
    print("逐条对比：同一批查询在「余弦」和「精排」两把尺子下的 top-1")
    print("=" * 88)
    print(f'{"组":<5}{"查询":<16}{"余弦":>9}{"精排":>9}   命中的书')
    print("-" * 88)

    group_scores: dict[str, list[float]] = {"正例": [], "越界": [], "负例": []}

    def show(group: str, query: str) -> None:
        hit = top1_of(client, query)
        if hit is None:
            print(f'{group:<5}{query:<16}{"  —  ":>9}{"  —  ":>9}   (空)')
            return
        rr = hit.get("rerank_score")
        if rr is not None:
            group_scores[group].append(rr)
        name = (hit.get("book_name") or hit.get("text") or "?")[:34]
        print(f'{group:<5}{query:<16}{hit["score"]:>9.4f}'
              f'{(f"{rr:.4f}" if rr is not None else "  —  "):>9}   {name}')

    for q in positives:
        show("正例", q)
    for q in leaks:
        show("越界", q[0])
    for q in negatives:
        show("负例", q)

    # ── 判据：正例最小值 > 越界最大值 ────────────────────────────────────
    print()
    print("=" * 88)
    print("换了打分器之后，能不能分开")
    print("=" * 88)

    pos_rr = group_scores["正例"]
    leak_rr = group_scores["越界"]
    if not pos_rr or not leak_rr:
        print("  数据不足，先确认服务起着、索引建了。")
        return

    print(f'  正例 rerank_score   [{min(pos_rr):>7.4f}, {max(pos_rr):>7.4f}]'
          f'   （{len(pos_rr)} 条）')
    print(f'  越界 rerank_score   [{min(leak_rr):>7.4f}, {max(leak_rr):>7.4f}]'
          f'   （{len(leak_rr)} 条）')
    print()
    if min(pos_rr) > max(leak_rr):
        print(f'  ✅ 分开了。缝在 {max(leak_rr):.4f} ~ {min(pos_rr):.4f} 之间。')
        print("     这正是 cross-encoder 该有的样子：它看见了字面的交叉，")
        print("     所以「核工程 / Java核心技术」这种纯垃圾被压到了 0.5 附近（=不相关），")
        print("     而「同济版高数 / 高等数学」稳稳在 0.7 以上。")
    else:
        print("  ❌ 仍然重叠 —— cross-encoder 没解决这一类，得回去看是不是")
        print("     候选池（rerank_pool）里根本没捞到对的那本书。")
        print("     ⚠️ 注意区分：rerank 只在候选池内重排，粗排没捞到的它救不了。")

    # ── 扫描候选阈值 ────────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("扫描：每个阈值拦下几条越界、误杀几条正例")
    print("=" * 88)
    print(f'{"阈值":>8}{"越界被拦":>10}{"正例存活":>10}{"负例存活":>10}   备注')
    print("-" * 88)

    neg_rr = group_scores["负例"]
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        blocked = sum(1 for v in leak_rr if v < t)
        pos_alive = sum(1 for v in pos_rr if v >= t)
        neg_alive = sum(1 for v in neg_rr if v >= t)
        note = ""
        if pos_alive == len(pos_rr) and blocked == len(leak_rr):
            note = "← 两边都满分"
        print(f'{t:>8.2f}{blocked:>7}/{len(leak_rr)}{pos_alive:>7}/{len(pos_rr)}'
              f'{neg_alive:>7}/{len(neg_rr)}   {note}')

    print()
    print("  ⚠️ 判据不是「拦得越多越好」：漏掉一条越界只是多一条垃圾，")
    print("     误杀一条正例是用户搜不到本来有的书 —— 后者的代价更大。")
    print("=" * 88)


if __name__ == "__main__":
    main()
