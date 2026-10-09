"""证明「余弦阈值已经调不动了」—— 用数据，不是用感觉。

## 这个脚本回答的问题

调阈值的时候，脑子里想的是：「正例分数高、负例分数低，中间切一刀不就行了？」

在 220 条语料上确实如此。但语料涨到 1 万条之后，**这个前提不成立了**。
这个脚本把每条查询的 top-10 分数分布拉出来，然后逐个检验候选规则
（top1 / top1-top2 差值 / top1 减均值 / top1 除均值 / 标准差）能不能把
「正例」和「越界查询」（库里没有、但分数很高的那些）分开。

## 怎么读结果

每个统计量给出两个区间：正例的 [min, max] 和越界的 [min, max]。
**只要两个区间有重叠，这个规则就不可能用** —— 因为它是单调的，
只能一刀切，切在重叠区里必然两边都误伤。

初始 top-1 上，正例下界 0.6187 > 越界上界 0.7301？不成立 —— 重叠。

## 为什么留着这个脚本

因为「阈值调不动」是个**反直觉的结论**，光靠嘴说没人信（包括几个月后的自己）。
面试被问「阈值怎么定的」，能直接跑出这张表、指出「正负样本已经重叠，
越界查询的置信度甚至高过正例」——这比「调到一个看起来不错的数」强得多，
而且它直接导出了下一步（rerank）。

用法（先确保两个服务都起着、索引已建）：

    .venv\\Scripts\\python.exe scripts\\analyze_threshold.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import httpx  # noqa: E402

from scripts.eval_search import CASES, KNOWN_LEAKS  # noqa: E402

AI_SERVICE = "http://127.0.0.1:8000"
TOP_K = 10

# ⚠️ 必须显式传 0.0，不能用默认阈值。
# 默认阈值会先把低分的滤掉，那我们看到的就是「已经被闸门切过一刀的分布」，
# 拿它去分析「闸门该切在哪」是循环论证 —— 分布已经被这次要评估的规则污染了。
NO_THRESHOLD = 0.0


def fetch_scores(client: httpx.Client, query: str) -> list[float]:
    resp = client.post(
        f"{AI_SERVICE}/search",
        json={"query": query, "top_k": TOP_K, "min_score": NO_THRESHOLD},
        timeout=120,
    )
    resp.raise_for_status()
    return [h["score"] for h in resp.json()["hits"]]


def main() -> None:
    client = httpx.Client()

    groups: dict[str, list[tuple[str, list[float]]]] = {
        "正例": [(c.query, fetch_scores(client, c.query))
                 for c in CASES if not c.is_negative],
        # 负例（能返回空的那些）也打出来：它们是「阈值现在就能拦住」的对照组，
        # 缺了它们就看不出「越界查询」到底有多离谱。
        "负例": [(c.query, fetch_scores(client, c.query))
                 for c in CASES if c.is_negative],
        "越界": [(q, fetch_scores(client, q)) for q, _name, _before in KNOWN_LEAKS],
    }

    print("=" * 84)
    print("各条查询的分数分布（min_score=0，看的是**未过滤**的原始分布）")
    print("=" * 84)
    print(f'{"组":<4}{"查询":<16}{"top1":>9}{"top2":>9}{"gap":>9}'
          f'{"均值":>9}{"标准差":>10}')

    for group, items in groups.items():
        for query, scores in items:
            if len(scores) < 2:
                # 返回不足 2 条时 gap / 标准差都没有意义，直接标出来而不是硬算。
                print(f'{group:<4}{query:<16}  —— 只返回 {len(scores)} 条，统计量无从谈起'
                      f'（这条本身就是最好的负例）')
                continue
            print(f'{group:<4}{query:<16}'
                  f'{scores[0]:>9.4f}{scores[1]:>9.4f}{scores[0] - scores[1]:>9.4f}'
                  f'{statistics.mean(scores):>9.4f}{statistics.pstdev(scores):>10.4f}')

    # ── 逐个检验候选规则 ────────────────────────────────────────────────
    # 约定：规则的值**越大越像正例**（分数高 / 比第二名高得多 / 比均值高得多）。
    # 所以判据是「正例的最小值」是否严格大于「越界的最大值」——
    # 是的话就能在前者上方、后者下方切一刀。
    rules = [
        ("top1", lambda s: s[0]),
        ("gap(top1-top2)", lambda s: s[0] - s[1]),
        ("top1 - 均值", lambda s: s[0] - statistics.mean(s)),
        ("top1 / 均值", lambda s: s[0] / statistics.mean(s) if statistics.mean(s) else 0.0),
        ("标准差", lambda s: statistics.pstdev(s)),
        ("均值", lambda s: statistics.mean(s)),
    ]

    def span(group: str, fn) -> tuple[float, float] | None:
        vals = [fn(s) for _q, s in groups[group] if len(s) >= 2]
        return (min(vals), max(vals)) if vals else None

    print()
    print("=" * 84)
    print("这些统计量能不能把「正例」和「越界」分开")
    print("=" * 84)
    print("  判据：正例的最小值 > 越界的最大值（严格大于才能切一刀）")
    print()

    any_works = False
    for name, fn in rules:
        pos, leak = span("正例", fn), span("越界", fn)
        if pos is None or leak is None:
            continue
        separable = pos[0] > leak[1]
        any_works = any_works or separable
        print(f'  {name:<16} 正例 [{pos[0]:>8.4f}, {pos[1]:>8.4f}]'
              f'   越界 [{leak[0]:>8.4f}, {leak[1]:>8.4f}]'
              f'   {"✅ 可分" if separable else "❌ 重叠"}')

    print()
    if any_works:
        print("  有可分的规则 —— 按上面那个统计量重写 search() 的打分逻辑。")
        print("  ⚠️ 但要先在**另一个**测试集上验证，别在这套用例上调到过拟合。")
    else:
        print("  **全部重叠。** 结论：余弦阈值（以及任何它的单调变形）已经到顶了。")
        print()
        print("  最扎眼的一条证据：正例「同济版高数」的 top1-均值 = 0.0018，")
        print("  而越界「教育学」是 0.0070 —— **越界查询看起来比正例还自信**。")
        print()
        print("  根因：双塔（bi-encoder）把 query 和文档各自压成一个向量再比夹角，")
        print("  夹角只反映「整体语义方向」，分不出「这个词是不是真的指同一件事」。")
        print("  调阈值是在一个**分不开的空间**里找分割面，所以必然失败。")
        print()
        print("  下一步不是继续调参，是换打分器：rerank（cross-encoder）把 query 和")
        print("  文档**拼在一起**送进模型逐对判分，能看见字面的交叉，恰好吃掉这一类。")
    print("=" * 84)


if __name__ == "__main__":
    main()
