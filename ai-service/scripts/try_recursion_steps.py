"""量清楚：框架版里「一轮工具调用」占几步，5 轮该设多少。

思路：用 stream_mode="updates" 把**每个节点执行**按顺序打出来 ——
那串节点名就是「步」。一轮和两轮各测一次，差值就是每轮的步数。

跑一次约 3~5 次模型调用。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent

from app.agent.loop import SYSTEM_PROMPT
from app.agent.tools import get_post_detail, search_books
from app.services.llm import get_llm

agent = create_agent(
    model=get_llm(temperature=0.0),
    tools=[search_books, get_post_detail],
    system_prompt=SYSTEM_PROMPT,
)


def measure(tag: str, question: str, history=None):
    print("=" * 72)
    print(f"{tag}：{question}")
    print("=" * 72)

    msgs = list(history or []) + [{"role": "user", "content": question}]
    nodes = []
    final = None

    for chunk in agent.stream(
        {"messages": msgs},
        stream_mode="updates",
        config={"recursion_limit": 100},      # 放很大，只为看它自然跑完
    ):
        for node_name, update in chunk.items():
            nodes.append(node_name)
            if update and "messages" in update:
                final = update["messages"]

    print(f"    节点执行顺序（{len(nodes)} 步）：")
    for i, n in enumerate(nodes):
        print(f"        [{i}] {n}")

    # 数 tools 节点执行了几次 —— 那才是「工具轮数」。
    # （不能数消息里的 tool_calls：final 只装着最后一次更新，看不到前面几轮。）
    rounds = nodes.count("tools")
    print(f"    实际工具轮数：{rounds}（= tools 节点出现 {rounds} 次）")
    print(f"    → 照这个规律，等效的 recursion_limit 至少要 "
          f"{len(nodes) + 1}（实测：节点数 + 1）")
    print()
    return len(nodes), rounds, final


s1, r1, f1 = measure("第一组：只搜一次", "有高数吗")
s2, r2, f2 = measure("第二组：先搜再查详情", "最便宜的线代的成色怎么样")

print("=" * 72)
print("结论")
print("=" * 72)
print(f"    一轮：节点 {s1} 个 —— 应该等于 2×1+1 = 3")
print(f"    两轮：节点 {s2} 个 —— 应该等于 2×2+1 = 5")
print()
print("    规律：N 轮 = 2N + 1 个节点；recursion_limit 至少要是 2N + 2。")
print("    （实测：3 个节点时 limit=3 被拦、4 才过；5 个节点时 5 被拦、6 才过。）")
for n in [1, 2, 3, 5]:
    print(f"    {n} 轮 → {2 * n + 1:>2} 个节点 → recursion_limit ≥ {2 * n + 2}")
print()
print("    → MAX_ROUNDS=5 的等效值是 recursion_limit = 12")
print()
print("    ⚠ 5 轮那一行是**推算**的，不是实测：没法命令模型正好调 5 轮。")
print("      公式在 1 轮和 2 轮两点上都对上了，所以按线性外推。")
