"""不调模型，把你那段扫描逻辑原样抄下来，喂一份手造的多轮消息。

目的：看清两个只在「多轮」或「有工具回执」时才暴露的问题。
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 手造一份【两轮】对话 —— 单轮看不出问题，必须两轮
out_msgs = [
    HumanMessage("最便宜的线代，成色怎么样"),
    AIMessage(content="", tool_calls=[           # [1] 工单 1
        {"name": "search_books", "args": {"query": "线性代数"}, "id": "c1", "type": "tool_call"}]),
    ToolMessage(content='[{"post_id": 88}, {"post_id": 91}]', tool_call_id="c1", name="search_books"),
    AIMessage(content="", tool_calls=[           # [3] 工单 2
        {"name": "get_post_detail", "args": {"post_id": 88}, "id": "c2", "type": "tool_call"}]),
    ToolMessage(content='{"post_id": 88, "condition": "九成新"}', tool_call_id="c2", name="get_post_detail"),
    AIMessage(content="编号 88 那本九成新，18 元。"),   # [5] 最终回答
]


def 你的版本():
    """跟你 loop.py 里那段一字不差的缩进与判断。"""
    trace = []
    for i, m in enumerate(out_msgs):
        if not isinstance(m, AIMessage):
            continue

        if not m.tool_calls:
            continue

        for call in m.tool_calls:
            count = 0
            for later in out_msgs[i + 1:]:
                if not isinstance(later, AIMessage):      # ← 你写的
                    continue
                if later.id != call["id"]:                # ← 你写的
                    continue
                try:
                    count = len(json.loads(later.content))
                except Exception:
                    count = 0
                break
            trace.append({
                "name": call["name"],
                "arguments": call["args"],
                "result_count": count,
            })
        reply = out_msgs[-1].content
        return reply, trace


print("=" * 72)
print("你的版本跑两轮的结果")
print("=" * 72)
reply, trace = 你的版本()
print(f"  reply = {reply!r}")
print(f"  trace 有 {len(trace)} 条：")
for t in trace:
    print(f"      {t}")
print()
print("  应该是什么：")
print("      trace 有 2 条（search_books 和 get_post_detail 各一条）")
print("      search_books 的 result_count 应该是 2")
print("      get_post_detail 的 result_count 应该是 1（它返回 list[dict]，只装一条）")


def 正确版本():
    trace = []
    for i, m in enumerate(out_msgs):
        if not isinstance(m, AIMessage):
            continue
        if not m.tool_calls:
            continue

        for call in m.tool_calls:
            count = 0
            for later in out_msgs[i + 1:]:
                if not isinstance(later, ToolMessage):     # ← ToolMessage
                    continue
                if later.tool_call_id != call["id"]:       # ← tool_call_id
                    continue
                try:
                    count = len(json.loads(later.content))
                except Exception:
                    count = 0
                break
            trace.append({
                "name": call["name"],
                "arguments": call["args"],
                "result_count": count,
            })

    reply = out_msgs[-1].content                            # ← 移到循环外
    return reply, trace


print()
print("=" * 72)
print("改过的版本")
print("=" * 72)
reply, trace = 正确版本()
print(f"  reply = {reply!r}")
print(f"  trace 有 {len(trace)} 条：")
for t in trace:
    print(f"      {t}")
