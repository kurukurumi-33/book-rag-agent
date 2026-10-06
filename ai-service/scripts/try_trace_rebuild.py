"""从框架返回的消息列表里，把 trace 扫出来。

扫描就是 for 循环，但要认识清楚消息列表长什么样：
它不是一个平铺的列表，而是「工单」和「结果」成对出现的。

手造的这份完全照抄探针里的真实形状，跑起来不花钱（不调模型）。
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 照真实形状手造：一轮工具调用的完整往返
msgs = [
    HumanMessage("有高数吗"),                                            # [0]
    AIMessage(                                                          # [1] 工单
        content="",
        tool_calls=[{"name": "search_books",
                     "args": {"query": "高等数学"},
                     "id": "call_00_xxx"}],
    ),
    ToolMessage(                                                        # [2] 结果
        content='[{"post_id": 209}, {"post_id": 187}]',
        tool_call_id="call_00_xxx",
        name="search_books",
    ),
    AIMessage(content="有，库里找到 2 本《高等数学》..."),                  # [3] 最终回答
]

print("消息列表的样子：")
for i, m in enumerate(msgs):
    tag = ""
    if type(m).__name__ == "AIMessage" and m.tool_calls:
        tag = f"  ← 工单：{m.tool_calls[0]['name']}  id={m.tool_calls[0]['id']}"
    elif type(m).__name__ == "ToolMessage":
        tag = f"  ← 结果：tool_call_id={m.tool_call_id}"
    print(f"    [{i}] {type(m).__name__:<14} {tag}")

# ── 扫描：三层循环 ────────────────────────────────────────────────
trace = []

for i, m in enumerate(msgs):                       # 第一层：走一遍消息
    if type(m).__name__ != "AIMessage":
        continue
    if not m.tool_calls:                           # 没有工单 = 最终回答，跳过
        continue

    for call in m.tool_calls:                      # 第二层：这张 AIMessage 可能下了好几张工单
        count = 0
        for later in msgs[i + 1:]:                 # 第三层：从这张工单**往后**找它的结果
            if type(later).__name__ != "ToolMessage":
                continue
            if later.tool_call_id != call["id"]:   # 靠 id 配对，不是靠位置
                continue
            try:
                count = len(json.loads(later.content))   # 字符串 -> 列表 -> 数条数
            except Exception:
                count = 0                          # 工具报错时 content 不是 JSON
            break                                  # 找到就停，别继续往后扫
        trace.append({
            "name": call["name"],
            "arguments": call["args"],
            "result_count": count,
        })

print(f"\n扫出来的 trace（{len(trace)} 条）：")
for t in trace:
    print(f"    {t}")

print("\n最后一个 AIMessage 的 content 就是 reply：")
print(f"    {msgs[-1].content}")
