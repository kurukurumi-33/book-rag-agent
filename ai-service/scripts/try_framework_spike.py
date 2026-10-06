"""探针：LangChain 1.x 的 create_agent 实际返回什么结构。

写框架版之前先看清这个 —— 尤其两件事：
  1. 它给出的 messages 列表里**有没有**工具往返（手写版是筛掉的）
  2. tool_calls 轨迹**要不要**自己从消息里重建（手写版是循环里顺手记的）

跑一次约 2 次模型调用。
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
print(f"create_agent 返回的类型：{type(agent).__name__}")
print()

out = agent.invoke({"messages": [{"role": "user", "content": "有高数吗"}]})

print(f"invoke 返回的类型：{type(out).__name__}，键：{list(out)}")
print()

msgs = out["messages"]
print(f"messages 有 {len(msgs)} 条：")
for i, m in enumerate(msgs):
    body = m.content if isinstance(m.content, str) else str(m.content)
    body = body.replace("\n", " ")
    tc = getattr(m, "tool_calls", None)
    extra = ""
    if tc:
        extra = f"  tool_calls={[{'name': c['name'], 'args': c['args']} for c in tc]}"
    if type(m).__name__ == "ToolMessage":
        extra = f"  tool_call_id={m.tool_call_id}  name={getattr(m, 'name', None)}"
    print(f"    [{i}] {type(m).__name__:<14} {len(body):>4} 字 {body[:45]!r}{extra}")

print()
print("关键：上面列表里有没有 ToolMessage（工具往返）？"
      + ("有 —— 跟手写版不同，手写版是筛掉的" if any(
          type(m).__name__ == "ToolMessage" for m in msgs) else "没有"))
print()
print("最后一条的 content（这就是要返回给用户的 reply）：")
print(f"    {msgs[-1].content[:200]}")
