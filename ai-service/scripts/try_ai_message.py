"""真跑一轮，把里面每条 AIMessage 的 content 原样打出来。

一轮带工具调用的对话会出现**两条** AIMessage：
  第一条 = 下工单的（说我要调工具）→ content 是空的
  第二条 = 最终回答的（看完结果说话）→ content 是人话
两条都打，才能看清 content 什么时候有、什么时候空。

跑一次约 2 次模型调用。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.loop import SYSTEM_PROMPT
from app.agent.tools import get_post_detail, search_books
from app.services.llm import get_llm

llm = get_llm(temperature=0.0)
bound = llm.bind_tools([search_books, get_post_detail])

问题 = "有高数吗"
msgs = [SystemMessage(SYSTEM_PROMPT), HumanMessage(问题)]

print("=" * 72)
print(f"用户问：{问题}")
print("=" * 72)

resp = bound.invoke(msgs)          # 第 1 次模型调用
msgs.append(resp)

n = 0
while resp.tool_calls:
    for call in resp.tool_calls:
        tool = {"search_books": search_books, "get_post_detail": get_post_detail}[call["name"]]
        结果 = tool.invoke(call["args"])
        print(f"  → 工具被执行：{call['name']}  参数={call['args']}  返回 {len(结果)} 条")
        msgs.append(ToolMessage(content=str(结果), tool_call_id=call["id"]))
    resp = bound.invoke(msgs)      # 第 2 次模型调用（+ 每多一轮再多一次）
    msgs.append(resp)

print()
print("=" * 72)
print("本次对话里所有 AIMessage")
print("=" * 72)

i = 0
for m in msgs:
    if not isinstance(m, AIMessage):
        continue
    i += 1
    有工单 = bool(m.tool_calls)
    print(f"\n--- AIMessage #{i} ---")
    print(f"  content    = {m.content!r}")
    print(f"  长度       = {len(m.content)} 字")
    print(f"  tool_calls = {m.tool_calls if 有工单 else '[]  （空 —— 这就是最终回答那条）'}")
    print(f"  → 判定：{'下工单，content 故意留空' if 有工单 else '最终回答，content 才是给用户看的话'}")

    if m.response_metadata:
        print(f"  response_metadata 里的 finish_reason = "
              f"{m.response_metadata.get('finish_reason')!r}")
