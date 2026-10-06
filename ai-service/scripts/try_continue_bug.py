"""错误写法演示：挪进 try 之后用 `continue` 跳过编造的工具名。

「只挪进去」= 正确；「挪进去 + continue 跳过」= 更糟。
这一版把这个区别跑出来。跑一次约 2 次模型调用。

背景：模型很少真编工具名（bind_tools 把 schema 约束住了），所以用一个**手造的**
工单来模拟 —— 这正是「从正常问句走不到的分支，就手动撬到那条路上」的做法。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.loop import SYSTEM_PROMPT
from app.agent.tools import get_post_detail, search_books
from app.services.llm import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

TOOLS = {t.name: t for t in [search_books, get_post_detail]}

llm = get_llm(temperature=0.0)
bound = llm.bind_tools(list(TOOLS.values()))


def 手造的工单() -> AIMessage:
    """假装模型编了个不存在的工具名。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": "查天气", "args": {"city": "广州"}, "id": "call_1"}],
    )


def 走一遍(tag: str, 用continue: bool):
    print("=" * 72)
    print(f"{tag}")
    print("=" * 72)

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage("有高数吗")]
    resp = 手造的工单()
    messages.append(resp)          # 对应 loop.py 第 124 行 messages.append(resp)
    print(f"    模型下了 1 张工单：{resp.tool_calls[0]['name']}")

    for call in resp.tool_calls:
        if 用continue and call["name"] not in TOOLS:
            print("    → 名字不在 TOOLS 里，continue 跳过这条")
            continue
        # 正确做法：不管名字真假，都回一条结果
        tool = TOOLS.get(call["name"])
        content = (f"没有叫「{call['name']}」的工具，"
                   f"可用的只有：{'、'.join(TOOLS)}") if tool is None else "…"
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        print(f"    → 回了 ToolMessage：{content}")

    print(f"    现在 messages 里有 {len(messages)} 条："
          f"{[type(m).__name__ for m in messages]}")
    print("    拿这串消息再问模型一次……")
    try:
        out = bound.invoke(messages)
        if out.tool_calls:
            print(f"    ✅ 通。模型改口，重开一张真工单："
                  f"{out.tool_calls[0]['name']}({out.tool_calls[0]['args']})")
        else:
            print(f"    ✅ 通。模型直接回答：{out.content[:80]}")
    except Exception as e:
        print(f"    ✗ {type(e).__name__}:")
        msg = str(e)
        print(f"        {msg[:300]}")
    print()


走一遍("错误写法：挪进 try 里，但用 continue 跳过", 用continue=True)
走一遍("正确写法：照样回一条 ToolMessage 说明情况", 用continue=False)

print("=" * 72)
print("两版的区别只有一处：那条工具结果到底回没回。")
print("continue 版本看着更「安全」（不调不存在的工具了），")
print("但工单和结果必须配对是 API 的硬规则 —— 少一条，整个对话就废在 400。")
print("=" * 72)
