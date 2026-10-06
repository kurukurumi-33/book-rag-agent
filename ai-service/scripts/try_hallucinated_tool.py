"""幻觉工具名：为什么 line 127 那个 KeyError 麻烦，以及为什么不能「跳过它」。
跑一次约 1 次模型调用（第二段）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.loop import SYSTEM_PROMPT
from app.agent.tools import get_post_detail, search_books
from app.services.llm import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 跟 loop.py 里那句一模一样（TOOLS 是在 chat_once 内部建的，导不出来）
TOOLS = {t.name: t for t in [search_books, get_post_detail]}

print("=" * 72)
print("第一段：查表在 try 外面会怎样（不用模型，白跑）")
print("=" * 72)
print(f"    TOOLS 里有的名字： {list(TOOLS)}")

call = {"name": "查天气", "args": {"city": "广州"}, "id": "call_1"}

# 现在 loop.py 的写法：查表在 try 外面
try:
    tool = TOOLS[call["name"]]      # ← line 127
    try:
        tool.invoke(call["args"])
    except Exception as e:
        print(f"    工具执行出错：{type(e).__name__}")
except KeyError as e:
    print(f"    ✗ KeyError 冒出去了： {e}")
    print("      —— 上面那层 except Exception 根本没机会接住，")
    print("         它会一路冒到 chat()，被 FastAPI 变成 500。")

print()
print("=" * 72)
print("第二段：那把查表挪进 try 就完了吗？—— 不能「跳过这条工单」")
print("=" * 72)
print("    模型下的工单和工具的结果是**成对**的。只留工单不留结果，下一轮 invoke 会拒。")

llm = get_llm(temperature=0.0)
bound = llm.bind_tools(list(TOOLS.values()))

# 手造一段「模型下了工单、但没有对应结果」的消息
fake_ai = AIMessage(
    content="",
    tool_calls=[{"name": "search_books", "args": {"query": "高数"}, "id": "call_1"}],
)
msgs = [SystemMessage(SYSTEM_PROMPT), HumanMessage("有高数吗"), fake_ai]

try:
    bound.invoke(msgs)
    print("    （居然没报错？看看返回了什么）")
except Exception as e:
    print(f"    ✗ {type(e).__name__}:")
    for line in str(e).splitlines():
        print(f"        {line}")

print()
print("    ↑ 所以 `continue` 跳过是行不通的 ——")
print("      即使工具名是编的，也必须回一条 ToolMessage，")
print("      并且 tool_call_id 要跟那条工单的 id 对上。")
