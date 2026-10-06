"""把 agent loop 拆开，一步一步打印 —— 只为了看清「每样东西长什么样」。

不接你的项目（用一个玩具工具 get_price），只需要 DeepSeek key。
跑一遍，把输出从头读到尾，再回去写 chat_once。

用法：python scripts/walk_agent_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.services.llm import get_llm

# ─────────────────────────────────────────────────────────────────────
# 一个玩具工具。@tool 之后，get_price 不再是函数，是一个工具对象。
# 注意：它只是一个普通 Python 函数，用 return 返回结果。
# ─────────────────────────────────────────────────────────────────────
@tool(parse_docstring=True)
def get_price(book: str) -> str:
    """查一本书的二手价格。

    Args:
        book: 书名，如「线代」。
    """
    return "30 元"


llm = get_llm(temperature=0.0)
bound = llm.bind_tools([get_price])  # 绑一次。bound 是新对象，llm 没变

# ─────────────────────────────────────────────────────────────────────
# 位置 1：消息列表。它是你自己拼的，用「消息类」构造。
#   SystemMessage  = 系统提示（模型的角色）
#   HumanMessage   = 用户说的话
#   AIMessage      = 模型说的话（下面第 2 步你会拿到一个）
#   ToolMessage    = 工具执行的结果（第 4 步你会自己造一个）
# ─────────────────────────────────────────────────────────────────────
messages = [
    SystemMessage("你是二手书助手。"),
    HumanMessage("《线代》多少钱？"),
]

print("① 发出去的消息列表")
for m in messages:
    print(f"    {type(m).__name__:16} | {m.content!r}")

# ─────────────────────────────────────────────────────────────────────
# 位置 2：bound.invoke(messages) 返回什么？
#   返回一个 AIMessage —— 不管它是「说话」还是「下工单」，都是 AIMessage。
#   区别在于：
#     说话   → .content 有字，.tool_calls 是空列表
#     下工单 → .content 是空字符串，.tool_calls 有内容
# ─────────────────────────────────────────────────────────────────────
resp = bound.invoke(messages)
print()
print("② resp 是什么：", type(resp).__name__)
print("   resp.content    =", repr(resp.content), " ← 空的！它没说话")
print("   resp.tool_calls =", resp.tool_calls, " ← 工单在这儿")
print()
print("   完整对象：")
print("   ", resp)

# ─────────────────────────────────────────────────────────────────────
# 位置 3：一张「工单」就是个 dict，三个键
#   name = 要调哪个工具
#   args = 参数（dict，键名和工具函数的参数名一致）
#   id   = 这张工单的编号 ★ 关键，第 4 步全靠它
# ─────────────────────────────────────────────────────────────────────
call = resp.tool_calls[0]
print()
print("③ 一张工单的三个键")
print("   call['name'] =", call["name"])
print("   call['args'] =", call["args"])
print("   call['id']   =", call["id"], " ← 记牢这个")

# ─────────────────────────────────────────────────────────────────────
# 位置 4：把 AI 这条也塞进列表。不塞的话，模型下一轮就"忘了自己下过工单"。
# ─────────────────────────────────────────────────────────────────────
messages.append(resp)
print()
print("④ 把 AI 的回复 append 进 messages，现在列表有", len(messages), "条")

# ─────────────────────────────────────────────────────────────────────
# 位置 5：执行工具。
#   .invoke(call["args"]) 的参数就是工单里那个 args dict。
#   返回值 = 你工具函数 return 的东西（这里是个字符串）。
# ─────────────────────────────────────────────────────────────────────
result = get_price.invoke(call["args"])
print()
print("⑤ 执行 get_price.invoke(call['args']) 得到：", repr(result))

# ─────────────────────────────────────────────────────────────────────
# 位置 6：把结果包成 ToolMessage 塞回去。
#   content      = 结果的字符串形式
#   tool_call_id = 工单上的那个 id ★
#   为什么非要 id？因为一次工单可以来好几张，模型靠 id 把
#   「哪条结果」对上「哪张工单」。
# ─────────────────────────────────────────────────────────────────────
messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
print()
print("⑥ 造一条 ToolMessage(content=..., tool_call_id=call['id']) 塞进去")
print("   现在消息列表有", len(messages), "条：")
for m in messages:
    print(f"    {type(m).__name__:16} | {str(m.content)[:40]!r}")

# ─────────────────────────────────────────────────────────────────────
# 位置 7：再发一次。这回模型看得见工具结果了，该说人话了。
#   如果它还在下工单（tool_calls 非空）→ 你就得回到位置 3 再来一圈。
#   这就是「循环」。
# ─────────────────────────────────────────────────────────────────────
resp2 = bound.invoke(messages)
print()
print("⑦ 再 invoke 一次")
print("   resp2.content    =", repr(resp2.content))
print("   resp2.tool_calls =", resp2.tool_calls, " ← 空了，它说完了，循环结束")
