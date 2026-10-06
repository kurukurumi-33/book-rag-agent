"""手写版 vs 框架版，同一个问题各跑一遍，并排看。

比三件事：
  1. 回答一样吗
  2. trace 一样吗（手写版边跑边记，框架版事后扫 —— 两条路能不能碰头）
  3. 代码多少行（面试里那个"框架省了多少"的量化答案）

跑一次约 4~6 次模型调用。
"""

import ast
import inspect
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.loop import chat_once, chat_once_framework

问题 = sys.argv[1] if len(sys.argv) > 1 else "有高数吗？顺便告诉我最便宜那本的成色"


def 有效行数(fn) -> int:
    """数代码行 —— 空行、纯注释、docstring 都不算。

    ⚠️ 必须用 AST 把 docstring 挖掉。chat_once 的 docstring 有 48 行，
       光按"非空非注释"数会把它算成代码，得出"框架省了 40 行"的假结论。
    """
    src = textwrap.dedent(inspect.getsource(fn))
    body = ast.parse(src).body[0].body
    跳过 = []
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        跳过 = range(body[0].lineno, body[0].end_lineno + 1)
    return sum(
        1 for i, line in enumerate(src.splitlines(), 1)
        if line.strip()
        and not line.strip().startswith("#")
        and i not in 跳过
    )


def 概要(fn) -> str:
    """历史列表的形状，用来看两版筛得一样不一样。"""
    return " → ".join(
        "Human" if isinstance(m, HumanMessage)
        else "AI(工单)" if isinstance(m, AIMessage) and m.tool_calls
        else "AI(回答)" if isinstance(m, AIMessage)
        else type(m).__name__
        for m in fn
    )


print("=" * 74)
print(f"问题：{问题}")
print("=" * 74)

结果 = {}
for 名字, fn in [("手写版", chat_once), ("框架版", chat_once_framework)]:
    print(f"\n【{名字}】跑起来了...")
    reply, trace, history = fn(问题)
    结果[名字] = (reply, trace, history)

    print(f"  reply : {reply}")
    print(f"  trace :（{len(trace)} 条）")
    for t in trace:
        print(f"      {t}")
    print(f"  历史  : {概要(history)}")

print()
print("=" * 74)
print("并排对照")
print("=" * 74)

手写_reply, 手写_trace, 手写_hist = 结果["手写版"]
框架_reply, 框架_trace, 框架_hist = 结果["框架版"]

print(f"\n① trace 条数      手写 {len(手写_trace)}  vs  框架 {len(框架_trace)}")

print(f"\n② 调的工具序列")
print(f"     手写：{[t['name'] for t in 手写_trace]}")
print(f"     框架：{[t['name'] for t in 框架_trace]}")

print(f"\n③ result_count")
print(f"     手写：{[t['result_count'] for t in 手写_trace]}")
print(f"     框架：{[t['result_count'] for t in 框架_trace]}")

print(f"\n④ 存下来的历史形状")
print(f"     手写：{概要(手写_hist)}")
print(f"     框架：{概要(框架_hist)}")

print(f"\n⑤ 代码行数（空行 / 注释 / docstring 都不算）")
print(f"     手写 chat_once           {有效行数(chat_once):>3} 行")
print(f"     框架 chat_once_framework {有效行数(chat_once_framework):>3} 行")
print(f"     → 差 {有效行数(chat_once) - 有效行数(chat_once_framework):+d} 行"
      "（没有变少 —— 见下面的说明）")

print()
print("=" * 74)
print("回答原文")
print("=" * 74)
print(f"\n【手写版】\n{手写_reply}")
print(f"\n【框架版】\n{框架_reply}")
