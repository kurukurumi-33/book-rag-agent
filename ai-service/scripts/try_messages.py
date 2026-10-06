"""看 chat_once 返回的第三个值（整段会话）到底长什么样 —— 写 _messages_to_rows 之前看一眼。

需要：DeepSeek key + Chroma 索引都在。跑一次约 2 次模型调用。
用法：python scripts/try_messages.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.loop import chat_once

# history 传 [] 模拟一个新会话。第二句再带上第一句的结果，看「累加」的效果。
print("=" * 70)
print("第一轮：history 传空")
print("=" * 70)
reply1, trace1, msgs1 = chat_once("有高数吗", [])

print(f"\n返回了三个值：")
print(f"  1. reply  = {reply1[:60]!r}...")
print(f"  2. trace  = {len(trace1)} 条工具轨迹")
print(f"  3. msgs   = {type(msgs1).__name__}，共 {len(msgs1)} 个元素   <-- 就是你要看的")


def dump(tag: str, msgs: list):
    print(f"\n{tag}")
    for i, m in enumerate(msgs):
        body = m.content if isinstance(m.content, str) else str(m.content)
        body = body.replace("\n", " ")
        print(f"    [{i}] {type(m).__name__:<14} .type={m.type!r:<8} "
              f"正文 {len(body)} 字：{body[:50]}")
        print(f"        m.content 是 {type(m.content).__name__}")
        print(f"        这个对象上的字段：{list(m.model_fields)}")


dump("逐条摊开：", msgs1)

print("\n" + "=" * 70)
print("第二轮：把第一轮的结果当 history 传回去")
print("=" * 70)
reply2, trace2, msgs2 = chat_once("第一条多少钱", msgs1)

print(f"\n第三個值是 {len(msgs2)} 个元素 —— 注意它装的是**从头到尾全部**，")
print(f"不是这一轮新加的两条。新加的那两条在尾巴上：msgs2[{len(msgs1)}:]")
dump("整段会话：", msgs2)

new = msgs2[len(msgs1):]
print(f"\n切出来的尾巴（len(msgs1)={len(msgs1)}）有 {len(new)} 条，正好是这一轮新的：")
for m in new:
    print(f"    {type(m).__name__:<14} .type={m.type!r}")

print("\n_messages_to_rows 要产出的东西，就是把这些变成：")
print([{"role": m.type, "content": m.content} for m in new])
