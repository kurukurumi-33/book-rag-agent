"""把 load_history() 返回的东西原样打出来 —— 写 _rows_to_messages 之前先看一眼。

为什么要先看：数据库行是**字典**，靠键名取值（row["role"]），不靠位置，
不用切、不用拆。但键名有哪些、中文有没有坏、缺字段时键还在不在，
这些问题光看文档答不上来，跑一次最省事。

需要 Spring Boot 在跑：python scripts/try_rows.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.book_service import load_history, save_messages

sid = f"demo-{uuid.uuid4().hex[:8]}"

# ── 1. 先塞两条进去，否则读回来是空的，什么也看不到 ──────────────────
save_messages(sid, [
    {"role": "human", "content": "有高数吗"},
    {"role": "ai", "content": "有 5 本，40 元那本有笔记"},
])

# ── 2. 读回来，看它到底长什么样 ────────────────────────────────────
rows = load_history(sid)

print(f"load_history() 返回的整个东西：{type(rows).__name__}，共 {len(rows)} 个元素\n")
print(f"第 0 个元素是：{type(rows[0]).__name__}\n")

print("第 0 个元素的全部键：")
for k in rows[0]:
    print(f"    {k!r}  ->  {rows[0][k]!r}")

print("\n两行并排看（注意每行的键是一样的，值不一样）：")
for i, r in enumerate(rows):
    print(f"    第 {i} 行： role={r['role']!r}   content={r['content']!r}")

# ── 3. 这正是 _rows_to_messages 要做的：查两个键，造一个对象 ────────
print("\n照着写就是这一句（先在这跑通，再抄进 main.py）：")
from langchain_core.messages import AIMessage, HumanMessage

_ROLE_TO_CLASS = {"human": HumanMessage, "ai": AIMessage}
msgs = [_ROLE_TO_CLASS[r["role"]](content=r["content"]) for r in rows]

for m in msgs:
    print(f"    {type(m).__name__}   .type={m.type!r}   .content={m.content!r}")

print("\n注意最后这行：造出来的对象，.type 跟数据库里的 role 一模一样 ——")
print("所以存回去（_messages_to_rows）直接拿 m.type 就行，不用再转换一次。")

print(f"\n清理（可选）：DELETE FROM chat_message WHERE session_id = '{sid}';")
