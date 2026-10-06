"""不调模型，把你写的 post_ids 那两段原样抄下来跑，看会发生什么。

两个版本各有一个 bug，而且**都被 except 吞掉了** —— 所以不报错，只是静默出错。
"""

import json

from langchain_core.messages import ToolMessage

# ── 假数据：两条工具结果，形状跟真实工具返回一致 ──────────────────
搜索结果 = [{"post_id": 139, "book_name": "线性代数", "price": 8.0},
            {"post_id": 133, "book_name": "线性代数", "price": 12.0}]
详情结果 = [{"post_id": 139, "book_name": "线性代数", "edition": "第六版"}]


def 手写版():
    """照抄 loop.py 的 chat_once 里那段。"""
    trace = []
    post_ids = []                      # ← 在循环【外面】
    for results in [搜索结果, 详情结果]:   # 假装跑了两轮工具调用
        tool = object()                # 假装 TOOLS.get 拿到了工具
        if tool is None:
            content = "没有这个工具"
            count = 0
        else:
            try:
                content = str(results)
                count = len(results)
                post_ids.append(results["post_id"])   # ← 你写的
            except Exception as e:
                content = f"工具执行出错:{type(e).__name__}:{e}"
                count = 0
                post_ids = []                          # ← 你写的
        trace.append({"result_count": count, "post_ids": post_ids})
    return trace


print("=" * 72)
print("手写版")
print("=" * 72)
t = 手写版()
for i, r in enumerate(t):
    print(f"  trace[{i}] = {r}")

print()
print("  出了两件事：")
print("  ① results 是 list，`results['post_id']` 直接 TypeError")
print("     —— 而它在 try 里，所以 content / count【刚算好的值被 except 覆盖掉了】")
print("        下一轮模型看到的是「工具执行出错」，不是搜索结果")
print(f"  ② trace[0] 和 trace[1] 是两个独立的列表？ -> "
      f"{t[0]['post_ids'] is not t[1]['post_ids']}")
print("     —— 这里【看起来是对的】，但那是因为 except 每次都跑、")
print("        每次都 `post_ids = []` 重新绑到了新列表，把问题盖住了。")
print("        把 ① 修好，② 立刻露出来 ↓")


def 只修了一():
    """只把 results['post_id'] 改对，post_ids 还建在循环外面。"""
    trace = []
    post_ids = []                      # ← 还在循环外面
    for results in [搜索结果, 详情结果]:
        count = len(results)
        for row in results:
            if "post_id" in row:
                post_ids.append(row["post_id"])
        trace.append({"result_count": count, "post_ids": post_ids})
    return trace


print()
print("  --- 只修 ①、不修 ② ---")
t4 = 只修了一()
for i, r in enumerate(t4):
    print(f"  trace[{i}] = {r}")
print(f"     trace[0] 和 trace[1] 是两个独立的列表？ -> "
      f"{t4[0]['post_ids'] is not t4[1]['post_ids']}")
print("     ✗ trace[0] 的 post_ids 变成了 [139,133,139] —— 它被后面那轮【追加】了")
print("       两条 trace 指向同一个列表，全部一起变。")


# ── 框架版 ────────────────────────────────────────────────────────
out_msgs = [
    ToolMessage(content=json.dumps(搜索结果), tool_call_id="c1", name="search_books"),
    ToolMessage(content=json.dumps(详情结果), tool_call_id="c2", name="get_post_detail"),
]


def 框架版():
    """照抄 chat_once_framework 那段里 count / post_ids 的部分。"""
    resp = {"messages": out_msgs}      # 假装 agent.invoke 的返回值
    trace = []
    post_ids = []                      # ← 同样在循环外
    for i, m in enumerate(out_msgs):
        later = m
        count = 0
        try:
            count = len(json.loads(later.content))
            post_ids.append(resp["post_id"])   # ← 你写的
        except Exception:
            count = 0
            post_ids = []
        trace.append({"result_count": count, "post_ids": post_ids})
    return trace


print()
print("=" * 72)
print("框架版")
print("=" * 72)
t2 = 框架版()
for i, r in enumerate(t2):
    print(f"  trace[{i}] = {r}")

print()
print("  出了两件事：")
print("  ① `resp` 是整个返回值 dict（{'messages': [...]}），它没有 post_id 这个键")
print("     —— KeyError，同样在 try 里，把刚算好的 count【又清成 0】")
print("  ② 同一个引用问题")


# ── 正确写法 ──────────────────────────────────────────────────────
def 正确版():
    trace = []
    for results in [搜索结果, 详情结果]:
        post_ids = []                              # ← 挪进循环里，每次新建
        content = str(results)
        count = len(results)
        for row in results:                        # ← 遍历列表，不是 results["post_id"]
            if "post_id" in row:                   # ← 判键在不在
                post_ids.append(row["post_id"])
        trace.append({"result_count": count, "post_ids": post_ids})
    return trace


print()
print("=" * 72)
print("正确写法")
print("=" * 72)
t3 = 正确版()
for i, r in enumerate(t3):
    print(f"  trace[{i}] = {r}")
print()
print(f"  trace[0] 和 trace[1] 是两个独立的列表？ -> {t3[0]['post_ids'] is not t3[1]['post_ids']}")
