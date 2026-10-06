"""试跑手写 agent 循环（M3 第 3 步）。

用真实 LLM + 真实库存，跑 §6.4 的验收。
需要 .env 里的 DeepSeek key。
「接力」那段还需要 Spring Boot 在跑 —— get_post_detail 要回调它，其余几段不用。
用法：python scripts/try_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.loop import chat_once

CASES = [
    ("有高数吗", "应调用 search_books"),
    ("你好", "**不应**调用任何工具 —— 验证它真在做决策，不是无脑调"),
    ("带笔记的便宜线代", "应调 search_books（可带 has_notes / price_max 参数），"
                        "而不是把整句丢给向量检索"),
    # ⚠ 这条只验证「参数拆对了」，压不到过滤本身：库里 5 本高数**全都没标价**，
    #   price_max=50 一个也筛不掉，模型会老实说「都没标价、无法确认」——那是对的回答。
    #   想让过滤真被压到，得换有价格的书：线代是 8/12/15 元 → 「50 元以下的线代」。
    ("50 元以下的高数", "应带 price_max 过滤，而不是把『50元以下』塞进 query"),
]


def show(msg, want, history=None):
    """跑一轮并打印现场。返回 chat_once 的三元组，跑不通返回 None。"""
    print("=" * 76)
    print(f"用户：{msg}")
    print(f"期望：{want}")
    if history is not None:
        print(f"带进去的 history：{len(history)} 条")

    try:
        out = chat_once(msg, history)
    except Exception as e:
        print(f"  ✗ 崩了：{type(e).__name__}: {e}")
        return None

    if len(out) != 3:
        print(f"  ⏸ 还没改完：chat_once 返回了 {len(out)} 个值，签名要求 3 个 "
              f"(reply, trace, history)")
        return None

    reply, trace, new_history = out
    print(f"工具轨迹（{len(trace)} 次）：")
    if not trace:
        print("    （没调工具）")
    for t in trace:
        print(f"    {t['name']}({t['arguments']}) → {t['result_count']} 条")
    print(f"回复：{reply}")
    # ★ 关键诊断：history 有没有变长
    if new_history is None:
        print("输出的 history：None   ← ⚠ 把传进来的 history 原样返回了，等于没记")
    elif history is None:
        print(f"输出的 history：{len(new_history)} 条")
    else:
        print(f"输出的 history：{len(new_history)} 条（进去 {len(history)} 条）"
              + ("" if len(new_history) > len(history) else "   ← ⚠ 没变长，等于没记忆"))
    return out


# ── 单轮：§6.4 里不依赖记忆的那几条 ──────────────────────────────
for msg, want in CASES:
    show(msg, want)

# ── 双工具：没指明是哪一本时，应该**先问**而不是逐条查（N+1 的分界）──
print("=" * 76)
print("双工具 / 未指明目标：问「有笔记的线代成色怎么样」")
print("-" * 76)
show(
    "有笔记的线代成色怎么样",
    "**只调 search_books**，然后反问是哪一本 —— 摘要里没有成色这一栏，"
    "但用户也没指明目标，逐条查详情是 N+1 次 HTTP，没必要",
)
print()
print("   ↑ 只有 search_books 一条 + 反问「哪一本」 = 正确（N+1 修复后的预期行为）")
print("   ↑ 有 5 条 get_post_detail = N+1 又回来了（查 SYSTEM_PROMPT 里那句话）")
print("   ↑ 有 get_post_detail 但报错 = Spring Boot 没起，或工具本身有问题")
print()
print("   （指明目标是哪一本时该走接力 —— 见下面「多轮」第二轮：")
print("     问了「第一条多少钱」它自己调了 get_post_detail 去取详情）")
print()

# ── 多轮：§6.4「第一条多少钱」→ 验证记忆 ─────────────────────────
print("=" * 76)
print("多轮（验证记忆）：先「有高数吗」，再「第一条多少钱」")
print("-" * 76)
out1 = show("有高数吗", "第一轮：应调 search_books")
if out1 is None:
    print("第一轮没跑通，多轮跳过")
else:
    _, _, hist1 = out1
    print()
    show("第一条多少钱", "第二轮：应能说出上一轮那批书里的第一条是谁、多少钱",
         history=hist1)
    print()
    print("   ↑ 第二轮要是反过来问「你说的是哪本书」，说明历史没带上")
