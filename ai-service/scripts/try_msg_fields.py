"""AIMessage 和 ToolMessage 到底差在哪。

不调模型，手工两条按真实形状构造，把字段摊开看。
"""

import json

from langchain_core.messages import AIMessage, ToolMessage

工单 = AIMessage(
    content="",                       # ← 注意：空的
    tool_calls=[{
        "name": "search_books",
        "args": {"query": "高等数学"},
        "id": "call_00_xxx",
        "type": "tool_call",
    }],
)

回执 = ToolMessage(
    content='[{"post_id": 209, "title": "高等数学 同济第七版"}]',   # ← 字符串
    tool_call_id="call_00_xxx",
    name="search_books",
    status="success",
)


def 摊开(m, 标题):
    print("=" * 72)
    print(f"{标题}  （{type(m).__name__}）")
    print("=" * 72)
    for k, v in m.model_dump().items():
        if k in ("additional_kwargs", "response_metadata", "usage_metadata"):
            pass                                   # 这两个跟本题无关，跳过
        elif v in (None, "", [], {}):
            print(f"    {k:<18} = {v!r}      ← 空")
        else:
            s = repr(v)
            print(f"    {k:<18} = {s[:72]}{'...' if len(s) > 72 else ''}")
    print()


摊开(工单, "【下工单的那条】模型说：我要调工具")
摊开(回执, "【回执的那条】工具说：给你结果")


print("=" * 72)
print("三处关键区别")
print("=" * 72)

print("1) content 是什么")
print(f"     工单.content = {工单.content!r}   ← 空字符串！模型这轮没说人话")
print(f"     回执.content = {回执.content[:40]!r}...")
print("     → 两个都叫 content，但一个是『模型要说的』，一个是『工具的返回值』")
print()

print("2) 工具信息放在哪")
print(f"     工单.tool_calls  = {工单.tool_calls}   ← 名字+参数+id，全在这")
print(f"     回执.tool_call_id = {回执.tool_call_id!r}      ← 只有 id 这一根线")
print(f"     回执.name         = {回执.name!r}")
print("     → 工单是『订单』，回执是『凭条』。靠 id 认领")
print()

print("3) 反过来的字段是空的")
print(f"     工单.tool_call_id = {getattr(工单, 'tool_call_id', None)!r}")
print(f"     回执.tool_calls   = {getattr(回执, 'tool_calls', None)!r}")
print()

print("4) 内容类型")
print(f"     回执.content 是 {type(回执.content).__name__} —— 所以能 json.loads")
print(f"     json.loads 之后：{json.loads(回执.content)}")
print(f"     len() 出来就是 trace 里的 result_count：{len(json.loads(回执.content))}")
