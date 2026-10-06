"""M5 边界与压测：往 POST /api/chat 打畸形输入和并发。

分三块，成本差很多，所以用 flag 分开跑：

    --free    参数校验。全在 Java 的 @Valid 层被拦下，**一次模型调用都不发**
    --race    并发打**同一个会话**（2 次模型调用）—— 找历史写入的竞态
    --load    并发打**不同会话**（5 次模型调用）—— 压 embedding / Chroma 的线程安全
    --long    超长输入（1 次模型调用，约 20k token）

    --all     全跑

为什么并发要分「同会话」和「不同会话」两种：
    不同会话压的是**服务的并发能力**（线程安全、连接池）；
    同会话压的是**数据的并发正确性**（两个请求同时读历史、同时写回）。
    前者坏了大不了 500，后者坏了是**静默写脏数据** —— 更难发现，更该测。

跑之前确认两个服务都起着：
    MySQL80（手动启动）+ Spring Boot :8080 + uvicorn :8000
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = "http://localhost:8080"
CHAT = f"{BASE}/api/chat"
# AI 服务的读超时是 2 分钟，脚本这里留够余量；并发时排队会更久
TIMEOUT = 180

FAILED = []


def 标题(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def 判定(name, ok, detail=""):
    mark = "[OK]  " if ok else "[FAIL]"
    print(f"  {mark} {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")
    if not ok:
        FAILED.append(name)


def post(body, session_id=None):
    """发一次真实对话。返回 (状态码, 响应体文本)。"""
    if session_id is not None and isinstance(body, dict):
        body = {**body, "session_id": session_id}
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.post(CHAT, json=body)
        return r.status_code, r.text


def 历史(session_id):
    """读回这个会话落库的历史。"""
    with httpx.Client(timeout=30) as c:
        r = c.get(f"{BASE}/api/chat/{session_id}/messages", params={"limit": 50})
        return r.json() if r.status_code == 200 else []


# ============================================================================
# 参数校验 —— 0 成本
# ============================================================================

# (名字, 请求体) —— 全部应该在 @Valid 层被拒，**不进模型**
畸形请求体 = [
    ("空对象（没有 message）", {}),
    ("message 空串", {"message": ""}),
    ("message 全空格", {"message": "   "}),
    ("message 是 null", {"message": None}),
    ("session_id 带斜杠", {"message": "hi", "session_id": "a/b"}),
    ("session_id 空串", {"message": "hi", "session_id": ""}),
    ("session_id 65 字符", {"message": "hi", "session_id": "x" * 65}),
    ("session_id 带空格", {"message": "hi", "session_id": "a b"}),
    ("session_id 带中文", {"message": "hi", "session_id": "会话一"}),
    ("session_id 带百分号", {"message": "hi", "session_id": "a%2Fb"}),
]

# 连 JSON 都不是的请求体
非法JSON = [("半截 JSON", "{"), ("空 body", ""), ("顶层是数组", "[]")]


def 说清哪里错了吗(code, text):
    """400 的响应体必须能告诉调用方是哪个字段错了。

    断言的是 {error, detail} 这个形状，不是具体文案 —— 文案会改，
    形状是接口契约。曾经的版本返回 Spring 默认的
    {timestamp, status, error:"Bad Request", path}，**一个字的理由都没有**。
    """
    try:
        import json
        d = json.loads(text)
    except Exception:
        return False, f"响应体不是 JSON：{text[:80]}"
    if "detail" not in d or not d["detail"]:
        return False, f"没有 detail 字段：{text[:120]}"
    if "timestamp" in d:
        return False, "还是 Spring 默认错误体"
    return True, d["detail"]


def 跑参数校验():
    标题("① 参数校验（0 成本，全部应在 @Valid 层被拦）")

    for name, body in 畸形请求体:
        code, text = post(body)
        ok, detail = 说清哪里错了吗(code, text)
        判定(f"{code}  {name:22} {detail}", code == 400 and ok)

    print()
    print("  连 JSON 都不是的：")
    for name, raw in 非法JSON:
        with httpx.Client(timeout=60) as c:
            r = c.post(CHAT, content=raw.encode(),
                       headers={"Content-Type": "application/json"})
        ok, detail = 说清哪里错了吗(r.status_code, r.text)
        判定(f"{r.status_code}  {name:22} {detail}", r.status_code == 400 and ok)

    print()
    print("  多个字段同时不合法，要一次列全：")
    code, text = post({"message": "", "session_id": "a/b"})
    ok, detail = 说清哪里错了吗(code, text)
    判定(f"{detail}", ok and "message" in detail and "session_id" in detail)

    print()
    print("  detail 里必须用**对外**的键名（session_id），不能是 Java 属性名（sessionId）：")
    code, text = post({"message": "hi", "session_id": "a/b"})
    _, detail = 说清哪里错了吗(code, text)
    判定(f"{detail}", "session_id" in detail and "sessionId：" not in detail)


# ============================================================================
# 并发 · 同一会话 —— 2 次调用，找竞态
# ============================================================================

def 跑同会话竞态():
    标题("② 并发打同一个会话（2 次调用）—— 找历史写入的竞态")

    print("  背景：每轮的处理是「读历史 → 跑 agent（好几秒）→ 写回两条」。")
    print("        两个请求同时进来，会**都读到空历史**，然后各写各的。")
    print()

    # 自己编一个合法 id，省掉「先发一次请求去拿 session_id」那一次模型调用。
    # 同一个 id 连着发两个请求 —— 这就是竞态的触发条件。
    sid = "race" + str(int(time.time()))

    问题 = ["有高数吗", "有线代吗"]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        结果 = list(pool.map(lambda q: post({"message": q}, sid), 问题))
    耗时 = time.time() - t0

    for q, (code, text) in zip(问题, 结果):
        print(f"  {code}  「{q}」-> {text[:110]}")

    print()
    rows = 历史(sid)
    print(f"  落库 {len(rows)} 条，实际顺序：")
    for r in rows:
        print(f"    [{r['role']:5}] {r['content'][:52]!r}")

    print()
    角色 = [r["role"] for r in rows]
    交替 = all(角色[i] != 角色[i + 1] for i in range(len(角色) - 1))
    print(f"  两个请求耗时 {耗时:.1f}s（串行的话要 ~2 倍，并发说明确实同时在跑）")
    判定("两个请求都成功", all(c == 200 for c, _ in 结果))
    判定("落库 4 条（两轮各一对）", len(rows) == 4, f"实际 {len(rows)} 条")
    判定("human/ai 严格交替", 交替, f"实际顺序 {角色}")

    if not 交替:
        print()
        print("  ⚠️ 记录：同一会话并发会出现 **human, human, ai, ai** 这种顺序。")
        print("     两个请求各自读到的都是空历史，所以**互相看不见对方**。")
        print("     这段历史再喂回模型，就是两条连续的 human —— 上下文是错乱的。")


# ============================================================================
# 并发 · 不同会话 —— 5 次调用，压线程安全
# ============================================================================

def 跑不同会话并发(n=5):
    标题(f"③ 并发打 {n} 个不同会话（{n} 次调用）—— 压 embedding / Chroma")

    print("  这条压的是：uvicorn 的线程池 + BGE 模型单例 + Chroma 客户端")
    print("  在**同时被多个请求用**的时候会不会炸。")
    print("  （FastAPI 的 def 路由跑在线程池里，不是单线程 —— 所以并发是真的）")
    print()

    base = int(time.time())
    任务 = [(f"load{base}{i}", q) for i, q in enumerate(
        ["有高数吗", "带笔记的线代", "有没有算法导论", "50 元以下的线代", "英语四级的书"])]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as pool:
        结果 = list(pool.map(lambda t: post({"message": t[1]}, t[0]), 任务))
    耗时 = time.time() - t0

    print(f"  {n} 个并发请求，总耗时 {耗时:.1f}s")
    print()
    工具次数 = []
    for (sid, q), (code, text) in zip(任务, 结果):
        try:
            import json
            d = json.loads(text)
            n_tool = len(d.get("tool_calls", []))
            工具次数.append(n_tool)
            print(f"  {code}  「{q}」-> 工具 {n_tool} 次，卡片 {len(d.get('books', []))} 张")
        except Exception:
            工具次数.append(-1)
            print(f"  {code}  「{q}」-> 响应不是 JSON：{text[:160]}")

    print()
    判定(f"{n} 个并发请求全部 200", all(c == 200 for c, _ in 结果),
         "有非 200 说明线程池/模型/向量库哪一个撑不住")

    # 逐个确认每个会话的历史是干净的 —— 并发没把会话之间串起来
    串会话 = []
    for sid, q in 任务:
        rows = 历史(sid)
        if len(rows) != 2 or rows[0]["role"] != "human" or rows[1]["role"] != "ai":
            串会话.append((sid, [r["role"] for r in rows]))
    判定("每个会话各自落库 2 条、human→ai", not 串会话, 串会话)


# ============================================================================
# 超长输入 —— 1 次调用
# ============================================================================

def 跑超长输入():
    标题("④ 超长输入（1 次调用，约 20k token）")

    # 重复一段真实帖子文本，凑出极端长度。约 20000 个中文字 ≈ 20000 token
    一句 = "高数同济七版上下册一起40有笔记划重点了可小刀。"
    长文本 = 一句 * (20000 // len(一句))
    print(f"  长度 {len(长文本)} 字符")
    print(f"  开头：{长文本[:60]}...")
    print()

    t0 = time.time()
    code, text = post({"message": 长文本})
    耗时 = time.time() - t0

    print(f"  {code}  耗时 {耗时:.1f}s")
    if code == 200:
        import json
        d = json.loads(text)
        print(f"  reply: {d['reply'][:160]}")
        print(f"  工具调用：{[(t['name'], t['result_count']) for t in d['tool_calls']]}")
    判定("超长输入没有 5xx", code < 500, text[:200])


# ============================================================================

def main():
    flags = set(sys.argv[1:])
    if not flags or "--all" in flags:
        flags = {"--free", "--race", "--load", "--long"}

    if "--free" in flags:
        跑参数校验()
    if "--race" in flags:
        跑同会话竞态()
    if "--load" in flags:
        跑不同会话并发()
    if "--long" in flags:
        跑超长输入()

    标题("汇总")
    if FAILED:
        print(f"  {len(FAILED)} 项没过：")
        for name in FAILED:
            print(f"    - {name}")
        sys.exit(1)
    print("  全部通过")


if __name__ == "__main__":
    main()
