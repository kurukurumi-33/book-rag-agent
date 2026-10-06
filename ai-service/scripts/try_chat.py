"""验收 `POST /chat` —— 走 HTTP，带记忆（M3 第 4 步 / 4c）。

跟 try_agent.py 的分工：
    try_agent.py  直接调 chat_once，测 agent 本身（不启服务）
    这个脚本      打 HTTP，测**接口层**：session_id 的收发、历史进没进库

需要三个东西同时在跑：
    1. MySQL
    2. Spring Boot（历史存在它那儿）
    3. AI 服务：uvicorn app.main:app --port 8000

用法：cd ai-service && python scripts/try_chat.py
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"
TIMEOUT = 120.0   # agent 可能要连调几次工具，给足

ok = True


def check(label: str, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] {label}")
    if not good:
        print(f"         期望 {want!r}，实际 {got!r}")


def ask(message: str, session_id: str | None = None) -> dict:
    body = {"message": message}
    if session_id:
        body["session_id"] = session_id
    r = httpx.post(f"{BASE}/chat", json=body, timeout=TIMEOUT)
    if r.status_code != 200:
        print(f"  [X] HTTP {r.status_code}：{r.text[:400]}")
        raise SystemExit(1)
    return r.json()


def show(tag: str, resp: dict):
    print(f"  {tag}")
    print(f"    session_id: {resp['session_id']}")
    print(f"    工具轨迹（{len(resp['tool_calls'])} 次）：")
    for t in resp["tool_calls"]:
        print(f"        {t['name']}({t['arguments']}) -> {t['result_count']} 条")
    if not resp["tool_calls"]:
        print("        （没调工具）")
    print(f"    回复：{resp['reply'][:200]}")


def main() -> int:
    # 必须声明：下面 except 里有一句 ok = False，Python 因此把 ok 当成局部变量，
    # 不加这句的话成功路径上会 UnboundLocalError（脚本自己的 bug，不是接口的）。
    global ok

    try:
        httpx.get(f"{BASE}/health", timeout=5)
    except httpx.HTTPError:
        print(f"[X] AI 服务没起。先跑：uvicorn app.main:app --port 8000")
        return 1

    # ── 1. 不传 session_id -> 服务端新建一个 ───────────────────────────
    print("1. 新会话")
    r1 = ask("有高数吗")
    show("用户：有高数吗", r1)
    check("返回了 session_id", bool(r1.get("session_id")), True)
    check("调了 search_books",
          [t["name"] for t in r1["tool_calls"]], ["search_books"])
    print()

    # ── 2. 带上 session_id 追问 -> 必须记得上一轮 ─────────────────────
    print("2. 同一会话追问（验证记忆真的落了库）")
    r2 = ask("第一条多少钱", r1["session_id"])
    show("用户：第一条多少钱", r2)
    check("session_id 原样带回", r2["session_id"], r1["session_id"])
    # 这是关键：它得说得出「第一条」是哪一条。说不出来就是历史没带上。
    mentions = any(w in r2["reply"] for w in ["号", "第一条", "这本", "那本"])
    check("回答里指代了上一轮的具体某条（不是反问『哪一本』）", mentions, True)
    print()

    # ── 3. 换个 session_id -> 必须不记得 ──────────────────────────────
    print("3. 另一个会话（验证隔离）")
    r3 = ask("第一条多少钱")
    show("用户：第一条多少钱（新会话）", r3)
    check("是新 session_id", r3["session_id"] != r1["session_id"], True)
    # 历史是空的，它无从知道「第一条」指什么，应该反问
    looks_lost = any(w in r3["reply"] for w in ["哪", "什么书", "具体", "不清楚", "没有上"])
    check("没有记忆，答不上来（反问或说明）", looks_lost, True)
    print()

    # ── 4. 记忆确实写进 MySQL 了 ──────────────────────────────────────
    print("4. 直接查库（绕过接口，验证是真的落库了而不是存在内存里）")
    try:
        from app.clients.book_service import load_history
        rows = load_history(r1["session_id"])
        print(f"    {r1['session_id']} 在库里有 {len(rows)} 条：")
        for m in rows:
            print(f"        [{m['role']}] {m['content'][:40]}")
        check("两轮 = 4 条", len(rows), 4)
        check("角色是 human / ai 交替",
              [m["role"] for m in rows], ["human", "ai", "human", "ai"])
    except Exception as e:
        print(f"  [X] 查库失败：{type(e).__name__}: {e}")
        ok = False

    print()
    print("全部通过" if ok else "有失败项，见上面 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
