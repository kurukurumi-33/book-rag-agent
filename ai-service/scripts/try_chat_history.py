"""验收 Spring Boot 的会话历史接口（M3 第 4 步 / 4c 的 Java 侧）。

只测 Java，不碰 LLM、不碰 Chroma。用法：python scripts/try_chat_history.py

⚠️ 为什么用 httpx 而不是 curl：
   Windows 控制台默认 GBK，命令行里的中文在**到达 curl 之前**就被转坏了，
   Java 那边收到 `Invalid UTF-8 middle byte 0xd0`。这不是接口的 bug，
   是终端的编码问题 —— 用 Python 发请求能全程控制编码，结论才可信。
"""

import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

BASE = f"{settings.book_service_url}/api/chat"

ok = True


def check(label: str, got, want):
    global ok
    good = got == want
    ok = ok and good
    mark = "OK " if good else "FAIL"
    print(f"  [{mark}] {label}")
    if not good:
        print(f"         期望 {want!r}，实际 {got!r}")


def main() -> int:
    sid = f"test-{uuid.uuid4().hex[:8]}"
    print(f"会话 ID：{sid}（每次跑都是新的，不会污染之前的测试数据）\n")

    def post(messages):
        return httpx.post(f"{BASE}/{sid}/messages", json=messages, timeout=10)

    def get(limit=None):
        params = {"limit": limit} if limit is not None else {}
        return httpx.get(f"{BASE}/{sid}/messages", params=params, timeout=10)

    # ── 1. 空会话读回来是空列表 ────────────────────────────────────────
    print("1. 空会话")
    check("新会话读回来是 []", get().json(), [])

    # ── 2. 写一轮，两条 ────────────────────────────────────────────────
    print("\n2. 写第一轮")
    r = post([{"role": "human", "content": "有高数吗"},
              {"role": "ai", "content": "有 5 本"}])
    check("HTTP 200", r.status_code, 200)
    rows = r.json()
    check("返回 2 行", len(rows), 2)
    check("带上了自增 id", all(r_["id"] for r_ in rows), True)
    check("顺序是 human 在前", [r_["role"] for r_ in rows], ["human", "ai"])
    check("中文没坏", [r_["content"] for r_ in rows], ["有高数吗", "有 5 本"])

    # ── 3. 再写一轮 ────────────────────────────────────────────────────
    print("\n3. 写第二轮")
    post([{"role": "human", "content": "第一条多少钱"},
          {"role": "ai", "content": "面议"}])
    check("累计 4 条", len(get().json()), 4)

    # ── 4. limit 取的是「最近」N 条，不是「最老」N 条 ──────────────────
    #    这是最容易写错的地方：orderByAsc + LIMIT 会取反。
    print("\n4. limit 语义（最容易写反的地方）")
    last2 = get(limit=2).json()
    check("limit=2 拿到第 2 轮那两条", [x["content"] for x in last2],
          ["第一条多少钱", "面议"])
    first2 = get(limit=2).json()
    check("且顺序仍是正序（老的在前）", first2[0]["role"], "human")

    # ── 5. 校验 ────────────────────────────────────────────────────────
    print("\n5. 入参校验")
    check("非法 role -> 400",
          post([{"role": "robot", "content": "x"}]).status_code, 400)
    check("空列表 -> 400", post([]).status_code, 400)
    check("content 为空 -> 400",
          post([{"role": "human", "content": ""}]).status_code, 400)

    # ── 6. 原子性：一轮里有一条非法，应该一条都不落库 ──────────────────
    print("\n6. 原子性（写一半比不写更糟）")
    before = len(get().json())
    post([{"role": "human", "content": "这条合法"},
          {"role": "robot", "content": "这条非法"}])
    check("整轮回滚，库里条数没变", len(get().json()), before)

    # ── 7. session 之间互不串味 ────────────────────────────────────────
    print("\n7. 会话隔离")
    other = f"test-{uuid.uuid4().hex[:8]}"
    httpx.post(f"{BASE}/{other}/messages", timeout=10,
               json=[{"role": "human", "content": "别的会话"}])
    check("新会话只有自己那 1 条",
          [x["content"] for x in httpx.get(f"{BASE}/{other}/messages", timeout=10).json()],
          ["别的会话"])
    check("原会话不受影响", len(get().json()), before)

    # ── 8. session_id 的字符集（两道防线）──────────────────────────────
    #    背景：session_id 会拼进 URL 的路径段，而路径段装不下任意输入。
    #    "a/b c" 编码成 "a%2Fb%20c" 会被 Tomcat 在容器层直接 400，Spring 都进不去。
    #    所以改成约束字符集 —— Python 侧先拦，Java 侧再拦一次。
    print("\n8. session_id 字符集（Python 先拦，Java 再拦）")

    from app.clients.book_service import load_history

    try:
        load_history("a/b")
        check("Python 侧拒绝带斜杠的 session_id", "没抛异常", "ValueError")
    except ValueError:
        check("Python 侧拒绝带斜杠的 session_id（发请求前就拦下）", True, True)

    # 绕过 Python 侧的校验直接打 HTTP，验证 Java 那道防线也在
    raw = f"{settings.book_service_url}/api/chat"
    check("Java 侧拒绝非法字符 '!'",
          httpx.get(f"{raw}/bad!id/messages", timeout=10).status_code, 400)
    check("Java 侧拒绝超长 session_id",
          httpx.get(f"{raw}/{'x' * 70}/messages", timeout=10).status_code, 400)
    check("Java 侧接受合法的 uuid 形状",
          httpx.get(f"{raw}/{sid}/messages", timeout=10).status_code, 200)

    print()
    print("全部通过" if ok else "有失败项，见上面 FAIL")
    print(f"\n清理（可选）：DELETE FROM chat_message WHERE session_id LIKE 'test-%';")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
