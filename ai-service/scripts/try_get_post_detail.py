"""第 4b 步的现场探针：MySQL 里的帖子，经 Spring Boot，到手是什么形状。

跑之前先起两个服务（本脚本只需要 Spring Boot + MySQL，不需要 LLM）：
    BookServiceApplication  →  8080

用法：python scripts/try_get_post_detail.py

================================ 它要回答的两个问题 ================================

问题 1：hasNotes 真的有三态吗？
    向量库那边 `bool(p.get("hasNotes"))` 把 None 压成了 False，
    「卖家明说没笔记」和「卖家压根没提」在检索结果里长得一模一样。
    这个脚本直接数库里的 TRUE / FALSE / NULL 各有多少条 ——
    数量为 0 的话，三态就是我编的，get_post_detail 的这个卖点不成立。

问题 2：get_post(id) 吐回来的 JSON 长什么样？
    不猜、不照抄实体类，直接打出来看。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.book_service import BookServiceError, fetch_posts, get_post


def main() -> int:
    # ── 问题 1：hasNotes 的三态分布 ─────────────────────────────────────
    print("=" * 74)
    print("问题 1：hasNotes 在库里的真实分布")
    print("=" * 74)
    try:
        posts = fetch_posts(extract_status="DONE", limit=500)
    except BookServiceError as e:
        print(f"  [X] {e}")
        print()
        print("  -> Spring Boot 没起。先把 BookServiceApplication 跑起来再来。")
        return 1

    if not posts:
        print("  [X] 一条 DONE 的帖子都没有。")
        print("     先跑 M1 的抽取：POST http://127.0.0.1:8080/api/posts/extract-pending")
        return 1

    buckets: dict[str, list] = {"True（有笔记）": [], "False（明说没笔记）": [], "None（没提）": []}
    for p in posts:
        v = p.get("hasNotes")
        if v is True:
            buckets["True（有笔记）"].append(p)
        elif v is False:
            buckets["False（明说没笔记）"].append(p)
        else:
            buckets["None（没提）"].append(p)

    print(f"  库里有 {len(posts)} 条 DONE 的帖子：")
    for label, group in buckets.items():
        sample = f"，例如 id={group[0]['id']}" if group else ""
        print(f"    {label:<20} {len(group):>4} 条{sample}")

    if len(buckets["None（没提）"]) == 0:
        print()
        print("  [!] 没有 hasNotes = null 的帖子 —— 三态退化成两态。")
        print("      要么是抽取时模型给每条都填了 true/false，要么是数据太单一。")
        print("      get_post_detail 仍然有价值（conditionDesc / rawText / status），")
        print("      但「修 has_notes 语义丢失」这个卖点要说清楚是设计上的，不是实测的。")
    else:
        print()
        print(f"  [OK] 三态成立。{len(buckets['None（没提）'])} 条帖子卖家没提笔记 ——")
        print("       这些在 search_books 的结果里全被显示成 has_notes=False，")
        print("       只有 get_post_detail 回源头才看得见区别。")

    # ── 问题 2：get_post(id) 的原始形状 ────────────────────────────────
    # 优先挑一条 hasNotes 为 null 的 —— 那是向量库表达不了的那种
    target = (buckets["None（没提）"] or posts[0])[0]["id"]
    print()
    print("=" * 74)
    print(f"问题 2：get_post({target}) 吐回来的原始 JSON")
    print("=" * 74)
    try:
        raw = get_post(target)
    except BookServiceError as e:
        print(f"  [X] {e}")
        return 1

    if raw is None:
        print(f"  get_post({target}) -> None（主服务返回 JSON null，帖子不存在）")
    else:
        for k, v in raw.items():
            print(f"    {k:<16} {v!r}")

    print()
    print("  注意上面每个值的类型 —— 下面写工具时，你要决定哪些字段原样透传、")
    print("  哪些要像 search_books 处理 price=-1 那样翻译成人话。")

    # ── 问题 3（可选）：你写完工具后自动跑 ─────────────────────────────
    print()
    print("=" * 74)
    print("问题 3：你写的 get_post_detail 工具")
    print("=" * 74)
    try:
        from app.agent.tools import get_post_detail
    except ImportError:
        print("  [ ] tools.py 里还没有 get_post_detail —— 正常，先去写。")
        print("      写完再跑这个脚本，下面会自动接上。")
        return 0

    for pid in [target, 99999999]:
        print(f"  get_post_detail.invoke({{'post_id': {pid}}})")
        try:
            r = get_post_detail.invoke({"post_id": pid})
        except Exception as e:
            print(f"    [X] 崩了：{type(e).__name__}: {e}")
            continue
        print(f"    -> {type(r).__name__}，{len(r)} 条")
        for item in r:
            print(f"       {item}")

    print()
    print("  验收：")
    print("    - 存在的那条 -> 1 条，且 has_notes 能区分 True / False / None")
    print("    - 不存在的那条 -> 空列表（不能返回 None —— 见下）")
    print("    - 主服务没起时 -> 抛 BookServiceError，由 loop.py 的 try/except 兜住")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
