"""以图搜书的手工验收脚本 —— 拿一张真书图，跑完整链路，看结果。

## 为什么用脚本而不是 curl

`curl` 传 base64 图要手写 `data:image/jpeg;base64,...` 前缀、还要把图编码进去，
在 Windows 的 GBK 控制台上中文还会乱码。这个脚本三件事一起做了：
**读图 → 编码 → 发请求 → 把「识别出什么 + 搜到什么」并排打出来**。

## 怎么用

    cd ai-service

    # 本机图片
    ./.venv/Scripts/python.exe scripts/check_vision.py D:/pics/gaoshu.jpg

    # 网络图片（直接透传，不本地编码）
    ./.venv/Scripts/python.exe scripts/check_vision.py https://example.com/a.jpg

    # 只看识别结果，不打印命中
    # ⚠️ 只是**不打印**：服务端该搜还是会搜（一次请求就是一次完整链路）。
    #    在书脊图上这意味着一整批检索照样会跑，别以为这样就省了。
    ./.venv/Scripts/python.exe scripts/check_vision.py D:/pics/gaoshu.jpg --no-search

## 它故意把两类故障分开打

    recognized_books 是空的          →  **视觉**没识别出书（图太糊 / 没书 / prompt 要调）
    某本书 count=0                   →  **识别对了，但库里没这本书**（检索或语料的问题）

「识别错了」和「搜不到」在日志里长得一模一样，这个脚本靠每本书的 `count`
把它们分开 —— 这正是 `BookSearchResult` 里要单列 `count` 的原因。

## 一排书脊（这个脚本的主场）

二手书群里流传的不是干净的封面照，是**一个书架上十几本书的书脊**。
所以这条链路抽的是一批书名，脚本也是按本分行打的。看输出时注意两件事：

    「认出 N 本，只检索了前 M 本」   →  接口层故意的上限（省算力），不是 bug
    左边书名那一列里可能混着编的     →  书脊字小、竖排、反光，模型会漏也会编

两个上限的数字别在这里写死：它们是要被调优的（改过一轮了），
脚本按接口实回的值打，所以永远跟得上。

编出来的那个名字照样能检索出几条看着挺像的结果。**肉眼分辨的唯一办法
就是对着原图逐本核**，所以脚本把识别结果和命中并排打，而不是只打命中。

## 退出码

    0  一切正常
    1  服务没起 / 请求失败
    2  视觉模型没配 key（先去看 .env）
    3  一本书都没识别出来（不是脚本出错，是这张图本身没跑通）
"""

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

import httpx

# Windows 控制台默认 GBK，打中文会炸。这行必须在任何 print 之前执行。
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

_DEFAULT_URL = "http://127.0.0.1:8000/search/by-image"
_MIME_FALLBACK = "image/jpeg"


def to_data_url(target: str) -> str:
    """把「本地路径」或「http 链接」统一成接口能吃的形式。"""
    if target.startswith(("http://", "https://")):
        return target  # 直接透传，省一次本地编码

    path = Path(target)
    if not path.exists():
        sys.exit(f"❌ 找不到这个文件：{path}")

    mime = mimetypes.guess_type(path.name)[0] or _MIME_FALLBACK
    if not mime.startswith("image/"):
        print(f"⚠️ 这个扩展名不像图片（猜成 {mime}），还是按图片发了；模型多半会说「无」")

    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"📷 {path.name}  {len(raw) / 1024:.0f} KB  → base64 {len(encoded) / 1024:.0f} KB")
    return f"data:{mime};base64,{encoded}"


def main() -> int:
    parser = argparse.ArgumentParser(description="以图搜书手工验收")
    parser.add_argument("image", help="本地图片路径，或 http(s) 图片链接")
    parser.add_argument("--url", default=_DEFAULT_URL, help=f"接口地址（默认 {_DEFAULT_URL}）")
    parser.add_argument("--top-k", type=int, default=5, help="返回几条（默认 5）")
    parser.add_argument(
        "--no-search",
        action="store_true",
        help=(
            "只打印识别出的书名，不打印命中。"
            "⚠️ 仅影响打印 —— 服务端该搜还是会搜一整批，省不了算力"
        ),
    )
    args = parser.parse_args()

    # ── 先探活，让报错更有指向性 ──────────────────────────────────────
    # 不探活的话，服务没起会表现成 ConnectError 一长串栈 ——
    # 看起来像脚本写错了，其实只是服务没开。
    base = args.url.rsplit("/search/by-image", 1)[0]
    try:
        health = httpx.get(f"{base}/health", timeout=10).json()
    except Exception as e:
        print(f"❌ 连不上服务（{base}）：{e}")
        print("   → 先起 AI 服务：")
        print("     cd ai-service && ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000")
        return 1

    vis = health.get("vision", {})
    print(f"🔍 /health: vision.configured={vis.get('configured')}  model={vis.get('model')}")
    if not vis.get("configured"):
        print("❌ 视觉模型没配 key —— 接口现在只会回 503。")
        print("   → 在 ai-service/.env 里填 VISION_API_KEY（默认用智谱 GLM-4V-Flash，有免费额度）")
        print("   → 步骤见 docs/private/环境与依赖操作记录.md 第 6 节")
        return 2
    print()

    payload = {"image": to_data_url(args.image), "top_k": args.top_k}

    try:
        resp = httpx.post(args.url, json=payload, timeout=120)
    except Exception as e:
        print(f"❌ 请求失败：{e}")
        return 1

    if resp.status_code != 200:
        print(f"❌ HTTP {resp.status_code}")
        print(f"   {resp.json().get('detail', resp.text)[:300]}")
        return 1

    data = resp.json()
    books = data.get("recognized_books") or []
    results = data.get("results") or []

    print("─" * 60)
    if not books:
        print("⚠️ 一本书都没识别出来（recognized_books = []）")
        print("   → 图里确实没有书？还是太糊 / 竖排 / 反光太厉害？")
        print("   → 这不是检索的问题：检索压根没被调用。")
        return 3

    print(f"📖 识别出 {len(books)} 本：{'、'.join(books)}")
    if len(results) < len(books):
        # 这不是失败，是接口层的算力闸门。打出来是因为沉默地少搜几本
        # 比慢一点糟得多 —— 你会以为「图里就那么几本」。
        print(
            f"   ⓘ 只检索了前 {len(results)} 本（接口上限），"
            f"剩下 {len(books) - len(results)} 本没搜"
        )
    if args.no_search:
        return 0

    total = 0
    for r in results:
        total += r["count"]
        # 一行书名的标题行，下面挂它的命中
        print(f"\n🎯 《{r['book']}》 命中 {r['count']} 条")
        if r["count"] == 0:
            print("   （空）→ **识别是对的，但库里没有这本书** —— 这和「识别错了」是两回事：")
            print("          前者看语料覆盖，后者看视觉模型。")
            continue
        for i, h in enumerate(r["hits"], 1):
            rr = h.get("rerank_score")
            rr_txt = f"  精排 {rr:.4f}" if rr is not None else ""
            price = "未标价" if h["price"] is None else f"{h['price']:g} 元"
            notes = "有笔记" if h["has_notes"] else "未提及笔记"
            print(
                f"   {i}. #{h['post_id']:<6} {h['book_name'] or '（无书名）':<16}"
                f" cos {h['score']:.4f}{rr_txt}   {price}  {notes}"
            )

    print("─" * 60)
    print(f"合计 {total} 条命中。")
    print("👉 现在对着原图逐本核一遍左边那些书名：**有没有图里根本没有的书**。")
    print("   漏一本只是少看几本；编一本会检索出看着挺像的结果，你分不出来 ——")
    print("   那才是这个功能真正危险的失败模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
