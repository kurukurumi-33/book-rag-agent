"""临时探针：只测 search_books 的「过滤 + 截断 + 返回形状」，不加载 BGE、不碰 Chroma。

假池子照抄真实「线代」那 12 条的分布（6 条满足「带笔记且 <=30」，2 条价格未知）。
用法：python scripts/_probe_search_books.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SEARCH_POOL = 50
_PRICE_UNKNOWN = -1.0

FAKE = [
    {"id": "139", "metadata": {"post_id": 139, "book_name": "线性代数(同济七版)", "has_notes": True,  "price": 25.0}},
    {"id": "133", "metadata": {"post_id": 133, "book_name": "线性代数(同济六版)", "has_notes": True,  "price": 18.0}},
    {"id": "124", "metadata": {"post_id": 124, "book_name": "线性代数", "has_notes": True,  "price": 30.0}},
    {"id": "98",  "metadata": {"post_id": 98,  "book_name": "线性代数辅导", "has_notes": True,  "price": 12.0}},
    {"id": "85",  "metadata": {"post_id": 85,  "book_name": "线性代数(第五版)", "has_notes": True,  "price": 28.0}},
    {"id": "144", "metadata": {"post_id": 144, "book_name": "线性代数习题集", "has_notes": True,  "price": 22.0}},
    {"id": "92",  "metadata": {"post_id": 92,  "book_name": "线性代数", "has_notes": False, "price": 40.0}},
    {"id": "157", "metadata": {"post_id": 157, "book_name": "线性代数讲义", "has_notes": False, "price": -1.0}},  # 价格未知
    {"id": "151", "metadata": {"post_id": 151, "book_name": "线性代数(同济)", "has_notes": True,  "price": 55.0}},
    {"id": "121", "metadata": {"post_id": 121, "book_name": "线性代数", "has_notes": False, "price": 20.0}},
    {"id": "115", "metadata": {"post_id": 115, "book_name": "线性代数考研", "has_notes": True,  "price": -1.0}},  # 价格未知
    {"id": "109", "metadata": {"post_id": 109, "book_name": "线性代数(旧版)", "has_notes": False, "price": 9.0}},
]


def search(query, top_k=20):
    """替身：不真检索，直接按 top_k 截断假池子。"""
    return FAKE[:top_k]


def search_books(query, has_notes=None, price_max=None, top_k=5):
    # ▼▼▼ 下面是你贴的版本，原样照抄，连缩进都没动 ▼▼▼
    old_list = search(query, _SEARCH_POOL)
    results = []
    for hits in old_list:
        meta = hits["metadata"]

        # 用户没给这一栏就不查
        if has_notes is not None and meta["has_notes"] != has_notes:
            continue

        if price_max is not None:
            price = meta["price"]
            if price != _PRICE_UNKNOWN and price > price_max:
                continue

        results.append({
            "post_id": meta["post_id"],
            "book_name": meta["book_name"],
            "price": meta["price"] if meta["price"] != _PRICE_UNKNOWN else "未知",
            "has_notes": meta["has_notes"],
        })
    return results[:top_k]
    # ▲▲▲ 原样照抄结束 ▲▲▲


CASES = [
    ("完全不筛 price_max=None", {}, "走不到 append 里的 price 判断"),
    ("只要便宜(<=30)", {"price_max": 30.0}, "price 有值，会不会还是崩？"),
    ("带笔记 + 便宜", {"has_notes": True, "price_max": 30.0}, "同上"),
    ("无笔记 + 便宜", {"has_notes": False, "price_max": 30.0}, "157 价格未知，看它渲染成什么"),
]

for label, kw, note in CASES:
    try:
        r = search_books("线代", **kw)
        out = f"✓ 返回 {len(r)} 条，第一条 = {r[0]}"
    except Exception as e:
        out = f"✗ {type(e).__name__}: {e}"
    print(f"{label:<24} {out}")
    print(f"{'':24} （{note}）")
