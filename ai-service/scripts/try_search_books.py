"""单测 app/agent/tools.py 的 search_books —— 不接 agent、不接 LLM。

用真实库存（Chroma 里 220 条），跑 docstring 里那三条验收。
用法：python scripts/try_search_books.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.tools import search_books

CASES = [
    ("高数", {}, "验收 1：应命中《高等数学》（库里 12 条）"),
    ("线代", {"has_notes": True, "price_max": 30.0}, "验收 2：必须非空（M2 返回空的那个场景）"),
    ("量子力学", {}, "验收 3：库里没有，应返回空"),
]

for q, kw, note in CASES:
    print("=" * 74)
    print(f"search_books({q!r}, {kw})")
    print(f"  {note}")
    try:
        r = search_books.invoke({"query": q, **kw})
    except Exception as e:
        print(f"  ✗ 崩了：{type(e).__name__}: {e}")
        continue
    print(f"  → {len(r)} 条")
    for item in r:
        print("   ", item)
