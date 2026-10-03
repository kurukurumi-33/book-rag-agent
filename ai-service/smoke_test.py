"""D1 冒烟测试：验证 LLM 能调通。

运行：.venv/Scripts/python.exe smoke_test.py
"""

import sys

from app.services.llm import get_llm


def main() -> int:
    print("正在调用 LLM ...")
    try:
        llm = get_llm()
        resp = llm.invoke("用一句话说明什么是 RAG。")
        print("\n模型回答：")
        print(resp.content)
        print("\n[OK] 环境跑通")
        return 0
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
