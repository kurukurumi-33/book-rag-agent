"""检索质量测试集（集成测试）。

## 和 `scripts/eval_search.py` 的关系

**同一份用例，两种跑法** —— 不重复维护两套测试集：

| | 命令行脚本 | 这里的 pytest |
|---|---|---|
| 用途 | 调参时看细节、存快照、前后对比 | CI 里卡红线（过不了就不许合并） |
| 输出 | 人看的一屏报告 | 断言 + 失败清单 |
| 用例来源 | `scripts/eval_search.py` 的 `CASES` | **同一个 `CASES`**（import 过来的） |

改用例只改 `eval_search.py` 一处，两边同时生效。

## 为什么要 `@pytest.mark.integration`

它要起 MySQL(:3306) + Spring Boot(:8080) + AI 服务(:8000) 并建好索引才能跑，
**不是**「随时能跑的单元测试」。所以默认不跑：

    pytest -m "not integration"    # 只跑快的（默认这样跑）
    pytest -m integration          # 起齐服务后跑这个

## 为什么坚持走 HTTP 而不是直接 import search()

这是原脚本刻意的设计（见它文件头的说明）：走接口才是**真实调用路径**，
能顺带验证「默认阈值有没有真的接上」「响应结构对不对」。
直接调函数会绕过这些 —— 见面试素材第 12 条「验收用例必须能走到目标代码路径」。
"""

import importlib.util
from pathlib import Path

import pytest

# 把 scripts/eval_search.py 当模块加载（它在 scripts/ 下，不是包，只能按路径加载）
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_search.py"
_spec = importlib.util.spec_from_file_location("eval_search", _SCRIPT)
assert _spec and _spec.loader
eval_search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_search)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    """整个模块共用一个 HTTP 客户端 + 只做一次 preflight。

    scope="module"：preflight 里有探针查询（打一次 /search），
    每条用例都做一遍纯属浪费。
    """
    import httpx

    with httpx.Client() as c:
        if not eval_search.preflight(c):
            pytest.skip("AI 服务 / 索引未就绪 —— 先起服务并建索引（见 eval_search.py 的提示）")
        yield c


@pytest.mark.parametrize("case", eval_search.CASES, ids=lambda c: c.query)
def test_检索用例(client, case):
    """逐条跑。用 parametrize 是为了失败时能**精确定位是哪条 query 挂了**，
    而不是笼统地报一句「命中率不达标」。"""
    hits = eval_search.call_search(client, case.query, None)
    ok = eval_search.judge(case, hits)

    if not ok:
        expect = "（要求返回空）" if case.is_negative else " / ".join(case.expect)
        got = eval_search.top_str(hits[: eval_search.HIT_AT])
        # 把 top-K 全打出来：常见情况是「第 1 名错了但第 4 名是对的」，
        # 那说明是**阈值**把它砍了，不是检索找不到 —— 两种问题修法完全不同。
        all_hits = eval_search.top_str(hits)
        pytest.fail(
            f"用例未命中\n"
            f"  query  : {case.query!r}\n"
            f"  期望   : {expect}\n"
            f"  实际   : {got}\n"
            f"  top-10 : {all_hits}\n"
            f"  备注   : {case.note}"
        )


def test_正例命中率达到目标(client):
    """单条都过了不代表整体达标 —— 目标线是整体 top-3 命中率 ≥ 80%。
    （个别用例允许失败，因为测试集里有几条是故意挑的难例。）"""
    rows = [
        {"case": c, "ok": eval_search.judge(c, eval_search.call_search(client, c.query, None))}
        for c in eval_search.CASES
        if not c.is_negative
    ]
    rate = sum(1 for r in rows if r["ok"]) / len(rows)
    assert rate >= eval_search.TARGET_RATE, f"正例命中率 {rate:.1%} < 目标 {eval_search.TARGET_RATE:.0%}"
