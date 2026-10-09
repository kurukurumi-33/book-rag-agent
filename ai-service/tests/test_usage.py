"""`services/usage.py` 的 token 统计。

这个文件的重点**不是**「加法对不对」，是三件事：

1. **两种 usage 形状都要认**（新式 `usage_metadata` / 老式 `llm_output.token_usage`）,
   以及 provider 换了字段名时**不能悄悄变成 0**。
2. **并发累加不能丢** —— `extract_book_info_batch` 是并发打的，
   不加锁的 `self.n += 1` 会少算，而且**不报错**，是最难发现的 bug 类型。
3. 成本算式和单价要能追溯（`pricing_cny_per_mtok` 随响应带出）。
"""

import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.services import usage as usage_service


@pytest.fixture(autouse=True)
def 干净账本():
    """全局单例在用例之间会互相污染 —— 每条用例前后都清零。"""
    usage_service.reset()
    yield
    usage_service.reset()


def _result_with_usage_metadata(input_tokens: int, output_tokens: int) -> LLMResult:
    """形状 1：每个 Generation 的 message.usage_metadata（OpenAI 兼容接口走这条）。"""
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _result_with_legacy_token_usage(prompt: int, completion: int) -> LLMResult:
    """形状 2：老的 llm_output.token_usage。"""
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output={
            "token_usage": {"prompt_tokens": prompt, "completion_tokens": completion}
        },
    )


# ---------------------------------------------------------------------------
# 两种形状
# ---------------------------------------------------------------------------


def test_认得新式的usage_metadata():
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(_result_with_usage_metadata(100, 20))

    快照 = 追踪.snapshot()
    assert 快照["llm_calls"] == 1
    assert 快照["prompt_tokens"] == 100
    assert 快照["completion_tokens"] == 20
    assert 快照["total_tokens"] == 120
    assert 快照["calls_without_usage"] == 0


def test_认得老式的token_usage():
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(_result_with_legacy_token_usage(300, 40))

    快照 = 追踪.snapshot()
    assert 快照["prompt_tokens"] == 300
    assert 快照["completion_tokens"] == 40
    assert 快照["calls_without_usage"] == 0


def test_两种形状混着来也能累加():
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(_result_with_usage_metadata(100, 20))
    追踪.on_llm_end(_result_with_legacy_token_usage(300, 40))

    快照 = 追踪.snapshot()
    assert 快照["llm_calls"] == 2
    assert 快照["total_tokens"] == 460
    assert 快照["calls_without_usage"] == 0


def test_一次响应里有多个generation也要全算():
    """批量调用可能一次回多条 —— 只取第一条会少算。"""
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="a",
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 1,
                                "total_tokens": 11,
                            },
                        )
                    ),
                    ChatGeneration(
                        message=AIMessage(
                            content="b",
                            usage_metadata={
                                "input_tokens": 20,
                                "output_tokens": 2,
                                "total_tokens": 22,
                            },
                        )
                    ),
                ]
            ]
        )
    )

    快照 = 追踪.snapshot()
    assert 快照["prompt_tokens"] == 30
    assert 快照["completion_tokens"] == 3


# ---------------------------------------------------------------------------
# ⚠️ 拿不到 usage —— 这个文件里最该存在的一组
# ---------------------------------------------------------------------------


def test_拿不到usage时不能算成0个token():
    """provider 换了字段名的话，账目是**不全**的，不是「成本为 0」。

    这一条就是 `calls_without_usage` 这个字段存在的全部理由：
    没有它，「统计整个失效」会伪装成一个漂亮的 0 元。
    """
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]))

    快照 = 追踪.snapshot()
    assert 快照["llm_calls"] == 1
    assert 快照["total_tokens"] == 0
    # ← 关键：要么这里 > 0，要么调用方根本不知道统计已经瞎了
    assert 快照["calls_without_usage"] == 1


def test_部分拿不到时两个计数都要如实反映():
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(_result_with_usage_metadata(100, 20))
    追踪.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]))

    快照 = 追踪.snapshot()
    assert 快照["llm_calls"] == 2
    assert 快照["total_tokens"] == 120
    assert 快照["calls_without_usage"] == 1


# ---------------------------------------------------------------------------
# 成本
# ---------------------------------------------------------------------------


def test_成本按配置单价算():
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(_result_with_usage_metadata(1_000_000, 1_000_000))

    快照 = 追踪.snapshot()
    # 100 万输入 + 100 万输出 = 单价之和
    进价 = 快照["pricing_cny_per_mtok"]["input"]
    出价 = 快照["pricing_cny_per_mtok"]["output"]
    assert 快照["estimated_cost_cny"] == pytest.approx(进价 + 出价, abs=1e-9)


def test_单价随响应带出来():
    """不把单价带出去的话，看数字的人没法判断这笔账还算不算数。"""
    快照 = usage_service.snapshot()
    assert "input" in 快照["pricing_cny_per_mtok"]
    assert "output" in 快照["pricing_cny_per_mtok"]


def test_空账本成本是0():
    快照 = usage_service.snapshot()
    assert 快照["llm_calls"] == 0
    assert 快照["estimated_cost_cny"] == 0.0


# ---------------------------------------------------------------------------
# 并发 —— 不加锁会「少算但不报错」
# ---------------------------------------------------------------------------


def test_并发累加不丢计数():
    """`extract_book_info_batch` 是真并发打 LLM 的，多个 on_llm_end 会同时进来。

    没有锁的话，`self._prompt_tokens += x` 这个「读-改-写」在字节码层面不是原子的，
    并发下会丢更新。表现是**计数偏小**，没有异常、没有报错 ——
    这正是它必须先有测试的原因。
    """
    追踪 = usage_service.UsageTracker()
    次数 = 200
    每条 = 100

    def 打一次():
        追踪.on_llm_end(_result_with_usage_metadata(每条, 0))

    线程们 = [threading.Thread(target=打一次) for _ in range(次数)]
    for t in 线程们:
        t.start()
    for t in 线程们:
        t.join()

    快照 = 追踪.snapshot()
    assert 快照["llm_calls"] == 次数
    assert 快照["prompt_tokens"] == 次数 * 每条


# ---------------------------------------------------------------------------
# 复位
# ---------------------------------------------------------------------------


def test_reset清零():
    追踪 = usage_service.UsageTracker()
    追踪.on_llm_end(_result_with_usage_metadata(100, 20))
    追踪.reset()

    快照 = 追踪.snapshot()
    assert 快照 == {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_cny": 0.0,
        "calls_without_usage": 0,
        "pricing_cny_per_mtok": 快照["pricing_cny_per_mtok"],
    }


def test_模块级snapshot是同一个账本():
    """接口层用的是模块级 `snapshot()`，不是自己 new 一个 —— 必须共用。"""
    usage_service.TRACKER.on_llm_end(_result_with_usage_metadata(50, 5))
    assert usage_service.snapshot()["prompt_tokens"] == 50
