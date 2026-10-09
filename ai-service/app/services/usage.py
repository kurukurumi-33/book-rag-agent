"""token 用量与成本统计（M3.2）。

## 为什么是 callback，而不是在调用处数

抽取那条路是这样写的：

    llm.with_structured_output(BookInfo, ...).invoke([...])   -> 直接返回 BookInfo

DeepSeek 的响应里**本来是带 usage 的**，但 `with_structured_output` 在中间
把它解析成了 pydantic 对象 —— usage 就在这一步被丢掉了，调用方拿到的是一个
干净的 `BookInfo`，看不到任何 token 信息。

要想拿到 usage，只有两条路：

1. 不走 `with_structured_output`，自己解析 tool_calls → **要重写整条抽取链路**
2. `BaseCallbackHandler` 挂到 LLM 客户端上，在 `on_llm_end` 里从原始
   `LLMResult` 掏 usage → **不改任何调用处的返回结构**

选 2。它是这个项目里少数「用框架特性不是偷懒」的地方：callback 是 LangChain
留给「观测」的正式扩展点，不是 hack。

## 覆盖范围（别高估它）

- 只统计**文本 LLM**（DeepSeek）。挂点是 `get_llm()`，所有走它的调用自动进账：
  `/extract`、`/extract/batch`、`/chat` 的手写循环、LangChain 版对照实现。
- **不统计**视觉模型（GLM-4V-Flash 自带免费额度、计费口径也不同），
  也不统计本地 BGE embedding（本地 CPU，不花钱）。
- 计数是**进程内**的，多副本各算各的。所以它是个「演示 / 自查」工具，
  不是账单 —— 真心要卡成本得把计数也搬进 Redis（现成的 pattern，见 ratelimit.py）。
  这点主动说，别说成「做了成本监控系统」。

## 线不线程安全

`extract_book_info_batch` / `extract_book_info_many` 是**并发**打 LLM 的
（`asyncio.gather` / `batch` 起了线程池），多个 `on_llm_end` 会同时进来。
所以累加必须上锁 —— 不加锁的 `self.n += 1` 在 Python 里不是原子的，
并发下会丢计数。这是个很难发现的问题：**它只是少算，不报错**。
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings


def _extract_usage(response: LLMResult) -> tuple[int, int] | None:
    """从 LLMResult 里掏 (输入 token, 输出 token)。掏不到就返回 None。

    要兼容两种形状，因为不同 provider / 不同 langchain 版本给的不一样：

    1. 每个 Generation 的 `message.usage_metadata`（新式，OpenAI 兼容接口走这条）
       → `{"input_tokens": .., "output_tokens": .., "total_tokens": ..}`
    2. `response.llm_output["token_usage"]`（老式 OpenAI 形状）
       → `{"prompt_tokens": .., "completion_tokens": ..}`

    返回 None 而不是 (0, 0) 是有意的：**「没拿到 usage」和「用了 0 个 token」
    是两回事**。前者要在 stats 的 `calls_without_usage` 里显式暴露出来，
    否则一旦某个 provider 改了字段名，你就会看到「一切正常，成本 0 元」。
    """
    # ---- 形状 1：逐条 message 的 usage_metadata ----
    total_in = total_out = 0
    found = False
    for gen_list in response.generations or []:
        for gen in gen_list:
            message = getattr(gen, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if not meta:
                continue
            total_in += int(meta.get("input_tokens") or 0)
            total_out += int(meta.get("output_tokens") or 0)
            found = True
    if found:
        return total_in, total_out

    # ---- 形状 2：llm_output.token_usage（老式） ----
    llm_output = response.llm_output or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if token_usage:
        return (
            int(token_usage.get("prompt_tokens") or 0),
            int(token_usage.get("completion_tokens") or 0),
        )

    return None


class UsageTracker(BaseCallbackHandler):
    """累加 token 用量。挂在 `get_llm()` 上，全进程共用这一份。"""

    def __init__(self) -> None:
        # ⚠️ 必须显式调父类 __init__：BaseCallbackHandler 里有 raise_error 等属性，
        # 不调的话触发回调时可能报「没有这个属性」——而且只在**真发请求**时才报。
        super().__init__()
        self._lock = threading.Lock()
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        # 成功结束但**拿不到 usage** 的调用数。>0 就说明账目不全，别当成 0 成本
        self._calls_without_usage = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _extract_usage(response)
        with self._lock:
            self._calls += 1
            if usage is None:
                self._calls_without_usage += 1
                return
            prompt_tokens, completion_tokens = usage
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, Any]:
        """当前累计值 + 按配置单价算出的估算成本（元）。"""
        with self._lock:
            calls = self._calls
            prompt_tokens = self._prompt_tokens
            completion_tokens = self._completion_tokens
            calls_without_usage = self._calls_without_usage

        cost = (
            prompt_tokens / 1_000_000 * settings.llm_price_in_per_mtok
            + completion_tokens / 1_000_000 * settings.llm_price_out_per_mtok
        )
        return {
            "llm_calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_cny": round(cost, 6),
            "calls_without_usage": calls_without_usage,
            # 把用例带出去，免得调用方（或面试官）以为换过价了
            "pricing_cny_per_mtok": {
                "input": settings.llm_price_in_per_mtok,
                "output": settings.llm_price_out_per_mtok,
            },
        }

    def reset(self) -> None:
        """清零。测试用 —— 全局单例在用例之间会互相污染。"""
        with self._lock:
            self._calls = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._calls_without_usage = 0


# 全局单例。挂到每个 LLM 客户端上，所以整个进程共用一个账本。
TRACKER = UsageTracker()


def snapshot() -> dict[str, Any]:
    """给接口层用的函数式入口（main.py 不必知道 TRACKER 这个全局名）。"""
    return TRACKER.snapshot()


def reset() -> None:
    TRACKER.reset()
