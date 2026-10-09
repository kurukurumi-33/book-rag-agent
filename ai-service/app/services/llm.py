"""LLM 客户端。

DeepSeek 提供 OpenAI 兼容接口，所以直接复用 langchain-openai，
只需把 base_url 指过去即可。
"""

from langchain_openai import ChatOpenAI

from app.config import settings
from app.services.usage import TRACKER


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回配置好的 LLM 客户端。

    默认 temperature=0，因为信息抽取要求稳定、可复现的输出。

    ## 两件容易被忽略的事

    **1. `timeout` 和 `max_retries` 是补上的，不是一开始就有的。**
    之前全项目只有这一个出网调用没有超时保护 —— 对端卡住时请求会一直挂着，
    占着 FastAPI 的 worker 直到客户端自己放弃。这两个参数是 M3.2 加的。

    **2. `callbacks=[TRACKER]` 是「不改调用处就能拿到 token 用量」的唯一入口。**
    抽取链路用的是 `with_structured_output(...).invoke()`，它把响应解析成
    pydantic 对象返回，usage 在解析那一步就没了。callback 挂在**客户端**上，
    所有走这个客户端的调用（含 batch / 并发）都会自动进账。
    详见 services/usage.py 开头的说明。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        callbacks=[TRACKER],
    )
