"""LLM 客户端。

DeepSeek 提供 OpenAI 兼容接口，所以直接复用 langchain-openai，
只需把 base_url 指过去即可。
"""

from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回配置好的 LLM 客户端。

    默认 temperature=0，因为信息抽取要求稳定、可复现的输出。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
    )
