# -*- coding: utf-8 -*-
"""ragas 0.4.3 与本机 langchain-community 的兼容垫片（**只在 .venv-eval 里用**）。

## 为什么要这么个东西

`import ragas` 会执行 `ragas/llms/base.py` 第 12 行：

    from langchain_community.chat_models.vertexai import ChatVertexAI

这个模块在 langchain-community 0.4 里**已经被删掉了**（Vertex 集成迁去了独立的
`langchain-google-vertexai` 包，见 langchain-community 的 sunset 公告）。实测报错：

    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

ragas 0.4.3 是当前最新版（`pip index versions ragas` 查过），所以这是
**上游还没修**的问题，不是我们把版本装错了。

## 为什么不降级 langchain-community

ragas 在模块级还同时引着 `langchain_openai`（`llms/base.py:16-18`、
`embeddings/base.py:12`）。要装回带 `chat_models.vertexai` 的老
langchain-community，就得把 langchain-core 跟着降到 0.3.x，langchain-openai 也得
跟着降 —— 为的只是一个我们**从不使用**的 Vertex 分支，把整棵依赖树拽回半年前。
评测环境里不值当做这个交换。

## 这个垫片做了什么（以及边界）

在 `import ragas` **之前**往 `sys.modules` 塞一个同名空模块，让那行 import 过得去。

⚠️ 边界说清楚：`ChatVertexAI` 在 ragas 里只出现在 `llms/base.py:43` 的一个
**类型元组**里 —— 用来判断「调用方传进来的 langchain LLM 是不是我认识的那几种」。
我们走的是 `llm_factory(..., provider="openai")`，压根不碰 Vertex。

所以占位类**故意什么都不实现**：真被实例化就直接抛。与其让它悄悄假装能用、
把「provider 用错了」伪装成一个莫名其妙的 AttributeError，不如当场炸掉。

⚠️ **这个文件是给 .venv-eval 用的，不要 import 进服务代码。**
"""
import sys
import types

_MISSING = "langchain_community.chat_models.vertexai"


def install() -> bool:
    """本机缺那个模块时打上垫片。返回 True 表示真的打上了。"""
    if _MISSING in sys.modules:
        return False

    try:
        __import__(_MISSING)
        return False          # 本机有，不需要垫
    except ImportError:
        pass

    module = types.ModuleType(_MISSING)

    class ChatVertexAI:  # noqa: N801 —— 名字必须和上游一致，否则 import 对不上
        """占位类。ragas 只用它做类型判断，正常路径下不会被实例化。"""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ChatVertexAI 是 scripts/_ragas_compat.py 塞的占位类，没有实现。"
                "运行到这里说明 ragas 真去建 Vertex 客户端了 —— 那意味着用了 "
                "provider='google'，而 .venv-eval 里没装那个集成包。"
                "本项目走的是 provider='openai'（DeepSeek 兼容），不该走到这里。"
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules[_MISSING] = module
    return True
