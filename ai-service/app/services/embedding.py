"""文本向量化：把一句话变成 512 个浮点数，让「意思相近」变成「距离相近」。

用本地 BGE 模型，不走 API。理由见 app/config.py 的注释。

## ⚠️ 一个必须记住的坑：千万不要给 query 加指令前缀

网上大量教程（包括 BGE 官方 README）会让你把 query 写成：

    为这个句子生成表示以用于检索相关文章：同济版高数

**那是 v1.0 时代的用法。** 在 v1.5 上实测，加了前缀分数反而变低：

    query="高等数学"  不加前缀相似度 1.0000 / 加前缀只有 0.8167

而且这个错误**不会报错**——检索还是能返回结果，只是排名悄悄变差。
「结果莫名其妙不准」比「直接崩溃」难查得多，所以这里记一笔。

（v1.5 的官方说明是：短 query 检索长文档时可以加，同长度检索不要加。
我们的场景是短 query 检索短文档，所以不加。）
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

# ⚠️ 必须在 import sentence_transformers **之前** 设好环境变量：
# huggingface_hub 是在被 import 的那一刻读 HF_ENDPOINT 的，
# 之后再改 os.environ 已经晚了。
#
# 不设的话，就算模型已经在本地缓存里，SentenceTransformer 每次启动
# 仍会去 huggingface.co 探一遍版本，连不上就重试 5 次、白等 ~25 秒。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# sentence_transformers 只在**类型检查时**可见，运行时不 import —— 原因见 get_model()
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from app.config import settings

# 模型是懒加载的单例：加载一次要十几秒、占几百 MB 内存，
# 每个请求都重新加载的话服务直接不可用。
#
# 实测耗时（Windows + CPU，模型已在本地缓存）：
#     import torch        ~1.7s     ← 光是 import 就这么多
#     加载模型 + 首次向量化 ~13s
#     ------------------------------------
#     合计                ~15s     ← 冷启动（系统文件缓存也冷）能到 34s
#
# 所以绝不能让它发生在一个真实请求里 —— 见 warmup() 的说明。
_model: SentenceTransformer | None = None

# 用锁而不是裸 if：warmup() 在后台线程里加载，主线程可能同时接了第一个请求。
# 两个线程都看到 _model is None 就会各加载一份 —— 内存翻倍，还多等十几秒。
_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """拿到全局唯一的模型实例，第一次调用时才真正加载。"""
    global _model
    if _model is None:
        with _lock:
            # 双重检查：抢到锁之后要再看一眼，
            # 因为可能已经有别的线程加载完了
            if _model is None:
                # ⚠️ 延迟到**这里**才 import，不放在模块顶层。
                #
                # 光 `import torch` 就要 ~1.7s，冷启动更久。放在顶层的话，
                # 任何 `import app.services.embedding` 的代码都要背这个代价 ——
                # 包括跑单元测试（实测全套单测从 34s 降到 2s 就是这样来的）。
                #
                # 顺序仍然安全：上面的 HF_ENDPOINT 在**模块加载时**就设好了，
                # 而这里才第一次触发 huggingface_hub 的 import。
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(settings.embedding_model)
    return _model


def warmup() -> None:
    """预热：提前把模型加载好。服务启动时调一次。

    **为什么必须预热**：模型是懒加载的，不预热的话，
    服务启动后的**第一个** `/search` 请求要背这 15 秒，
    表现成「接口卡死」——演示的时候这一下很难看。

    跑在后台线程里，所以**不会拖慢服务启动** ——
    uvicorn 打印 "Application startup complete" 之后模型才慢慢加载，
    这段空档期正好用来开浏览器 / 打开 Swagger。

    预热失败不影响服务启动（比如模型文件被删了）：
    真到用的时候 get_model() 会再试一次并抛出真实错误，
    在启动阶段就把服务搞挂反而更难排查。
    """
    thread = threading.Thread(target=get_model, name="warmup-embedding", daemon=True)
    thread.start()


def dim() -> int:
    """向量维度。bge-small-zh-v1.5 是 512。"""
    return get_model().get_sentence_embedding_dimension()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """把一批文档文本转成向量（建索引用）。

    normalize_embeddings=True 把每个向量归一化成长度 1，
    这样「点积」直接等于「余弦相似度」，后面算分不用再除模长。
    """
    vectors = get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """把用户查询转成向量。

    注意这里**故意不加**指令前缀，原因见模块开头的说明。
    """
    vector = get_model().encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vector.tolist()
