"""精排（rerank）：用 cross-encoder 给混合召回出来的候选重新打分排序。

## 为什么需要它 —— 这是 §5.4 实测逼出来的结论，不是「标准流程里有这一步」

项目的检索链路本来是「向量 + BM25 融合」，靠**余弦阈值**决定什么算命中。
在 220 条语料上这个够用。语料涨到 1 万条之后，`scripts/analyze_threshold.py`
把每条查询的 top-10 分数分布拉出来，逐个检验了 6 种候选规则
（top1 / top1−top2 / top1−均值 / top1÷均值 / 标准差 / 均值）——

**每一种都重叠，没有一条能把正例和越界查询分开。**

最扎眼的一条：

    正例「同济版高数」的 top1−均值 = 0.0018
    越界「教育学」     的 top1−均值 = 0.0070   ← 越界的看起来比正例还自信

## 根因：双塔（bi-encoder）分不出「是不是同一件事」

    双塔：query ──编码器──> 向量 ╲
                                  ╱ 比夹角
    doc   ──编码器──> 向量 ╱

query 和文档**从头到尾没见过面**，各自压成一个点，然后比夹角。
夹角只反映「整体语义方向」。「教育」和「数学」在语料里经常一起出现（教数学），
方向就是近的 —— 但「教育学」这本书根本不存在。**这个信息在双塔里被彻底丢掉了。**

    交叉编码器（cross-encoder）：[query, doc] ──编码器──> 一个分数

拼在一起送进去，模型能看见字面的交叉、看见「教育学」这三个字压根没出现在文档里。
这是在**一个分得开的空间**里打分 —— 所以它恰好能吃上面那一类。

## 二段式：为什么不是「全用 cross-encoder」

cross-encoder **没法预先建索引** —— 每来一个查询，都要和库里的每篇文档两两过一遍模型。
1 万条语料 = 1 万次前向，CPU 上要几分钟。所以标准做法是两段：

    粗排（召回）  bi-encoder + BM25 → 从 1 万条里捞 20 条      快，可离线
    精排（重排）  cross-encoder      → 给这 20 条重新排序       慢，但只跑 20 次

**粗排保证「不漏」，精排保证「顺序对」。** 两者不是替代关系。

## 选型：为什么是 BAAI/bge-reranker-base

和现有的 `bge-small-zh-v1.5` 同家族、同一套词表和训练数据分布，
配套使用效果最稳；中文支持好；base 版 CPU 上 20 条候选约 1~2 秒，演示可接受。

## 一个必须说清的边界

**rerank 不能救「粗排没捞到的书」。** 它只在候选池里重排，
候选池是 20 条，第 21 名的书哪怕再相关也是 0 分机会。
所以粗排的阈值**不能为了「更准」而调得过严** —— 那条线是「召回率」，
rerank 管的是「准确率」，两者别互相抢活。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

# ⚠️ 和 embedding.py 同一个理由：huggingface_hub 在**被 import 的那一刻**
# 读 HF_ENDPOINT，之后再改 os.environ 已经晚了。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 只在类型检查时可见，运行时不 import（import torch 要 ~1.7s，见 embedding.py）。
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)

_model: "CrossEncoder | None" = None
_lock = threading.Lock()

# 加载失败过一次就不再重试，避免每个请求都卡在「重新加载 → 又失败」上。
# 一旦置位，rerank 降级成「原样返回」，由 /health 暴露出来。
_load_failed = False

# 手动下载的模型放这儿（见 docs/private/环境与依赖操作记录.md）。
# ⚠️ 为什么要有这条本地路径：hf-mirror.com **不支持 HuggingFace 新的 Xet 存储协议**，
# 走 snapshot_download 会被 302 到 cas-bridge.xethub.hf.co 然后 401 / 超时，
# 连 HF_HUB_DISABLE_XET=1 都拦不住。所以模型是一次性手工拉下来的，
# 放本地目录、不走 hub。目录存在就用它，否则才回退到按模型名去 hub 找。
_LOCAL_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-reranker-base"


def _model_source() -> str:
    """优先用本地目录，没有才回退到 hub 上的模型名。"""
    if (_LOCAL_DIR / "config.json").exists():
        return str(_LOCAL_DIR)
    return settings.rerank_model


def get_model() -> "CrossEncoder | None":
    """拿到全局唯一的 cross-encoder 实例。加载失败返回 None（不抛）。

    **为什么失败不抛异常**：rerank 是「锦上添花」的一环 —— 它挂了检索还能返回
    （只是顺序退化成 RRF 的排序）。一个可选组件把整条 /search 打挂，是拿用户体验
    为自己的洁癖买单。
    **但也不能悄悄降级** —— 那正是本项目反复批判的「静默失败」（见主服务那个
    `Math.min(limit, 500)`）。所以这里 WARNING 日志 + `status()` 暴露给 /health，
    让「rerank 现在其实没在工作」变成一件看得见的事。
    """
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder

            source = _model_source()
            # max_length=512：bge-reranker 的训练长度。我们的文档是
            # 「书名 版次 作者 出版社」这种短文本，远用不到，但留个显式值
            # 免得库默认值变了都不知道。
            _model = CrossEncoder(source, max_length=512)
            logger.info("rerank 模型已加载：%s", source)
        except Exception:  # noqa: BLE001 —— 任何加载失败都降级，不挑异常类型
            _load_failed = True
            logger.warning(
                "rerank 模型加载失败，本次进程内将降级为「不精排」；"
                "检索仍可用，只是排序退化成 RRF。详见 /health 的 rerank 字段",
                exc_info=True,
            )
    return _model


def warmup() -> None:
    """预热：服务启动时后台线程里调一次，理由同 embedding.warmup()。"""
    threading.Thread(target=get_model, name="warmup-rerank", daemon=True).start()


def status() -> dict:
    """给 /health 用：rerank 现在到底在不在工作。"""
    return {
        "enabled": settings.rerank_enabled,
        "loaded": _model is not None,
        "failed": _load_failed,
        "model": settings.rerank_model,
        "source": _model_source(),
    }


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
    min_score: float | None = None,
) -> list[dict]:
    """给候选重排，返回新的 top_k。

    :param query:      用户查询原文
    :param candidates: 粗排结果（有序即可，函数不依赖它的顺序）
    :param top_k:      返回几条
    :param min_score:  精排分的下限（**sigmoid 之后的 0~1 值**，和余弦不是一回事）。
                       None = 不按分过滤，只重排。

    返回的每一项是**原 dict 的浅拷贝**，多一个 `rerank_score` 字段。

    调用方必须知道的三件事：

    1. **`score`（余弦）字段原样保留** —— 接口对外承诺 score 是余弦相似度，
       这里塞进来路不明的数会让调用方没法用。精排分走**新字段** `rerank_score`。
    2. **只在候选池内重排**，池子外的书不会因为 rerank 冒出来（见模块开头的边界）。
    3. 模型没加载成功时**原样返回前 top_k 条**，不抛异常。
    """
    if not candidates:
        return []

    # 关掉时零成本：一个 if 都不多花，也不碰模型。
    model = get_model() if settings.rerank_enabled else None
    if model is None:
        return candidates[:top_k]

    # 只精排前 N 条 —— 精排是 CPU 上的逐对前向，20 条约 1~2 秒，
    # 再多就明显拖慢响应了。已经被 RRF 排到 20 名开外的，本来也进不了 top_k。
    pool = candidates[: settings.rerank_pool]

    # CrossEncoder.predict 收的是 [(query, doc), ...] 这种**成对**输入 ——
    # 这正是 cross-encoder 和 bi-encoder 在接口上的分水岭：
    # bi-encoder 是两次独立 encode 再比，cross-encoder 必须成对喂进去。
    pairs = [(query, c.get("text", "")) for c in pool]
    scores = model.predict(pairs)

    rescored = []
    for cand, raw in zip(pool, scores):
        item = dict(cand)  # 浅拷贝：不改调用方手里的原对象
        # bge-reranker 输出的是**未归一化的 logit**（可正可负、无上界），
        # 直接展示会让人以为是相似度。sigmoid 压到 (0,1)，才是个能看的「相关性」。
        # ⚠️ 压完之后它和余弦仍然**不可比** —— 那是两个模型的输出，别混着排序。
        item["rerank_score"] = _sigmoid(float(raw))
        rescored.append(item)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)

    if min_score is not None:
        rescored = [c for c in rescored if c["rerank_score"] >= min_score]

    # 池子外（第 N+1 名往后）的原样接在后面 —— 它们没被精排过，
    # 但也没被否定，直接丢掉反而会让「要 50 条」的调用方拿不满。
    rest = candidates[settings.rerank_pool :]
    return (rescored + rest)[:top_k]


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid：直接 exp(-x) 在 x 很负时会溢出。"""
    if x >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -x))
    e = pow(2.718281828459045, x)
    return e / (1.0 + e)
