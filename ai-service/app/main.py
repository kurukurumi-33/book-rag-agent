"""二手书 AI 服务入口。

职责：
- LLM 信息抽取（帖子文本 -> 结构化字段）
- Embedding 语义检索
- Agent 对话 / tool 编排（tool 回调 Spring Boot 主服务）

设计原则：**本服务无状态，不直连业务数据库**。
需要业务数据时，通过 Spring Boot 的 REST API 获取。

启动：uvicorn app.main:app --reload --port 8000
接口文档（自动生成，不用手写）：http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.clients.book_service import BookServiceError
from app.config import settings
from app.schemas import (
    BatchItemResult,
    BookInfo,
    ExtractBatchRequest,
    ExtractBatchResponse,
    ExtractRequest,
    ExtractResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services import embedding, search as search_service
from app.services.extract import extract_book_info, extract_book_info_batch

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务启动 / 关闭时要做的事。

    这里只做一件事：**在后台线程里预热 embedding 模型**。

    模型是懒加载的，不预热的话第一个 `/search` 请求要背 15 秒的加载时间
    （实测：import torch ~1.7s + 加载模型 ~13s），表现成接口卡死。
    放后台线程是为了不拖慢服务启动 —— 见 app/services/embedding.py 的 warmup()。
    """
    if settings.warmup_on_startup:
        log.info("正在后台预热 embedding 模型 %s ...", settings.embedding_model)
        embedding.warmup()
    yield


# tag 用来在 Swagger UI 里给接口分组，description 显示在分组标题下面
TAGS_METADATA = [
    {"name": "系统", "description": "健康检查、配置确认"},
    {"name": "抽取", "description": "把混乱的帖子文本抽成结构化字段（LLM 信息抽取）"},
    {"name": "检索", "description": "语义检索：向量化 + 向量库（RAG 里的 R）"},
]

app = FastAPI(
    title="二手书 AI 服务",
    version="0.1.0",
    summary="把大学二手书交易里的自然语言帖子变成可检索的结构化数据",
    description="""
基于 LangChain + DeepSeek 的 AI 能力服务。

## 设计原则

**本服务无状态，不直连业务数据库。**
需要帖子数据时，通过 Spring Boot 主服务的 REST API 获取。

```
Spring Boot 主服务 (:8080)   ← MySQL / 业务逻辑 / 事务
        │  HTTP
        ▼
  本服务 (:8000)             ← 只负责「理解」：抽取 / 向量化 / 检索 / 对话
```

## 为什么拆两个服务

业务逻辑（校验、权限、状态流转）集中在主服务，
AI 服务只做「文本进、结果出」，两边可以独立部署和扩缩容。
""",
    openapi_tags=TAGS_METADATA,
    contact={"name": "项目文档", "url": "https://github.com/"},
    lifespan=lifespan,
)


@app.get(
    "/health",
    tags=["系统"],
    summary="健康检查",
    description="确认服务活着、`.env` 配置读到了。部署后用来探活。",
    response_description="服务状态 + 当前使用的模型 + 主服务地址",
)
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "book_service": settings.book_service_url,
    }


@app.post(
    "/extract",
    response_model=ExtractResponse,
    tags=["抽取"],
    summary="单条抽取",
    description="""
把一条帖子原文抽成结构化书籍信息。

**这个接口是无状态的**：文本进，结构出，不碰数据库。
批量处理由 Spring Boot 主服务编排（它负责查数据和写回）。

## 抽取规则要点

- **抽不到的字段填 `null`，绝不编造**（LLM 爱用常识补全，是主要风险）
- 价格统一成数字：`"20r"` / `"二十"` / `"20块"` → `20.0`
- 书名归一化：`"高数"` / `"高数同济七版"` → `"高等数学"`

完整规则见 `app/services/extract.py` 的 `SYSTEM_PROMPT`。
""",
    response_description="原文 + 抽取出的结构化字段（缺失字段为 null）",
    responses={
        500: {"description": "LLM 调用失败（key 失效、限流、超时等）"},
        501: {"description": "抽取功能未实现（开发期占位）"},
    },
)
def extract(req: ExtractRequest) -> ExtractResponse:
    try:
        info = extract_book_info(req.raw_text)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return ExtractResponse(raw_text=req.raw_text, book_info=info)


@app.post(
    "/extract/batch",
    response_model=ExtractBatchResponse,
    tags=["抽取"],
    summary="批量抽取",
    description="""
一次抽多条。内部是**并发发 N 个请求**，不是一次请求拿 N 条结果，
所以省的是等待时间，**token 消耗和计费不变**。

## 关键设计：单条失败不中断整批

返回的 `results` 顺序与请求的 `texts` **严格一致**（不是完成顺序）。
某条失败时，那一项的 `book_info` 为 null、`error` 填错误信息，其余照常返回。

这样调用方（Spring Boot 主服务）能知道**具体是哪条帖子挂了**，
而不是整批丢掉重跑。
""",
    response_description="与请求等长、顺序一致的结果列表",
)
def extract_batch(req: ExtractBatchRequest) -> ExtractBatchResponse:
    # 这一行返回的 raw 是 list[BookInfo | Exception] —— 成功和失败混在一起。
    # return_exceptions=True 的作用：失败了不整批抛错，而是把异常对象放进那个槽位。
    raw = extract_book_info_batch(
        req.texts,
        max_concurrency=req.max_concurrency,
        return_exceptions=True,
    )
    result=[]
    for text,item in zip(req.texts,raw):
        if isinstance(item,BookInfo):
            B_result=BatchItemResult(
                raw_text=text,
                book_info=item,
                error=None
            )
            result.append(B_result)
        else:
            B_result=BatchItemResult(
                raw_text=text,
                book_info=None,
                error=f"{type(item).__name__}: {item}"
            )
            result.append(B_result)
    return ExtractBatchResponse(results=result)


# ============================================================
# 语义检索（M2）
# ============================================================


@app.post(
    "/index/build",
    response_model=IndexBuildResponse,
    tags=["检索"],
    summary="建检索索引",
    description="""
把主服务里 `extract_status = DONE` 的帖子向量化，写进向量库。

## 数据从哪来

**调 Spring Boot 的 `GET /api/posts?extractStatus=DONE`，不直连数据库。**
这是铁律一的落点 —— AI 服务重启不丢任何状态，因为它不持有数据。

## 建索引做了什么

1. 拉 DONE 的帖子，跳过抽不出书名的（拼出来是噪音，会抢走真实命中）
2. 每条按 `compose_text()` 拼成一段检索文本
3. BGE 向量化成 512 维向量
4. 写进 Chroma，**id 用帖子的数据库主键**，所以能直接对回主服务那一行

## 幂等性

写入用 `upsert`：同一个 id 重复写是覆盖，不是插入。
所以**反复调这个接口不会写出一堆重复条目**，不需要先清库。

只有两种情况要传 `rebuild: true`：
- 换了 embedding 模型
- 改了 `compose_text()` 的拼法

这两种情况下旧向量和新向量不在同一套语义空间里，必须全部重算。

## 成本

**不花 LLM token**。向量化用的是本地模型，建 220 条的索引只花几秒 CPU。
""",
    response_description="本次建索引的统计",
    responses={
        501: {"description": "compose_text 还没实现（开发期占位）"},
        502: {"description": "主服务返回错误"},
        503: {"description": "连不上 Spring Boot 主服务"},
    },
)
def index_build(req: IndexBuildRequest) -> IndexBuildResponse:
    try:
        stats = search_service.build_index(limit=req.limit, rebuild=req.rebuild)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except BookServiceError as e:
        # 主服务连不上 —— 这是下游的问题，用 503 让调用方知道该重试而不是改代码
        raise HTTPException(status_code=503, detail=str(e))
    return IndexBuildResponse(**stats)


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["检索"],
    summary="语义检索",
    description="""
用自然语言查帖子，**不需要跟帖子里的用词一致**。

## 为什么关键词搜索做不到

用户搜「同济版高数」，帖子写的是「高等数学 第七版 同济大学出版社」——
没有一个词完全相同，`LIKE '%高数%'` 直接漏掉。

BGE 把两句话映射到向量空间里相近的位置，所以能匹配上。
这是这个项目相对普通 CRUD 的核心价值。

## 为什么必须有阈值过滤

**向量检索永远会返回 `top_k` 条**，哪怕库里根本没有这本书。
用户搜「量子力学」，库里只有高数和线代，它照样返回 10 条，只是分数很低。

不过滤的话，agent 会拿着这堆垃圾去答用户，答非所问。
所以 `min_score` 不是可选优化，是**正确性的一部分**。

## 返回空列表是正常结果

`count = 0` 不是错误 —— 它表示「库里确实没有相关的书」，
agent 应该据此回答「没找到」，而不是硬凑一个答案。
""",
    response_description="按相似度降序的命中列表（可能为空）",
    responses={
        501: {"description": "阈值过滤逻辑还没实现（开发期占位）"},
    },
)
def search(req: SearchRequest) -> SearchResponse:
    try:
        hits = search_service.search(
            query=req.query, top_k=req.top_k, min_score=req.min_score
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

    return SearchResponse(
        query=req.query,
        min_score=req.min_score,
        count=len(hits),
        hits=[
            SearchHit(
                post_id=int(h["id"]),
                score=round(h["score"], 4),
                text=h["text"],
                # 元数据里的 price 用 -1 表示「原帖没写」，转回 None
                price=(h["metadata"].get("price") if h["metadata"].get("price", -1) >= 0 else None),
                has_notes=h["metadata"].get("has_notes", False),
                # 空串转回 None：接口对外应该说 null，而不是空字符串
                book_name=h["metadata"].get("book_name") or None,
                author=h["metadata"].get("author") or None,
                publisher=h["metadata"].get("publisher") or None,
                edition=h["metadata"].get("edition") or None,
            )
            for h in hits
        ],
    )
