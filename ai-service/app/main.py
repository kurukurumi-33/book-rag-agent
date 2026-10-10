"""智觅（二手书智能匹配 Agent）AI 服务入口。

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
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.loop import chat_once
from app.agent.tools import get_post_detail
from app.clients import book_service as book_service_client
from app.clients.book_service import BookServiceError
from app.config import settings
from app.schemas import (
    BatchItemResult,
    BookCard,
    BookInfo,
    BookSearchResult,
    ChatRequest,
    ChatResponse,
    ExtractBatchRequest,
    ExtractBatchResponse,
    ExtractRequest,
    ExtractResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    SearchByImageRequest,
    SearchByImageResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    UsageStats,
)
from app.services import (
    embedding,
    ratelimit,
    redis_client,
    rerank as rerank_service,
    search as search_service,
    session_cache,
    usage,
    vision,
)
from app.services.extract import (
    extract_book_info,
    extract_book_info_batch,
    extract_book_info_many,
)

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务启动 / 关闭时要做的事。

    这里做一件事：**在后台线程里预热两个模型**（embedding + rerank）。

    两个都是懒加载的，不预热的话第一个 `/search` 请求要背两次加载时间
    （embedding 实测 import torch ~1.7s + 加载 ~13s；rerank 是 1.1GB 的
    cross-encoder，更慢），表现成接口卡死。
    放后台线程是为了不拖慢服务启动 —— 见 embedding.py / rerank.py 的 warmup()。

    ⚠️ rerank 预热**失败不阻塞启动**，也不影响 embedding 那一路：
    它加载失败会降级成「不精排」并由 /health 暴露出来，理由见 rerank.get_model()。
    """
    if settings.warmup_on_startup:
        log.info("正在后台预热 embedding 模型 %s ...", settings.embedding_model)
        embedding.warmup()
        if settings.rerank_enabled:
            log.info("正在后台预热 rerank 模型 %s ...", settings.rerank_model)
            rerank_service.warmup()
    yield


# tag 用来在 Swagger UI 里给接口分组，description 显示在分组标题下面
TAGS_METADATA = [
    {"name": "系统", "description": "健康检查、配置确认、用量统计"},
    {"name": "抽取", "description": "把混乱的帖子文本抽成结构化字段（LLM 信息抽取）"},
    {"name": "检索", "description": "语义检索：向量化 + 向量库（RAG 里的 R）"},
]

app = FastAPI(
    title="智觅 AI 服务",
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


# ============================================================================
# 可观测：每个请求一行耗时日志（M3.3）
#
# 为什么是 middleware 而不是在每个接口里手写计时代码：
#   - 接口有 8 个，手写就是 8 处重复，加第 9 个必漏
#   - middleware 连 404、422 这些**没进到接口函数**的请求也覆盖得到 ——
#     而「某个路径一直 404」恰恰是需要日志的
#
# ⚠️ 它只记日志，**不改响应**，也不吞异常：
#     异常继续往上抛给 FastAPI 的异常处理器，日志里记 500，用户看到的仍是标准错误体。
# ============================================================================


@app.middleware("http")
async def log_request(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # 走到这里说明异常处理器也没兜住（罕见）。记一笔再抛，
        # 不把它吞掉 —— 吞掉的话日志里多一行、用户那边却永远挂着。
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.exception(
            "%s %s -> 未捕获异常 %.0fms", request.method, request.url.path, elapsed_ms
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "%s %s -> %d %.0fms", request.method, request.url.path, response.status_code, elapsed_ms
    )
    # 顺手塞进响应头：抓包/浏览器 F12 里能直接看到，不用翻服务端日志
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.0f}"
    return response


@app.get(
    "/health",
    tags=["系统"],
    summary="健康检查",
    description="""
确认服务活着、`.env` 配置读到了。部署后用来探活。

每一栏都对应一个**可以退化**的部件：rerank / vision / redis 挂了服务都照常跑，
降级成「不精排 / 以图搜书 503 / 限流放行」。所以这里必须把它们单列出来 ——
「服务 ok」不等于「每个部件都在工作」，那正是本项目反复批判的静默失败。
""",
    response_description="服务状态 + 各部件状态 + 主服务可达性 + 累计用量",
)
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "book_service": settings.book_service_url,
        # ⚠️ 之前这里只有一个**配置值**，"主服务是不是真的活着"没人知道。
        #    现在真打一次（1.5 秒超时，不烧 token）。见 book_service.ping()。
        "book_service_reachable": book_service_client.ping(),
        # ⚠️ rerank 加载失败时**服务仍然是 ok 的**（它降级成不精排，检索照常跑）。
        #    所以这里必须把它单列出来 —— 不然「rerank 其实没在工作」会一直没人发现，
        #    而那正是本项目反复批判的静默失败。
        "rerank": rerank_service.status(),
        # 同理：没配视觉 key 时服务也 ok，只是 /search/by-image 会回 503。
        # 这一栏让「以图搜书为什么不能用」在 /health 一眼可见。
        "vision": vision.status(),
        # 第三个同类：Redis 挂了服务照跑（限流放行、缓存直读），
        # 但「限流其实没生效」必须看得见 —— 否则你以为有闸门，实际敞着。
        "redis": redis_client.status(),
    }


@app.get(
    "/usage",
    response_model=UsageStats,
    tags=["系统"],
    summary="LLM token 用量与估算成本",
    description="""
进程启动以来累计的 token 用量和**估算**成本（元）。

## 怎么统计的

抽出链路用的是 `with_structured_output(...).invoke()`，它返回的是解析好的
`BookInfo` 对象 —— **usage 在解析那一步就被丢掉了**。所以拿不到 token。

解法不是重写链路，而是在 LLM 客户端上挂一个 `BaseCallbackHandler`，
在 `on_llm_end` 里从原始 `LLMResult` 掏 usage。调用处的返回结构一行没改。
见 `app/services/usage.py`。

## 三个边界（别当账单）

1. **进程内累计，重启清零**。多副本各算各的。
2. **只算文本 LLM**。视觉模型有免费额度、计费口径不同，不计入；
   本地 BGE embedding 在 CPU 上跑，本来就不花钱。
3. 成本按 `pricing_cny_per_mtok`（配置里的单价）估算，不是从账单同步的。

## `calls_without_usage` 那一栏最该看

它 > 0 就说明**有调用成功返回但没拿到 usage** —— 账目不全。
设这一栏是因为「统计失效」最容易伪装成「成本为 0 元」：
没这一栏的话，provider 换个字段名，你会看到一个漂亮的 0。
""",
    response_description="累计调用次数、token 数、估算成本",
)
def usage_stats() -> UsageStats:
    return UsageStats(**usage.snapshot())


# ============================================================================
# 限流（M4.5）
#
# 放在所有「要烧模型调用」的接口前面：/chat、/extract、/extract/batch。
# 检索系列（/search、/index/build）不烧 key，不限。
# ============================================================================


def _client_id(request: Request) -> str:
    """限流用的「谁」。取 TCP 对端地址。

    ⚠️ **故意不看 X-Forwarded-For。** 那个头是客户端可以随便写的：

        curl -H 'X-Forwarded-For: 1.2.3.4' ...   # 换个值就换一个桶

    也就是说，在没有「可信代理」的前提下信 XFF，等于给限流开了个
    **一行命令就能绕过**的后门。只有在明确知道前面有反代、且反代保证
    会覆写这个头的时候，才可以信它 —— 本项目是直连，没这回事。

    （真要上反代，正确做法是配置 uvicorn 的 `--proxy-headers` +
     `--forwarded-allow-ips`，让框架决定信不信，而不是手工读头。）
    """
    return request.client.host if request.client else "unknown"


def _enforce_limit(request: Request, scope: str) -> None:
    """超限就抛 429。Redis 不可用时**放行**（fail-open，见 ratelimit.py 开头）。"""
    result = ratelimit.check(
        scope=scope,
        identity=_client_id(request),
        limit=ratelimit.limit_for(scope),
        window_s=60,
    )
    if result.allowed:
        return

    # 带上 Retry-After：不告诉客户端「还要等几秒」的话，它们只能盲目重试，
    # 反而更容易把服务压垮 —— 拒绝的同时给出恢复时间，是 429 的标准做法。
    raise HTTPException(
        status_code=429,
        detail=(
            f"请求太频繁：{scope} 限制 {result.limit} 次/分钟，"
            f"请 {result.retry_after} 秒后再试。"
        ),
        headers={
            "Retry-After": str(result.retry_after),
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": "0",
        },
    )


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
        429: {"description": "请求太频繁（默认 60 次/分钟/IP），带 Retry-After"},
        500: {"description": "LLM 调用失败（key 失效、限流、超时等）"},
        501: {"description": "抽取功能未实现（开发期占位）"},
    },
)
def extract(request: Request, req: ExtractRequest) -> ExtractResponse:
    _enforce_limit(request, "extract")
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
一次抽多条。**怎么省，取决于 `batch_size`**：

| batch_size | 一次请求几条 | 省什么 |
|---|---|---|
| 1 | 1 条（并发发 N 个请求） | 只省**等待时间**，token / 计费**不变** |
| >1（默认 20） | N 条 | 真省 token —— system prompt 每 N 条才发一次 |

`SYSTEM_PROMPT` 有 1500 多字，逐条抽取时每条帖子都要把它重发一遍。
1 万条帖子光重复发送提示词就要 1200 万 token；批量后同样的钱能多抽约 6 倍。

代价：批量把一个 chunk 的失败绑在一起（20 条一起挂），逐条模式则互不影响。

## 关键设计：单条失败不中断整批

返回的 `results` 顺序与请求的 `texts` **严格一致**（不是完成顺序）。
某条失败时，那一项的 `book_info` 为 null、`error` 填错误信息，其余照常返回。

这样调用方（Spring Boot 主服务）能知道**具体是哪条帖子挂了**，
而不是整批丢掉重跑。
""",
    response_description="与请求等长、顺序一致的结果列表",
)
def extract_batch(request: Request, req: ExtractBatchRequest) -> ExtractBatchResponse:
    # 和 /extract 共用 "extract" 这一个桶（不是各自一个）——
    # 它们烧的是同一份额度，分开算的话「单条打满 + 批量打满」= 两倍上限。
    _enforce_limit(request, "extract")

    # 两条实现路径，对外返回**形状完全一样**（等长、顺序一致、成功给 BookInfo、
    # 失败给异常对象），所以下面的组装代码不用关心走了哪条。
    #
    # ⚠️ 区别只在钱的粒度：
    #   batch_size == 1 → 逐条发请求。system prompt（1500 多字）每条重发一遍。
    #   batch_size >  1 → 一次请求塞 N 条，system prompt 每 N 条才发一次。
    #                     实测 1 万条帖子：13.4M token → 2M token。
    # 代价：批量把一个 chunk 的失败绑在一起（20 条一起挂），
    #       而逐条模式下每条失败互不影响。默认 20 是省钱和失败粒度之间的折中。
    if req.batch_size > 1:
        raw = extract_book_info_many(
            req.texts,
            batch_size=req.batch_size,
            max_concurrency=req.max_concurrency,
        )
    else:
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

1. **翻页**拉全 DONE 的帖子（主服务单次最多给 500 条，客户端按 500 一页翻），
   跳过抽不出书名的（拼出来是噪音，会抢走真实命中）
2. 每条按 `compose_text()` 拼成一段检索文本
3. BGE 向量化成 512 维向量
4. 写进 Chroma，**id 用帖子的数据库主键**，所以能直接对回主服务那一行
5. 用同一批文本重建 BM25 索引（两个索引永远同源）

## 幂等性

写入用 `upsert`：同一个 id 重复写是覆盖，不是插入。
所以**反复调这个接口不会写出一堆重复条目**，不需要先清库。

只有两种情况要传 `rebuild: true`：
- 换了 embedding 模型
- 改了 `compose_text()` 的拼法

这两种情况下旧向量和新向量不在同一套语义空间里，必须全部重算。

## 成本

**不花 LLM token**。向量化用的是本地模型（BGE），只烧 CPU。
1 万条语料的实测耗时见响应里的 `elapsed_ms`。
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


def _to_hits(hits: list[dict]) -> list[SearchHit]:
    """把 search() 的原始结果转成接口的 SearchHit。

    **抽成函数是因为 /search 和 /search/by-image 返回的是同一种东西** ——
    以图搜书那条链路算完书名之后，走的就是同一个 search()。
    两边各抄一份转换代码的话，将来加一个字段（比如成色）必然漏改一处。

    ⚠️ 这个函数**必须定义在 `@app.post("/search", ...)` 装饰器之前**。
       我第一版把它插在装饰器和 `def search()` 中间，结果装饰器挂到了
       `_to_hits` 上 —— `/search` 整条路由对着一个「收 list 的函数」，
       每个请求都回 422 `Input should be a valid list`。
       服务能起来、OpenAPI 看着正常、单测也不会挂（没测那个接口），
       **只有真打一次才看得出来**。回归用例见 tests/test_ratelimit_wiring.py。
    """
    return [
        SearchHit(
            post_id=int(h["id"]),
            score=round(h["score"], 4),
            # 开了精排才有这个键 —— 没开时保持 null，别编一个假分数出来
            rerank_score=(round(h["rerank_score"], 4) if "rerank_score" in h else None),
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
    ]


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

## 两段式：粗排召回 + 精排重排

    query → 向量通道 + BM25 通道 → RRF 融合（捞出 20 条候选）
          → cross-encoder 逐对精排 → 取 top_k

粗排（双塔）保证**不漏**，精排（cross-encoder）保证**顺序对**。
`rerank_score` 那一栏是精排分，**和 `score`（余弦）不可比** —— 两个模型的输出。

## 为什么必须有阈值过滤

**向量检索永远会返回 `top_k` 条**，哪怕库里根本没有这本书。
用户搜「考古」，库里一本考古书都没有，它照样返回 10 条，只是分数很低。

不过滤的话，agent 会拿着这堆垃圾去答用户，答非所问。
所以 `min_score` 不是可选优化，是**正确性的一部分**。

⚠️ 但阈值只负责粗排的**召回**，它到 0.61 就到顶了：再往上正例开始死，
而「教育学 →《数学》」这类越界查询的分数**比正例还高**，靠调它永远拦不住。
那一类交给精排和 `rerank_score`（实测数据见 `app/services/search.py` 的注释块）。

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
            query=req.query,
            top_k=req.top_k,
            min_score=req.min_score,
            rerank_enabled=req.rerank,
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

    return SearchResponse(
        query=req.query,
        min_score=req.min_score,
        count=len(hits),
        hits=_to_hits(hits),
    )


# ============================================================================
# M2.6 多模态：以图搜书
# ============================================================================


@app.post(
    "/search/by-image",
    response_model=SearchByImageResponse,
    tags=["检索"],
    summary="以图搜书（一张图里可以有多本书）",
    description="""
传一张书的照片，返回图里每本书在库里的帖子。

## 怎么做的：图 → 书名 → 复用现有检索

    图片 → 视觉模型读出**一批**书名 → 逐本 search()（BM25+向量+RRF → 精排）→ 分组命中

**不是**「图片编码成向量去比相似度」。理由（面试会问）：

1. 语料是**纯文本**的，一条封面图都没有 —— 图像检索要求每本书有封面向量，
   这份数据里**建不出来**，不是建得慢。
2. 拍的是书（封面/扉页/一排书脊），上面印着书名 —— 拍图这个动作传递的本来就是**文本信息**。
3. 认出的书名直接进 `search()`，于是 BM25、RRF、精排、阈值**全部自动生效**，
   不用为图片另起一条链路、另标一套阈值。

## 一张图里有多本书（比如一排书脊）

二手书群里流传的典型图是**一个书架上十几本书的书脊**，所以这条接口抽的是列表。
认出来多少本在 `recognized_books` 里，真的去检索了几本看 `results` 的长度：

    len(results) < len(recognized_books)   →  认出的书超过了检索上限，后面的没搜

**这个差额是故意露出来的。** 识别是一次模型调用（认 1 本和认 20 本一样贵），
检索是每本一次完整链路（真算力），所以两个上限分开设：
认得出多少放宽（20），搜多少收紧（10）。默默丢掉一半装作只认出 10 本，
比慢一点糟得多。

## 识别不出来不是错误

`recognized_books: []`（`results` 也是空）表示「图里一本书都没认出来」。
这和「识别对了但库里没这本书」（`results` 里某条 `count == 0`）**是两回事**，
排查时要分开 —— 前者怪视觉模型、该让用户重拍，后者怪检索或语料。

⚠️ 书脊图是这条链路最难的一档：字小、竖排、有反光。
模型**会漏也会认错**。认错更麻烦 —— 一个图里没有的书名照样能检索出
几条看着挺像的结果，用户分不出来。所以 prompt 里明写了「看不清的不要猜」，
但上线前仍然必须拿真图跑 `scripts/check_vision.py`。

## 没配 key 时

返回 **503**，并告诉你缺哪个环境变量。默认用智谱 GLM-4V-Flash（有免费额度）。
""",
    response_description="识别出的书名列表 + 每本的命中列表",
    responses={
        400: {"description": "图片格式不合法（空 / 太大 / 不是 data URL 或 http 链接）"},
        429: {"description": "请求太频繁（这个接口一次烧一次视觉模型调用）"},
        503: {"description": "视觉模型未配置（.env 里缺 VISION_API_KEY）"},
    },
)
def search_by_image(request: Request, req: SearchByImageRequest) -> SearchByImageResponse:
    # ⚠️ 这是**唯一一个走检索路径却要限流的接口**。
    #    /search 烧的是本地 CPU，不限；这个接口一次请求就烧一次视觉模型调用
    #    （而且比文本模型慢得多），不限的话拿到端口就能刷。
    #    加 request 参数就是为了这一行 —— 见 tests/test_routes.py 里
    #    「不烧模型的接口没有被接上限流」那条，它当初就标了这个位置。
    _enforce_limit(request, "image")

    # ── 第一步：图 → 一批书名 ───────────────────────────────────────
    try:
        books = vision.extract_books_from_image(req.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # 「没配 key」是**服务端配置问题**，不是调用方参数问题，所以是 503 不是 400
        raise HTTPException(status_code=503, detail=str(e))

    # 图里没认出书 —— 直接返回空，**不要**拿一句敷衍的词去检索。
    # 拿「无」去检索会命中一堆书名里带「无」的书，比返回空更糟：用户会以为识别错了。
    if not books:
        return SearchByImageResponse(
            recognized_books=[], min_score=req.min_score, results=[]
        )

    # ── 第二步：逐本复用文本检索 ────────────────────────────────────
    #
    # ⚠️ books 来自图片，是**不可信输入**。它们在这里只作为查询字符串使用，
    #    不进任何 prompt。见 services/vision.py 的「边界」。
    #
    # 截断发生在**这里**而不是 `extract_books_from_image` 里，是因为两个上限
    # 管的是两件事：那边挡的是「模型吐了 200 行」这种垃圾输出，这边挡的是算力。
    # recognized_books 照原样回给调用方，差额看得见。
    to_search = books[: vision._MAX_SEARCH_PER_IMAGE]
    if len(to_search) < len(books):
        log.info(
            "以图搜书：认出 %d 本，超过检索上限 %d，只搜前 %d 本",
            len(books),
            vision._MAX_SEARCH_PER_IMAGE,
            len(to_search),
        )

    # 逐本顺序检索，不做并发。
    # 检索里最重的是本地的 cross-encoder 精排，它是 CPU 密集的，
    # 拿线程池并发只会互相抢核、还引入「谁能同时调 embedding 模型」的线程安全问题 ——
    # 为了 10 本书省那一两秒，不值得。
    results = []
    for book in to_search:
        hits = search_service.search(
            query=book, top_k=req.top_k, min_score=req.min_score
        )
        results.append(
            BookSearchResult(
                book=book,
                query=book,
                count=len(hits),
                hits=_to_hits(hits),
            )
        )

    return SearchByImageResponse(
        recognized_books=books,
        min_score=req.min_score,
        results=results,
    )


# ============================================================================
# M3 第 4 步：对话
# ============================================================================

# 数据库 role 列 -> LangChain 的消息类。
# 之所以能一行查表搞定，是因为建表时 role 存的就是 LangChain 自己的 .type 值
# ("human" / "ai")，没有另创一套 user / assistant 的叫法（见 ChatMessage 类注释）。
_ROLE_TO_CLASS = {"human": HumanMessage, "ai": AIMessage}


def _rows_to_messages(rows: list[dict]) -> list:
    """数据库行 -> LangChain 消息对象。给 chat_once 当 history 用。
    """
    result = []
    for row in rows:
        result.append(_ROLE_TO_CLASS[row["role"]](content = row["content"]))
    return result


def _messages_to_rows(messages: list) -> list[dict]:
    """LangChain 消息对象 -> 数据库行。给 save_messages 用。
    """
    result = []
    for message in messages:
        result.append(
            {
                "role": message.type,
                "content":message.content
            }
        )
    return result


def _collect_books(trace: list[dict]) -> list[BookCard]:
    """这一轮工具碰到的书 -> 完整书卡。给前端渲染用。

    为什么要这一步：`chat_once` 的 trace 里只有 `result_count`（几条），
    而卡片要书名/版次/价格/成色/原帖。**数据在 loop 里被丢掉了** ——
    所以让 trace 带上 `post_ids`，这里拿 id 回主服务补全。

    两条设计取舍：
      * **去重、保留首次出现的顺序**。模型可能先 search_books 拿到 5 个 id，
        再 get_post_detail 查其中一本 —— 那本会出现两次，但只该有一张卡。
      * **查不到的 id 直接跳过**，不补一张空卡。空卡（书名价格全空）比没有卡更难看，
        而且它表达不了「这本书被删了」这个信息。
    """
    # 用 dict 而非 set：要保留插入顺序（set 是无序的，卡片顺序会随机跳）
    ids: dict[int, BookCard] = {}

    for record in trace:
        for post_id in record.get("post_ids", []):
            if post_id in ids:
                continue
            # 复用 get_post_detail —— 它已经把主服务的原始字段翻译过了
            # （price 的 null -> "未标价"、hasNotes 的 null -> "未提及"、status 转中文）。
            # 这里要是自己再调一次 get_post() 再翻译一遍，那套口径就有了两份，
            # 改一处漏一处是迟早的事。
            rows = get_post_detail.invoke({"post_id": post_id})
            if not rows:
                continue  # 主服务说这条不存在（或已删），跳过而不是造空卡
            row = rows[0]
            ids[post_id] = BookCard(
                post_id=row["post_id"],
                book_name=row.get("book_name"),
                edition=row.get("edition"),
                price=row["price"],
                condition_desc=row.get("condition_desc"),
                raw_text=row.get("raw_text"),
                status=row.get("status", "状态未知"),
            )

    return list(ids.values())


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["对话"],
    summary="和 agent 对话（带会话记忆）",
    description="""
用户说一句，agent 自己决定要不要查库、查几次，然后回答。

## 记忆怎么工作

请求里**不传 `session_id` 就是开一个新会话**，服务端生成后放在响应里返回。
前端存住它，下一句带上 —— 这样 agent 才知道「第一条多少钱」问的是哪一批书。

历史存在**主服务的 MySQL** 里（`chat_message` 表），本服务每次通过 REST 读写。
不直连数据库是铁律一：本服务要能随便重启、随便扩多副本，不持有任何业务数据连接。

## 每一轮只存两条

agent 一轮跑完，`chat_once` 返回的是**整段会话**（旧历史 + 新两条）。
不能整个存回去 —— 旧历史已经在库里了，整个存会一轮比一轮重复。
只切出尾巴上那两条，**一次 POST 提交**（主服务那边一个事务写，不会出现有问无答）。

## `tool_calls` 为什么要暴露

这是整个项目最直观的加分项。只有 `reply` 的话，看的人分不清
「agent 自己决定查了库」和「硬编码了一段回答」。
把轨迹摊开，才能证明它真的在决策 —— 演示和排查都靠这一栏。

## `books` 是怎么来的

**不是模型给的，是服务端补的。** 模型只输出文字，卡片要的书名/版次/价格/成色/原帖
它一个字都不会给你。

流程是：工具返回的书 id 由 trace 带出来（`tool_calls[].post_ids`）→ 这里去重 →
按 id 回主服务取全量字段 → 组装成 `books`。

`tool_calls` 里只有 `result_count`（几条）是不够的：`search_books` 返回的摘要
只有书名/价格两栏，做不了卡片。**而且只调 `search_books` 时
`arguments` 里没有 post_id**，前端没法自己补 —— 所以只能由服务端带出来。
""",
    response_description="agent 的回答 + 这一轮的工具调用轨迹 + 涉及到的书卡",
    responses={
        400: {"description": "message 为空，或 session_id 形状不合法"},
        429: {"description": "请求太频繁（默认 20 次/分钟/IP），带 Retry-After"},
        503: {"description": "主服务不可用（读写历史失败）"},
    },
)
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    # 限流放在**最前面**：连 session_id 都还没生成就该挡掉。
    # 放到后面的话，被限的请求仍然会走完「读历史」那一步 —— 白打一次主服务。
    _enforce_limit(request, "chat")

    # 不传就是新会话。uuid4().hex 只含字母数字，
    # 正好落在 session_id 允许的字符集里（见 ChatRequest.session_id 的说明）。
    session_id = req.session_id or uuid.uuid4().hex

    try:
        # ---- 1. 读这个会话之前说过的话 ----------------------------------
        # 走带缓存的版本（cache-aside）。**这里不改变任何对外行为** ——
        # 缓存要么返回主服务真的给过的东西，要么就回源，见 session_cache.py。
        history = _rows_to_messages(session_cache.load_history_cached(session_id))

        # ---- 2. 跑一轮 agent --------------------------------------------
        reply, trace, memory = chat_once(req.message, history)

        # ---- 3. 把这一轮碰到的书补全成卡片 -------------------------------
        # 放在同一个 try 里、save_messages 之前：补卡片也要打主服务，
        # 主服务挂了的话第 4 步本来也存不了历史、照样 503 —— 不用单独兜。
        books = _collect_books(trace)

        # ---- 4. 只存这一轮新产生的那两条 ---------------------------------
        # chat_once 返回的第三个值是**整段会话**，但旧历史已经在库里了，
        # 整个存回去会一轮比一轮重复。只切尾巴上那两条，一次 POST 提交。
        # 带缓存的版本会**在写成功之后**把这个会话的缓存删掉（写时失效）。
        new_history = memory[-2:]
        session_cache.save_messages_cached(session_id, _messages_to_rows(new_history))

    except BookServiceError as e:
        # 主服务连不上 —— 这是下游故障，不是调用方的错，所以 503 而不是 400。
        # 消息原样带出去：BookServiceError 里已经写了「确认 Spring Boot 已启动」这类提示。
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        session_id=session_id, reply=reply, tool_calls=trace, books=books
    )
