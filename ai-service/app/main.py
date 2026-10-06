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
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.loop import chat_once
from app.agent.tools import get_post_detail
from app.clients.book_service import BookServiceError, load_history, save_messages
from app.config import settings
from app.schemas import (
    BatchItemResult,
    BookCard,
    BookInfo,
    ChatRequest,
    ChatResponse,
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
只有书名/价格/有没有笔记三栏，做不了卡片。**而且只调 `search_books` 时
`arguments` 里没有 post_id**，前端没法自己补 —— 所以只能由服务端带出来。
""",
    response_description="agent 的回答 + 这一轮的工具调用轨迹 + 涉及到的书卡",
    responses={
        400: {"description": "message 为空，或 session_id 形状不合法"},
        503: {"description": "主服务不可用（读写历史失败）"},
    },
)
def chat(req: ChatRequest) -> ChatResponse:
    # 不传就是新会话。uuid4().hex 只含字母数字，
    # 正好落在 session_id 允许的字符集里（见 ChatRequest.session_id 的说明）。
    session_id = req.session_id or uuid.uuid4().hex

    try:
        # ---- 1. 读这个会话之前说过的话 ----------------------------------
        history = _rows_to_messages(load_history(session_id))

        # ---- 2. 跑一轮 agent --------------------------------------------
        reply, trace, memory = chat_once(req.message, history)

        # ---- 3. 把这一轮碰到的书补全成卡片 -------------------------------
        # 放在同一个 try 里、save_messages 之前：补卡片也要打主服务，
        # 主服务挂了的话第 4 步本来也存不了历史、照样 503 —— 不用单独兜。
        books = _collect_books(trace)

        # ---- 4. 只存这一轮新产生的那两条 ---------------------------------
        # chat_once 返回的第三个值是**整段会话**，但旧历史已经在库里了，
        # 整个存回去会一轮比一轮重复。只切尾巴上那两条，一次 POST 提交。
        new_history = memory[-2:]
        save_messages(session_id, _messages_to_rows(new_history))

    except BookServiceError as e:
        # 主服务连不上 —— 这是下游故障，不是调用方的错，所以 503 而不是 400。
        # 消息原样带出去：BookServiceError 里已经写了「确认 Spring Boot 已启动」这类提示。
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        session_id=session_id, reply=reply, tool_calls=trace, books=books
    )
