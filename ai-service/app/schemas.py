"""AI 服务的请求 / 响应模型。

字段命名统一用 snake_case（Python 惯例），
Java 侧通过 @JsonProperty 做映射，两边各用各的习惯。

这里的 Field(description=...) 不只是注释 —— FastAPI 会把它们读进 OpenAPI，
直接显示在 Swagger UI 的字段说明里。**改说明等于改文档**，不用另写一份。
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class BookInfo(BaseModel):
    """从帖子文本中抽取出的结构化书籍信息。

    所有字段都允许为 None —— 真实的帖子信息残缺是常态，
    抽不到就填 null，**绝不允许模型编造**。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "book_name": "高等数学",
                "edition": "第七版",
                "publisher": None,
                "author": None,
                "condition_desc": "有笔记，划过重点",
                "price": 40.0,
                "has_notes": True,
            }
        }
    )

    book_name: Optional[str] = Field(
        default=None,
        description="标准书名，如「高等数学」；抽不到填 null",
        examples=["高等数学"],
    )
    edition: Optional[str] = Field(
        default=None,
        description="版次，如「第七版」；抽不到填 null",
        examples=["第七版"],
    )
    publisher: Optional[str] = Field(
        default=None,
        description="出版社；**只有原文明确写了才填**，靠常识补全的一律为 null",
        examples=[None],
    )
    author: Optional[str] = Field(
        default=None,
        description="作者；抽不到填 null",
        examples=["严蔚敏"],
    )
    condition_desc: Optional[str] = Field(
        default=None,
        description="成色描述，如「九成新」「书角有点卷」",
        examples=["有笔记，划过重点"],
    )
    price: Optional[float] = Field(
        default=None,
        description="价格（元），统一为数字；抽不到填 null。免费为 0.0，面议为 null",
        examples=[40.0],
    )
    has_notes: Optional[bool] = Field(
        default=None,
        description="是否带笔记；未提及填 null",
        examples=[True],
    )


class ExtractRequest(BaseModel):
    """抽取请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"raw_text": "高数同济七版 上下册一起 40 有笔记 划重点了 可小刀"}
        }
    )

    raw_text: str = Field(
        description="卖家发布的帖子原文（混乱的自然语言）",
        examples=["高数同济七版 上下册一起 40 有笔记 划重点了 可小刀"],
    )


class ExtractResponse(BaseModel):
    """抽取响应。"""

    raw_text: str = Field(description="原样回显输入的帖子文本，便于调用方对齐")
    book_info: BookInfo = Field(description="抽取出的结构化字段")


# ============================================================
# 批量抽取（M1）
# ============================================================


class ExtractBatchRequest(BaseModel):
    """批量抽取请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "高数同济七版 上下册一起 40 有笔记",
                    "线代同济六版 15r 九成新 无笔记",
                ],
                "max_concurrency": 8,
            }
        }
    )

    texts: list[str] = Field(
        min_length=1,
        max_length=200,
        description="帖子原文列表，1~200 条",
    )
    max_concurrency: Optional[int] = Field(
        default=8,
        ge=1,
        description="最大并发数。太小慢，太大会被对方 API 限流（429）",
    )
    batch_size: int = Field(
        default=20,
        ge=1,
        le=50,
        description=(
            "一次 LLM 请求里塞几条帖子。"
            "1 = 逐条抽取（每条都重发一遍 system prompt，token 最贵，但失败粒度最细）；"
            ">1 = 真批量，system prompt 每 N 条才发一次。"
            "默认 20 可把 token 降到约 1/6，代价是一个 chunk 失败会带走这一整批。"
        ),
    )


class BatchItemResult(BaseModel):
    """批量结果里的单条。

    book_info 和 error **恰好有一个非空**：
    成功时 book_info 有值、error 为 None；失败时反过来。
    """

    raw_text: str = Field(description="原样回显，便于调用方按文本对齐")
    book_info: Optional[BookInfo] = Field(
        default=None, description="抽取结果；失败时为 null"
    )
    error: Optional[str] = Field(
        default=None, description="错误信息；成功时为 null"
    )


class ExtractBatchResponse(BaseModel):
    """批量抽取响应。

    results 的顺序与请求的 texts **严格一致**（不是完成顺序）。
    """

    results: list[BatchItemResult]


# ============================================================
# 语义检索（M2）
# ============================================================


class IndexBuildRequest(BaseModel):
    """建索引请求。"""

    model_config = ConfigDict(
        json_schema_extra={"example": {"limit": 500, "rebuild": False}}
    )

    limit: int = Field(
        default=20000,
        ge=1,
        le=100000,
        description=(
            "最多建多少条。默认 20000 当「全量」用。"
            "**不是主服务的那个 500** —— 客户端会按 500 一页翻，总数由这里定。"
        ),
    )
    rebuild: bool = Field(
        default=False,
        description=(
            "True = 先清空整个向量库再重建。"
            "平时用 False：写入用的是 upsert，同一 id 会覆盖，重复调用是幂等的。"
            "只有在**换了 embedding 模型**或改了 compose_text 之后才需要 True —— "
            "那时旧向量和新向量不在同一个空间里，必须全部重算。"
        ),
    )


class IndexBuildResponse(BaseModel):
    """建索引结果统计。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fetched": 10000,
                "indexed": 9997,
                "skipped": 3,
                "collection_count": 9997,
                "dim": 512,
                "elapsed_ms": 22913,
            }
        }
    )

    fetched: int = Field(description="从主服务拉到的 DONE 帖子数")
    indexed: int = Field(description="真正写入向量库的数量")
    skipped: int = Field(description="被跳过的数量（抽不出书名的，拼出来是噪音）")
    collection_count: int = Field(description="写完以后向量库里的总条数")
    dim: int = Field(description="向量维度。bge-small-zh-v1.5 是 512")
    elapsed_ms: int = Field(description="耗时（毫秒）")
    note: Optional[str] = Field(default=None, description="异常情况下的说明")


class SearchRequest(BaseModel):
    """检索请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "同济版高数", "top_k": 10, "min_score": 0.5}
        }
    )

    query: str = Field(
        min_length=1,
        description="自然语言查询，不需要跟帖子里的用词一致，如「同济版高数」",
        examples=["同济版高数"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="最多返回几条",
    )
    min_score: Optional[float] = Field(
        default=None,
        description=(
            "相似度下限，低于它的丢掉。不传则用服务端的默认阈值。"
            "向量检索**永远**会返回 top_k 条，哪怕库里根本没这本书 —— "
            "所以必须有这一步，否则搜「考古」也会返回一堆高数。"
        ),
    )
    rerank: Optional[bool] = Field(
        default=None,
        description=(
            "是否走 cross-encoder 精排。不传则用服务端配置的默认值。"
            "**留这个开关是为了做 A/B** —— 传 false 能拿到纯召回的排序，"
            "用来量化「精排到底有没有净增益」，而不是靠感觉说它有用"
        ),
    )


class SearchHit(BaseModel):
    """一条检索命中。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "post_id": 42,
                "score": 0.83,
                "text": "高等数学 第七版 同济大学出版社",
                "book_name": "高等数学",
                "author": None,
                "publisher": "同济大学出版社",
                "edition": "第七版",
                "price": 40.0,
                "has_notes": True,
            }
        }
    )

    post_id: int = Field(description="帖子在主服务里的主键，拿它可以查详情")
    score: float = Field(description="余弦相似度，越大越像。理论上限 1.0")
    rerank_score: Optional[float] = Field(
        default=None,
        description=(
            "精排分（cross-encoder，sigmoid 到 0~1）；没开 rerank 时为 null。"
            "**它和 score 不可比** —— 两个模型的输出，别混着排序。"
            "它也不是余弦的修正版，是另一套判定：score 看「整体语义方向」，"
            "rerank_score 看「这两段文字是不是在说同一件事」"
        ),
    )
    text: str = Field(description="真正被向量化的那段文本，出问题时用它排查")
    book_name: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    publisher: Optional[str] = Field(default=None)
    edition: Optional[str] = Field(default=None)
    price: Optional[float] = Field(
        default=None, description="价格为 -1 表示原帖没写；没写的已转成 null"
    )
    has_notes: bool = Field(default=False)


class SearchResponse(BaseModel):
    """检索响应。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "同济版高数",
                "min_score": 0.5,
                "count": 2,
                "hits": [
                    {
                        "post_id": 42,
                        "score": 0.83,
                        "text": "高等数学 第七版 同济大学出版社",
                        "book_name": "高等数学",
                        "author": None,
                        "publisher": "同济大学出版社",
                        "edition": "第七版",
                        "price": 40.0,
                        "has_notes": True,
                    }
                ],
            }
        }
    )

    query: str = Field(description="原样回显查询，便于调用方对齐")
    min_score: Optional[float] = Field(
        default=None, description="本次实际生效的阈值（None 表示没过滤）"
    )
    count: int = Field(description="过滤后的命中数。可能为 0，这是正常结果不是错误")
    hits: list[SearchHit]


# ============================================================================
# M2.6 多模态：以图搜书
# ============================================================================


class SearchByImageRequest(BaseModel):
    """以图搜书的请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...", "top_k": 5}
        }
    )

    image: str = Field(
        min_length=1,
        description=(
            "图片，二选一："
            "① base64 data URL，形状 `data:image/jpeg;base64,<...>`；"
            "② http(s) 直链。**不接受本地文件路径** —— 那等于开了个读任意文件的口子。"
        ),
    )
    top_k: int = Field(default=10, ge=1, le=50, description="最多返回几条")
    min_score: Optional[float] = Field(
        default=None, description="余弦下限，不传则用服务端默认阈值（与 /search 同一套）"
    )


class BookSearchResult(BaseModel):
    """图里认出的**一本书**，以及它检索出来的帖子。

    `book` 有值而 `count == 0` 是有意义的结果，不是错误：
    **识别对了，但库里没有这本书** —— 该查语料覆盖，不是查视觉模型。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "book": "高等数学",
                "query": "高等数学",
                "count": 2,
                "hits": [],
            }
        }
    )

    book: str = Field(
        description="视觉模型从图里读出来的书名。⚠️ 来自图片，是**不可信输入**，只被当检索词用"
    )
    query: str = Field(
        description="实际拿去检索的词。目前等于 book，单独列出来是为了将来可替换"
    )
    count: int = Field(description="这一本命中几条，可能是 0")
    hits: list[SearchHit]


class SearchByImageResponse(BaseModel):
    """以图搜书的响应。

    ## 为什么是列表，不是一个书名

    真实图片常常是**一排书脊**（二手书群里的典型图），一张图里有十几本书。
    只回一个书名等于「随便挑了其中一本」，而且调用方看不出另外十几本被丢了。
    单本封面图走同一条路，结果就是长度为 1 的列表。

    ## 两个列表的长度可能不一样，这个差额是**故意露出来**的

    `recognized_books` 是模型认出来的全部（上限 20），
    `results` 是**真的去检索了**的那些（上限 10，只覆盖前 N 本，同序一一对应）。

    为什么分开设上限：识别是一次模型调用，认 1 本和认 20 本的钱一样；
    检索是每本一次完整链路（BM25 + 向量 + RRF + 精排），全是真算力。
    所以「认得出多少」放宽，「搜多少」收紧。
    **认了 20 本只搜 10 本，用户必须看得见**（`len(results) < len(recognized_books)`），
    而不是我们默默丢掉一半、装作只认出 10 本。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recognized_books": ["高等数学", "线性代数"],
                "min_score": None,
                "results": [
                    {
                        "book": "高等数学",
                        "query": "高等数学",
                        "count": 2,
                        "hits": [],
                    }
                ],
            }
        }
    )

    recognized_books: list[str] = Field(
        description=(
            "视觉模型从图里读出来的书名，按图里的顺序。"
            "**空列表 = 一本书都没认出来**（这种情况 results 必然也是空的，"
            "不是检索失败），该让用户重拍而不是去查语料"
        )
    )
    min_score: Optional[float] = Field(
        default=None, description="本次实际生效的阈值（None 表示没过滤）"
    )
    results: list[BookSearchResult] = Field(
        description=(
            "每本一条，**是 recognized_books 的前缀**（同序一一对应）。"
            "比 recognized_books 短就说明认出的书超过了检索上限"
        )
    )


# ============================================================================
# M3 第 4 步：会话接口
# ============================================================================


class ChatRequest(BaseModel):
    """对话请求。"""

    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "有笔记的线代成色怎么样"}}
    )

    message: str = Field(
        min_length=1,
        description="用户这一句原话",
        examples=["有笔记的线代成色怎么样"],
    )
    session_id: Optional[str] = Field(
        default=None,
        # ⚠️ 必须自己锚定 ^...$。Pydantic 的 pattern 是**子串匹配**：
        #    写成 [A-Za-z0-9_-]{1,64} 的话，"a/b c" 里含一个 a 就算通过，
        #    整个约束形同虚设。实测过。
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description=(
            "会话 ID。**不传就开一个新会话**，服务端生成后放在响应里返回；"
            "前端存下来、下一句带上，agent 就有记忆了。"
            "形状跟主服务 chat_message.session_id 列对齐（也是 VARCHAR(64)），"
            "因为它是会拼进 URL 路径的：带斜杠的会被路由拆开，"
            "带空格的编码后会被 Tomcat 直接拒掉。"
        ),
    )


class ToolCallRecord(BaseModel):
    """这一轮 agent 调了一次工具的记录。"""

    name: str = Field(description="工具名", examples=["search_books"])
    arguments: dict = Field(
        description="模型自己拆出来的参数。看这一栏能判断它是真会调、还是把整句话当 query 塞进去",
        examples=[{"query": "线性代数", "has_notes": True}],
    )
    result_count: int = Field(description="工具返回了几条。0 也是有效结果（没找到）", examples=[5])
    post_ids: list[int] = Field(
        default_factory=list,
        description=(
            "这次工具返回的书 id。**卡片的数据入口** —— search_books 的摘要只有"
            "书名/价格两栏，做不了卡片；靠这批 id 回主服务补全成 books。"
            "失败路径（幻觉工具名 / 工具报错）给空列表，不是缺字段。"
        ),
        examples=[[139, 133, 124, 98, 85]],
    )


class BookCard(BaseModel):
    """一张书卡。给前端渲染用 —— 字段口径跟 `get_post_detail` 返回的完全一致。

    ⚠️ `price` / `has_notes` 是**联合类型**，不是可选：源头分不清的事出口不替它下结论。
    `price` 为 null 时给 `"未标价"`（不是 "面议" —— 那是在替卖家承诺可以还价），
    `has_notes` 为 null 时给 `"未提及"`（不是 False —— 那是替卖家撒谎说没有笔记）。
    """

    post_id: int = Field(description="帖子 id，点卡片跳详情用")
    book_name: Optional[str] = Field(default=None, description="书名")
    edition: Optional[str] = Field(default=None, description="版次，如「第六版」")
    price: Union[float, str] = Field(
        description="价格。数字，或字符串 `未标价`（源头没标价，不是 0）", examples=[8.0]
    )
    condition_desc: Optional[str] = Field(
        default=None, description="成色描述，卖家自己的说法"
    )
    raw_text: Optional[str] = Field(default=None, description="卖家原帖原文")
    status: str = Field(default="状态未知", description="在售 / 已售 / ...")


class ChatResponse(BaseModel):
    """对话响应。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "6f1c0e6a9b3d4e2f8a7c5b1d0e3f4a6b",
                "reply": "139 号《线性代数》有笔记且划过重点，8 元，还在售。",
                "tool_calls": [
                    {"name": "search_books",
                     "arguments": {"query": "线性代数", "has_notes": True},
                     "result_count": 5,
                     "post_ids": [139, 133, 124, 98, 85]},
                    {"name": "get_post_detail",
                     "arguments": {"post_id": 139},
                     "result_count": 1,
                     "post_ids": [139]},
                ],
                "books": [
                    {"post_id": 139, "book_name": "线性代数", "edition": "第六版",
                     "price": 8.0, "condition_desc": "有笔记，划过重点",
                     "raw_text": "线代 同济六版 8块 有笔记 划重点", "status": "在售"}
                ],
            }
        }
    )

    session_id: str = Field(
        description="会话 ID。首次调用时由服务端生成，之后原样带回 —— 前端要存住它，否则没有记忆"
    )
    reply: str = Field(description="给用户看的回答")
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description=(
            "这一轮的工具调用轨迹。**必须暴露出来**：只有 reply 的话，"
            "看不出它是真 agent 还是硬编码的问答。演示和排查都靠这一栏。"
        ),
    )
    books: list[BookCard] = Field(
        default_factory=list,
        description=(
            "这一轮涉及到的书，给前端渲染卡片用。**不是模型给的，是服务端补的** ——"
            "从 tool_calls 里收集 post_ids，去重后回主服务取全量字段。"
            "模型一句没找到就是空列表。"
        ),
    )


class UsageStats(BaseModel):
    """进程启动以来的 LLM token 用量估算（M3.2）。

    ⚠️ 三个必须说清的边界，别当成账单：
    1. **进程内累计**，重启即清零；多副本各算各的。
    2. **只统计文本 LLM**。视觉模型自带免费额度、计费口径不同，不计入；
       本地 BGE embedding 在 CPU 上跑，本来就不花钱。
    3. 成本是按配置里的**单价估算**的（`pricing_cny_per_mtok` 会原样返回），
       不是从账单同步的真实值。官方调价了要改 `.env`。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "llm_calls": 42,
                "prompt_tokens": 51200,
                "completion_tokens": 8400,
                "total_tokens": 59600,
                "estimated_cost_cny": 0.1696,
                "calls_without_usage": 0,
                "pricing_cny_per_mtok": {"input": 2.0, "output": 8.0},
            }
        }
    )

    llm_calls: int = Field(description="累计 LLM 调用次数（含 batch 里的每一次）")
    prompt_tokens: int = Field(description="累计输入 token")
    completion_tokens: int = Field(description="累计输出 token")
    total_tokens: int = Field(description="输入 + 输出")
    estimated_cost_cny: float = Field(
        description="按配置单价估算的成本（元）。**估算值**，非账单"
    )
    calls_without_usage: int = Field(
        default=0,
        description=(
            "成功结束但**拿不到 usage** 的调用数。**这一栏 > 0 就说明账目不全** ——"
            "比如 provider 换了字段名。它存在的意义就是别让「统计失效」伪装成「成本为 0」"
        ),
    )
    pricing_cny_per_mtok: dict[str, float] = Field(
        default_factory=dict, description="算这笔账用的单价，方便判断数字是否还作数"
    )
