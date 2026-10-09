"""书籍信息抽取：混乱的帖子文本 -> 结构化字段。

思路：不给模型写规则解析器，而是把「怎么抽」写成 prompt，
让模型输出一个符合 BookInfo 的 JSON，再交给 pydantic 校验。

调用链只有三步：
    get_llm()                          -> DeepSeek 客户端
    llm.with_structured_output(...)    -> 包一层，保证输出能转成 BookInfo
    .invoke([system, human])           -> 真正发请求，拿回 BookInfo 对象

## 两个批量函数，别搞混（名字很像，省的完全不是一回事）

| 函数 | 一次请求几条帖子 | 省什么 |
|---|---|---|
| `extract_book_info_batch` | **1 条** | 只省**串行等待**的时间，token / 计费完全不变 |
| `extract_book_info_many`  | **N 条** | 真省 token —— system prompt 从「每条一次」变成「每 20 条一次」 |

`extract_book_info_many` 是给**离线灌数据**用的（建库时一次几万条）；
线上接口收帖子是**一条条来**的，用不上批量，走前一个或单条即可。

为什么批量能把 token 砍到 1/6：`SYSTEM_PROMPT` 有 1500 多字（≈1200 token），
逐条抽取时**每条帖子都要把这 1500 字重发一遍**。1 万条帖子 = 1200 万 token
全花在重复发送同一段提示词上。批量后 20 条共用一次，同样的钱能多抽 10 倍。

调 prompt 用：.venv/Scripts/python.exe scripts/try_extract.py
"""

from pydantic import BaseModel, Field

from app.schemas import BookInfo
from app.services.llm import get_llm

# 【关键在这。】字段说明写在 schemas.py 的 Field(description=...) 里，
# 这里只写「判定规则」——模型最容易搞错的那些边界情况。
SYSTEM_PROMPT = """你是二手书交易平台的信息抽取器。把卖家的帖子原文抽成结构化字段。

## 铁律
1. **只抽原文写了的信息。原文没提的，填 null —— 绝对不要靠常识补全或猜测。**
   - 例："出一本高数" → publisher=null, author=null, price=null
   - 例："同济七版高数" 只说明版次是第七版，不代表出版社是"同济大学出版社"
2. 原文里的错别字、口语、缩写要理解对，但输出用规范写法。

## 逐字段规则

**book_name**：归一到标准全名，不要保留版次、册数、成色、价格。
  - "高数" / "高等数学同济" / "高数同济七版上下册" → "高等数学"
  - "线代" / "线性代数" → "线性代数"
  - "概率论" / "概率论与数理统计浙大版" → "概率论与数理统计"
  - "数据结构严蔚敏" → "数据结构"
  - "大物" → "大学物理"
  - "四级词汇书" → "英语四级词汇"
  - 实在无法归一（如"考研资料"）就按原文填。

**edition**：只填版次，格式统一为「第X版」。
  - "七版" / "7版" / "第七版" / "第7版" → "第七版"（用中文数字）
  - "上下册" 是册数不是版次，不要填进 edition。

**publisher**：**只有原文出现「XX出版社」或明确的出版社简称时才填**（"清华"→"清华大学出版社"）。
  - ⚠️ 版本体系/编者单位的简称**不是出版社**："高数同济七版" 里的"同济"指同济大学编的版本，
    推论不出出版社 → publisher=null。"浙大版概率论" 同理 → null。
  - 宁可填 null，也不要靠常识补一个出版社出来。

**author**：原文明确写了作者名才填，用原文写法（"严蔚敏"就这样填，不要补全成"严蔚敏等"）。

**price**：统一成数字（float），单位元。删掉货币符号和量词。
  - "20" / "20出" / "20r" / "20块" / "二十" / "20元" → 20.0
  - "25.5" / "25.5包邮" → 25.5
  - "40打包" 如果指的是两本书一起 40，仍填 40.0
  - "免费" / "白送" / "送" → 0.0（这是明确的零价，不是缺失）
  - "价格可议" / "面议" / 没写价格 → null（**区别于上面，这才是缺失**）

**condition_desc**：把原文里所有关于**品相/状态**的描述合并成一句简短中文。
  - "九成新" / "八成新" / "几乎全新" / "书角有点卷" / "有点破损" / "翻过几次"
  - 笔记、划线类的说明**也归在这里**（它们是书的状态）：
    "有笔记 划重点了" → "有笔记，划过重点"；"无笔记" → "无笔记"
  - 完全没提到品相 → null
  - 注意这只是「抄写+归纳原文」，不是在判断好坏，不要加原文没有的评价

**has_notes**：只回答「有没有笔记/划重点/标注」这一个**布尔**问题，
  和 condition_desc 不冲突——原文提到笔记时，两个字段都要填。
  - "有笔记" / "划过重点" / "做了笔记" / "有批注" → true
  - "无笔记" / "没写过" / "很干净" → false
  - 完全没提笔记 → null

## 输出
只输出字段，不要解释。"""

# 客户端和结构化输出包装只建一次，别每次调用都重建
# （ChatOpenAI 本身无状态，反复构造只是浪费）
_structured_llm = None
# 批量路径的那个（输出 schema 不同，所以是另一个包装器，不能共用）
_batch_structured_llm = None


def _get_structured_llm():
    """拿到「输出必为 BookInfo」的 LLM 包装器，惰性初始化 + 复用。"""
    global _structured_llm
    if _structured_llm is None:
        llm = get_llm(temperature=0.0)
        # ⚠️ method 必须显式写 function_calling：
        # DeepSeek 不认默认的 json_schema 模式，会直接 400
        _structured_llm = llm.with_structured_output(
            BookInfo, method="function_calling"
        )
    return _structured_llm


def extract_book_info(raw_text: str) -> BookInfo:
    """从帖子原文抽取结构化书籍信息。

    Args:
        raw_text: 卖家发布的原始帖子文本

    Returns:
        抽取出的结构化信息，缺失字段为 None
    """
    text = (raw_text or "").strip()
    if not text:
        # 空帖子直接返回全 None，省一次 API 调用
        return BookInfo()

    return _get_structured_llm().invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", text),
        ]
    )


def extract_book_info_batch(
    texts: list[str],
    *,
    max_concurrency: int | None = None,
    return_exceptions: bool = False,
) -> list[BookInfo | Exception]:
    """批量抽取。内部是**并发发 N 个请求**，不是一次请求拿 N 条结果。

    所以它省的是串行等待的时间，token 消耗 / 调用次数 / 计费完全不变。

    Args:
        texts: 帖子原文列表
        max_concurrency: 最多同时发几个请求，必须 >= 1。
            不传（None）时交给 Python 线程池的默认值 min(32, CPU核数+4)——
            ⚠️ 注意这**不是「不限」**，本机 16 核就是 20 个线程同时打出去。
            条数多时务必显式设小，否则会打到对方 API 的 429 限流。
            传 0 或负数直接报错，不静默兜底。
        return_exceptions: True 时把失败的槽位放异常对象，而不是整批抛错。
            调 prompt 时用 True —— 能看清是哪条帖子失败，而不是全批丢掉。

    Returns:
        和 texts 等长、顺序一致的列表（返回顺序 ≠ 完成顺序，LangChain 负责对齐）。
    """
    cleaned = [(t or "").strip() for t in texts]

    # 空文本不进 LLM，直接给全 None 的占位，省调用
    results: list[BookInfo | Exception] = [BookInfo() for _ in cleaned]
    pending = [i for i, t in enumerate(cleaned) if t]
    if not pending:
        return results

    messages = [
        [("system", SYSTEM_PROMPT), ("human", cleaned[i])] for i in pending
    ]

    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError(
            f"max_concurrency 必须 >= 1（不传则用线程池默认值），收到 {max_concurrency}"
        )

    # 这里必须写 `is not None`，不能写 `if max_concurrency`：
    # Python 里 0 是假值，用真值判断会让 max_concurrency=0 悄悄退化成"用默认值"，
    # 调用方以为限制了并发、实际跑满 20 线程 —— 静默行为不一致比报错危险
    config = (
        {"max_concurrency": max_concurrency} if max_concurrency is not None else None
    )
    raw = _get_structured_llm().batch(
        messages, config=config, return_exceptions=return_exceptions
    )

    for i, item in zip(pending, raw):
        results[i] = item

    return results


# ============================================================================
# 真·批量抽取（离线灌数据用）
# ============================================================================

# 一次请求塞几条帖子。定 20 的理由：
#   - 太小（比如 3）：system prompt 还是被重复发送，省不下来
#   - 太大：单次输出太长，模型容易漏条、也容易触发最大输出长度截断
#   - 20 条约 2000 token 输出，DeepSeek 的输出上限（默认 4k~8k）里留足余量
_EXTRACT_BATCH = 20

# 批量模式额外加的一段提示。**不是**改 SYSTEM_PROMPT ——
# 逐条抽取那条路还在线上跑着，改共同的部分会影响它。
_BATCH_SUFFIX = """

## 这次是批量输入

你会一次收到**多条**帖子，每条用 `### 帖子 N` 开头（N 从 1 开始）。
请对**每一条**都独立抽一遍，输出里 `idx` 填那条帖子的 N。

- 一条都不能漏，也不要合并两条
- 每条都按上面同样的规则抽，互不影响"""


class _BatchItem(BaseModel):
    """批量输出里的一项：序号 + 这条帖子的抽取结果。"""

    idx: int = Field(description="对应第几条帖子，从 1 开始，必须和输入里的编号一致")
    info: BookInfo


class _BatchOut(BaseModel):
    """批量抽取的输出容器。

    ⚠️ 为什么要包一层 `items` 而不是直接返回 `list[BookInfo]`：
    function calling 的入参必须是**对象**（函数参数是带名字的），不能是裸数组。
    所以套一个 items 字段。这也是为什么生成脚本里 MockPostBatch 也长这样。
    """

    items: list[_BatchItem]


def _build_batch_messages(chunk: list[str]) -> list[tuple[str, str]]:
    """把一批帖子拼成一条请求的消息。

    每条前面加 `### 帖子 N` 编号 —— 模型靠它回填 idx，
    我们靠 idx 把结果对回原始下标（不然就只能靠顺序，而顺序不可靠）。
    """
    body = "\n\n".join(
        f"### 帖子 {i}\n{text}" for i, text in enumerate(chunk, start=1)
    )
    return [("system", SYSTEM_PROMPT + _BATCH_SUFFIX), ("human", body)]


def extract_book_info_many(
    texts: list[str],
    *,
    batch_size: int = _EXTRACT_BATCH,
    max_concurrency: int | None = None,
) -> list[BookInfo | Exception]:
    """**真·批量**抽取：一次请求塞多条帖子。给离线灌数据用。

    和 `extract_book_info_batch` 的区别见模块开头那张表 —— 这个才是省 token 的。

    Args:
        texts: 帖子原文列表
        batch_size: 一次请求塞几条。必须 >= 1
        max_concurrency: 最多同时发几个**请求**（注意这里的单位是请求，不是帖子 ——
            一个请求已经含 batch_size 条了，所以这个值该比逐条模式小得多）

    Returns:
        和 texts 等长、顺序一致的列表，**成功的是 BookInfo，失败的是 Exception**。
        形状和 `extract_book_info_batch` 保持一致，接口层才能两套实现换着用。

    ## 为什么失败要返回异常对象，而不是「全 None 的 BookInfo」

    这是这个函数**故意**和「省事写法」不一样的地方。省事的写法是：
    模型漏抽了某条 → 那一项保持初始值（全 None）→ 继续。

    问题是：**「全 None」在业务上是有意义的结局** —— 它表示「这条帖子确实什么
    都没写」（比如"出一本书，有意私聊"）。如果漏抽也变成全 None，这两种情况就
    **完全无法区分**了：数据悄悄丢了，你只会看到一堆"信息残缺"的帖子，
    还以为模型抽得挺好。

    所以漏抽必须变成一个**能被发现**的东西 → 抛个异常对象进那个槽位。
    调用方（接口层）会把它变成结果里的 `error` 字段。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")

    cleaned = [(t or "").strip() for t in texts]

    # 空文本不进 LLM，直接给全 None 占位，省调用（同逐条版本）
    results: list[BookInfo | Exception] = [BookInfo() for _ in cleaned]
    pending = [i for i, t in enumerate(cleaned) if t]
    if not pending:
        return results

    # 切成若干个 chunk。每个 chunk 记住它对应的**全局下标**，
    # 这样结果回来时能精确落位。
    chunks: list[tuple[list[int], list[str]]] = []
    for start in range(0, len(pending), batch_size):
        idxs = pending[start : start + batch_size]
        chunks.append((idxs, [cleaned[i] for i in idxs]))

    messages = [_build_batch_messages(texts_) for _, texts_ in chunks]

    structured = _get_batch_structured_llm()
    config = (
        {"max_concurrency": max_concurrency} if max_concurrency is not None else None
    )
    # return_exceptions=True：一个 chunk 失败不该让整批 1 万条全丢
    raw = structured.batch(messages, config=config, return_exceptions=True)

    for (idxs, _chunk_texts), out in zip(chunks, raw):
        if isinstance(out, Exception):
            # 整个 chunk 挂了 → 这一批每条都记上同一个异常。
            # 注意：粒度的代价。批量把「单条失败」放大成「一批失败」，
            # 所以 batch_size 越大省得越多、但要重跑的也越多。20 是个折中。
            for i in idxs:
                results[i] = out
            continue

        # 按 idx 落位。idx 是 1-based、chunk 内局部编号
        got: dict[int, BookInfo] = {
            item.idx: item.info for item in out.items if 1 <= item.idx <= len(idxs)
        }
        for local, global_i in enumerate(idxs, start=1):
            if local in got:
                results[global_i] = got[local]
            else:
                # 模型漏抽/编号对不上 —— 显式报出来，别让它伪装成「信息残缺」
                results[global_i] = ValueError(
                    f"模型漏抽了这一条（本批 {len(idxs)} 条，它是第 {local} 条）"
                )

    return results


def _get_batch_structured_llm():
    """拿到「输出必为 _BatchOut」的 LLM 包装器，惰性初始化 + 复用。"""
    global _batch_structured_llm
    if _batch_structured_llm is None:
        llm = get_llm(temperature=0.0)
        # 同样必须显式写 function_calling，理由见 _get_structured_llm
        _batch_structured_llm = llm.with_structured_output(
            _BatchOut, method="function_calling"
        )
    return _batch_structured_llm
