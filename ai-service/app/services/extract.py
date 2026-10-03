"""书籍信息抽取：混乱的帖子文本 -> 结构化字段。

思路：不给模型写规则解析器，而是把「怎么抽」写成 prompt，
让模型输出一个符合 BookInfo 的 JSON，再交给 pydantic 校验。

调用链只有三步：
    get_llm()                          -> DeepSeek 客户端
    llm.with_structured_output(...)    -> 包一层，保证输出能转成 BookInfo
    .invoke([system, human])           -> 真正发请求，拿回 BookInfo 对象

调 prompt 用：.venv/Scripts/python.exe scripts/try_extract.py
"""

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
