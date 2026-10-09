"""BM25 词法检索 —— 混合检索里的「稀疏」那一半。

## 为什么向量检索之外还要 BM25

向量（稠密）擅长「意思相近」：搜「同济版高数」能命中「高等数学 第七版」。
但它有个可预测的盲区 —— **精确词面**：

    「汤家凤 1800」里的「1800」
    「第五版」vs「第六版」      ← BGE 把这两个映射得非常近（都是"某版"）
    「谢希仁」这种罕见人名
    ISBN、课程代码之类的编号

这些词在向量空间里没有足够区分度（模型没见过几次，或语义上太像），
但在词面上是**唯一标识**。BM25 恰好相反：它只看词有没有出现，
完全不懂语义 —— 所以能精确区分「第五版」和「第六版」。

    BM25 强的地方正是向量弱的地方，向量强的地方正是 BM25 弱的地方。
    —— 所以是**融合**，不是二选一。

## 坑 1：中文必须分词

BM25 按空格切词。中文句子没有空格，「高等数学第七版」会被当成**一个词**，
结果只有整句一字不差才算命中 —— 等于废掉。
所以先用 jieba 切成 ["高等","数学","第七版"]。

## 坑 2：索引从向量库重建，不单独持久化

向量库（Chroma）的 `documents` 字段存着文档原文，那是**唯一的真相**。
BM25 索引纯内存、进程重启就没了 —— 那就第一次用到时**从向量库拉回来重建**。

好处：两个索引的语料永远一致，不存在「改了一个忘了改另一个」。
代价：重启后首次查询多几百毫秒（只发生一次）。

## 坑 3：BM25 的分数**无上界**，不能照搬余弦那套阈值

余弦相似度天然落在 [0, 1]，所以能定「0.60 以上算命中」这种有物理意义的线。
BM25 的分数取决于词频和文档长度，理论上没有上限 —— 同一个 8.5 分，
在这份语料里可能是最高分，换一份语料可能只是中等。

所以 BM25 通道的闸门**不能用绝对分数**，用两个与分数无关的条件：
  ① `bm25_score > 0` —— 至少有一个查询词真的出现在这篇文档里
  ② **查询词覆盖率 ≥ `_MIN_COVERAGE`** —— 查询里多数的词都命中，才算真匹配

### 为什么非要有覆盖率这第二条（真实案例）

最难的负样本是「法学概论」：库里没有法学书，但有一本
《毛泽东思想和中国特色社会主义理论体系概论》—— 两者共享「概论」二字。

**这正是 BM25 的攻击面**：它只看词面，会认为这是一次命中。
但这是**字面重叠，不是语义相近**（向量那侧给的余弦分只有 0.5902，拦得住；
BM25 这侧只看"命中了一个词"，拦不住）。

覆盖率就是用来挡这个的：查询切出 ["法学","概论"] 两个词，
那本书只命中 1 个 → 覆盖率 0.5 < 0.6 → **挡掉**。
而「汤家凤1800」这类真查询，两个词都在同一本书里 → 覆盖率 1.0 → 放行。

### 分母只算「实词」（第二版修正，实测出来的）

覆盖率 = 命中的查询词数 / 查询词总数。分母是**查询词总数**，
所以里面混进没意义的词就会稀释覆盖率、制造**假阴性**：

    "考研数学用的" → jieba 切成 ['考研', '数学', '用', '的']
    一篇含「考研数学」的文档 → 2/4 = 0.5 → 被闸门误杀（它是对的！）

修正：算覆盖率前先剔掉**单字**（处理见 `_content_terms()`）。
剔掉后 {考研, 数学} → 2/2 = 1.0 → 正确放行；
而「法学概论」两个词都是双字，不受影响，照样挡得住。

**注意这里的选择**：BM25 通道「漏掉一个正例」是可接受的
（向量通道还在跑，会把它捞回来）；但「放进一个负例」不可接受
（它绕过了余弦闸门）。所以闸门宁严勿宽。

### 顺带记一笔 BM25 的死穴（面试可讲）

BM25 只认**字面完全相同**，连近义词都不认。实测：

    '同济'        → ['同济']
    '同济大学出版社' → ['同济大学', '出版社']     ← 「同济」不在里面

人一眼就知道这两句说的是同一个出版社，BM25 认为毫无关系。
所以「同济版高数」这种查询在词法通道是 0 命中 —— 无所谓，
向量通道本来就擅长这个，两个通道各补各的盲区。

（阈值 0.6 和余弦那个 0.60 一样，是**标定**出来的，见 tests/test_lexical.py。）
"""

import threading

import jieba
from rank_bm25 import BM25Okapi

from app.services import vector_store

# 查询词覆盖率下限。定 0.6 的依据见模块开头「坑 3」——
# 核心是它必须能挡住「法学概论 → …理论体系概论」这种**只共享一个词**的假命中
# （覆盖率 0.5），同时放行「汤家凤1800」这种**两个词都命中**的真匹配（1.0）。
_MIN_COVERAGE = 0.6

# 所有单例都是**进程级**的：BM25 索引建一次、所有请求共用。
_lock = threading.Lock()
_bm25: BM25Okapi | None = None
_ids: list[str] = []
_term_sets: list[set[str]] = []   # 每篇文档的「实词」集合，算覆盖率用（set 查找 O(1)）


def _tokenize(text: str) -> list[str]:
    """中文分词。顺带丢掉纯空白的 token。

    用 jieba 的**精确模式**（默认）而不是搜索引擎模式：
    后者会把长词再切碎（"中华人民共和国" → "中华/华人/人民/共和/共和国"），
    召回变宽的同时噪声也变大 —— 我们是短文本检索，不需要。
    """
    return [t for t in jieba.lcut(text) if t.strip()]


def _content_terms(tokens) -> set[str]:
    """从分词结果里挑出**实词**，用于覆盖率计算。

    ⚠️ 必须剔掉单字。中文单字虚词居多（的 / 用 / 了 / 和 / 是），
    留在分母里会把覆盖率稀释成**假阴性** —— 实测踩到的例子：

        "考研数学用的"  →  jieba 切成 ['考研', '数学', '用', '的']
        一篇含「考研数学」的文档 → 覆盖率算成 2/4 = 0.5 → 被误杀

    剔掉单字后：{考研, 数学} → 2/2 = 1.0 → 正确放行。

    全是单字时（比如只查「书」）退回用原 token，
    否则分母为 0、覆盖率无定义。
    """
    terms = {t for t in tokens if len(t) > 1}
    return terms or set(tokens)


def _build_from_store() -> None:
    """从向量库拉全部文档，重建 BM25 索引。调用方负责持锁。"""
    global _bm25, _ids, _term_sets

    ids, docs, _metas = vector_store.all_documents()
    tokenized = [_tokenize(d) for d in docs]

    _ids = ids
    _term_sets = [_content_terms(t) for t in tokenized]
    # BM25Okapi 在语料为空时会抛，所以显式判断
    _bm25 = BM25Okapi(tokenized) if tokenized else None


def ensure_built() -> None:
    """确保索引已建好。首次调用时重建，之后是空操作。

    双重检查加锁：和 embedding.get_model() 同一个理由 ——
    多个请求同时打进来时，不能各建一份（浪费且内存翻倍）。
    """
    if _bm25 is not None:
        return
    with _lock:
        if _bm25 is None:
            _build_from_store()


def rebuild() -> None:
    """强制重建。建索引接口写完向量库之后调它。"""
    with _lock:
        _build_from_store()


def search(query: str, top_k: int) -> list[dict]:
    """BM25 检索。

    返回按 BM25 分数降序的候选，每项：
        {"id": "42", "bm25": 12.34, "coverage": 1.0}

    **不透出原始 BM25 分数给外部做判断** —— 它无上界，没法定阈值。
    外部（search.py）只用这个**顺序**去和向量结果做 RRF 融合。
    """
    ensure_built()
    if _bm25 is None or not _ids:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    # ⚠️ 覆盖率用**实词**算，不是原始 token。
    # 原始 token 里的单字虚词（的/用/了）会把分母撑大、制造假阴性，
    # 具体例子见 _content_terms() 的 docstring。
    query_terms = _content_terms(query_tokens)
    if not query_terms:
        return []

    scores = _bm25.get_scores(query_tokens)

    rows: list[dict] = []
    for idx, (post_id, score) in enumerate(zip(_ids, scores)):
        # 闸门 ①：至少要有一个词真的出现在这篇文档里
        if score <= 0:
            continue
        # 闸门 ②：查询词覆盖率 —— 挡「只共享一个词的假命中」
        coverage = len(query_terms & _term_sets[idx]) / len(query_terms)
        if coverage < _MIN_COVERAGE:
            continue
        rows.append({"id": post_id, "bm25": float(score), "coverage": coverage})

    rows.sort(key=lambda r: r["bm25"], reverse=True)
    return rows[:top_k]
