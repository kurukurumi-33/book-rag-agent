"""混合检索链路 —— RAG 里的那个 R（Retrieval）。

## 整条链路

    建索引：主服务 API 取 DONE 的帖子
            → compose_text() 拼成检索文本
            → BGE 向量化 → 存进 Chroma（id = 帖子主键）
            → 顺带用同一批文本重建 BM25 索引（见 lexical.py）

    查询：  用户 query
            ├─ 通道① 稠密：BGE 向量化 → Chroma 余弦 → 过余弦阈值
            ├─ 通道② 稀疏：jieba 分词 → BM25 → 过覆盖率阈值
            ↓
            RRF 倒数排名融合（两个通道的**排名**相加）        ← 粗排到此为止
            ↓
            cross-encoder 逐对精排（只对前 rerank_pool 条）   ← 见 rerank.py
            ↓
            取 top_k

**粗排/精排的分工**（2026-10-09 加的，起因见下面「已知拦不住」）：
    粗排（向量+BM25）负责**召回**：从 1 万条里捞出 20 条候选，保证「不漏」。
    精排（cross-encoder）负责**排序**：给这 20 条重新打分，保证「顺序对」。
rerank 只在候选池内重排，救不了粗排没捞到的书 —— 所以粗排的阈值是**召回率**
的旋钮，别为了「更准」把它调严，那是精排的活。

## 为什么两个通道都要（一句话版）

    向量强在「意思相近」，弱在**精确词面**（版本号 / 编号 / 罕见人名 / ISBN）；
    BM25 正相反：只认字面，但字面认得极准。

    实测的反面证据（见 tests/test_lexical.py）：
        '同济'  vs  '同济大学出版社' —— jieba 切成 ['同济'] 和 ['同济大学','出版社']
        BM25 认为毫无关系，向量一眼看出是一回事。

    「同济版高数」这种查询 BM25 是 0 命中，得靠向量；
    「汤家凤1800」这种查询向量分辨不出编号，得靠 BM25。
    **各补各的盲区，所以是融合，不是二选一。**

**为什么能匹配到「高等数学 同济大学出版社」**：两个句子没有任何一个词
完全相同，但 BGE 把它们映射到向量空间里相近的位置。这是关键词搜索
（`LIKE '%高数%'`）做不到的事 —— 也是这个项目相对 CRUD 的价值所在。

## 坑：两套闸门，各管各的

⚠️ 这是这段代码**最容易写错**的地方。两个通道的分数**不可比**：

    余弦相似度  → 天然落在 [0, 1]，能定出「0.60 以上算命中」这种有物理意义的线
    BM25 分数   → 取决于词频和文档长度，**无上界**，同一个 8.5 分换份语料含义就变了

所以**不能**把两者算出来的分丢进同一个数组再排 —— 那是拿摄氏度和华氏度混着比大小。

正确做法是**各过各的闸门，然后只融合排名**：

    通道①  余弦 < min_score 的，直接不进池子
    通道②  BM25 分数 > 0 **且** 查询词覆盖率 ≥ 0.6，才进池子（逻辑在 lexical.py）
    融合    只看两个池子里各自的**第几名**，不看分数（RRF 就是为了消掉量纲）

为什么不给融合结果再套一次余弦闸门？—— 套了 BM25 就彻底白做了。
BM25 能捞回来的东西，恰恰就是**余弦不够高**的那些（"1800" 这种编号 BGE
根本不认识）。再套一次余弦闸门，等于把它刚捞上来的又扔回去。

那负样本怎么办？—— **靠两个闸门都拦不住，这是双塔方案的原理性上限。**

实测（9997 条语料）：「核工程」余弦 0.6034 会返回《Java核心技术》，
「环境工程」0.6199 会返回《Linux环境编程》，「教育学」0.6158 会返回《数学》。
而正例的下界「同济高数」只有 0.6187。**分数线重叠了**，
任何单一阈值都救不了：想拦住它们就得抬到 0.62 以上，而 0.62 一过正例就开始死
（见下面 _DEFAULT_MIN_SCORE 的扫描表）。

根因是双塔结构的固有缺陷：query 和文档各自压成一个向量，夹角只反映「整体语义方向」，
分不出「这个词是不是真的指同一件事」。

**修法不是调参，是加一个打分器：rerank（cross-encoder），见 rerank.py。**
它把 query 和文档拼在一起送进模型逐对判分，能看见字面的交叉。

⚠️ 但这里有个**实测出来的、和直觉相反的结论**，别记错了：
**rerank 上线后，上面那 8 条越界查询仍然只拦下 1 条 —— 它没有「修好」这张表。**
cross-encoder 的收益在**正例那一侧**：正例 top-1 命中率 15/16 → **16/16**。
也就是说 **rerank 改善的是排序，不是过滤**。原因和分两类的详细拆解见
`scripts/eval_search.py` 的 `KNOWN_LEAKS` 注释块（那是本项目最值得读的一段）。
覆盖率闸门仍然要留：它是**通道②内部的**闸门，挡的是「BM25 命中了一个虚词就放行」，
跟上面这个跨通道的问题不是一回事。
"""

import math
import time

from app.clients import book_service
from app.config import settings
from app.services import embedding, lexical, rerank, vector_store

# BGE 模型名，写进元数据方便以后排查「这批向量是哪个模型生成的」。
# 换模型必须重建索引 —— 不同模型的向量空间没有任何可换算关系。
_EMBEDDING_MODEL_KEY = "embedding_model"

# ── RRF（Reciprocal Rank Fusion，倒数排名融合）────────────────────────────
# 公式：每篇文档的融合分 = Σ 1 / (K + 它在各通道里的名次)
#   K=60 是原论文（Cormack et al. 2009）的经验值，作用是**压低靠后名次的权重**：
#   第 1 名 1/61≈0.0164，第 10 名 1/70≈0.0143 —— 差距被压平，不会让
#   某一路的偶然高分独裁结果。
#
# 为什么用「排名」而不是「分数」：见上面「两套闸门」那段 —— BM25 分数无上界，
# 和余弦放一起算加权和，权重根本没法定。换成排名，两边的量纲就都是「第几名」了。
_RRF_K = 60

# 每个通道召回多少条**候选**（不是最终返回多少）。
# 要比 top_k 大：融合的价值就在于「A 通道排第 8、B 通道排第 2」这种交叉，
# 池子刚好等于 top_k 的话，两边看到的是同一批，融合退化成排序抖动。
_RECALL_N = 20
# ── 阈值怎么定的：在 1 万条语料上扫出来的（2026-10-09）─────────────────────
#
#   阈值    正例 top-3    负例     拦住「越界查询」
#   0.58    16/16        4/5      0/8     ← 土木工程漏进来
#   0.60    16/16        5/5      0/8
#   0.61    16/16        5/5      1/8     ← 取这个
#   0.62    15/16        5/5      3/8     ← 「同济高数」挂了
#   0.63    14/16        5/5      4/8
#   0.65    12/16        5/5      4/8
#
# 结论：**0.61 是「正例全活」的上限**。再高一点（0.62）正例就开始死，
# 而负例拦下的数量还在涨 —— 也就是说这是一个**此消彼长**的旋钮，
# 没有哪个值能两边都满意。挑 0.61 是因为它在保住 16/16 的前提下多拦了一条。
#
# ⚠️ 但别把这个数当安全带：最低正例「同济高数」0.6187，**能拦住的最高**越界查询
#    「核工程」0.6034 —— 缝 0.0153；而拦不住的那批（教育学 0.6158 / 环境工程 0.6199 /
#    国际关系 0.6233 / 纺织 0.6959）**全都落在正例区间里**，缝其实已经是负的。
#    也就是说这不是「找到了一个好阈值」，是「在一堆分不开的样本里尽量多拦一条」。
#    根因见下面「已知拦不住」。
#
# 上一版（220 条语料）定的是 0.60，依据是 scripts/eval_runs/阈值0.60.json。
# **那份快照对现在的语料已经完全作废** —— 语料从 37 个书种涨到 2869 个，
# 分数分布整个变了（例：「法学概论」的 top-1 从 0.5902 涨到 0.7004）。
#
# ⚠️ 这个常量只兜底 search() 的 min_score=None。eval_search.py 的 --min-score 会覆盖它——
#  改这里之后，评测要用**不带** --min-score 的跑法验证，否则测的不是线上行为。
_DEFAULT_MIN_SCORE = 0.61


def compose_text(post: dict) -> str:
    """把一条帖子拼成「用于向量化的检索文本」。

    输入 post 是主服务 `GET /api/posts` 返回的一条记录（dict），可用字段：

        id            42
        rawText       "高数同济七版 上下册一起 40 有笔记 划重点了 可小刀"   ← 卖家原文
        bookName      "高等数学"        （可能为 None）
        edition       "第七版"          （可能为 None）
        author        None              （可能为 None）
        publisher     "同济大学出版社"    （可能为 None）
        conditionDesc "有笔记，划过重点"   （可能为 None）
        price         40.0              （可能为 None）
        hasNotes      True              （可能为 None）

    目标：让「同济版高数」这种查询能排到这条记录前面。"""



    #② docs/需求与接口.md §5.1 的参考拼法。
    #书名 + 版次 + 作者 + 出版社 。不拼价格、不拼原文。
    #拼上成色会使得搜索结果出错，让不同书名但成色匹配的书排名更靠前，与设计理念相悖
    #rawText不拼，拼上了会使得书名以外的字段把向量排名提高（分数变得不干净），使得查询出现其他书名
    #价格更多体现的是书的状态而不是身份，将它拼上去会影响筛选，同时也不能精确过滤，所以不拼会更好
    #复制改变的是方向，不是幅度。 自注意力会让重复的 token 互相 attend，输出的向量方向会偏
    # 方向一偏，和查询的夹角就变了。所以效果是「书名的意思被微调了」，不是书名更重要了
    #排序会对分数产生细微的影响，但实测里没有使得排名出现变化，故不做排序调整
    #None要直接跳过，绝对不能占位，占位会毒化检索
    fields = [
        post.get("bookName"),
        post.get("edition"),
        post.get("author"),
        post.get("publisher"),
    ]
    return " ".join(x for x in fields if x)


def build_index(limit: int = 20000, rebuild: bool = False) -> dict:
    """从主服务拉 DONE 的帖子，向量化后写进 Chroma。

    :param limit: 最多建多少条。默认 20000 —— 当作「全量」用，
                  比这条线还多的语料得显式调大。
    :param rebuild: True 时先清空整个 collection 再写。
                    平时用 False —— upsert 是幂等的，重跑不会产生重复。

    ⚠️ 走的是 `fetch_all_posts`（翻页），不是 `fetch_posts`。
    主服务单次最多给 500 条，用 fetch_posts 的话 1 万条语料只会索引到最新的 500 条 ——
    **而且不报错**，接口返回的 indexed=500 看着完全正常，只有检索时才发现大半书搜不到。

    返回本次执行的统计，给接口直接当响应体。
    """
    started = time.perf_counter()

    if rebuild:
        vector_store.reset()

    posts = book_service.fetch_all_posts(extract_status="DONE", limit=limit)

    # 抽不出书名的帖子跳过：拼出来的文本只有成色和价格，
    # 跟「这是哪本书」完全无关，放进向量库只会变成噪音、抢走真实命中。
    usable = [p for p in posts if p.get("bookName")]
    skipped = len(posts) - len(usable)

    if not usable:
        # ⚠️ 就算没东西可写也要重建 BM25：rebuild=True 上面刚把向量库清空了，
        #    不跟着清的话，BM25 索引里还留着上一批文档 —— 两个索引的语料就对不上了。
        lexical.rebuild()
        return {
            "fetched": len(posts),
            "indexed": 0,
            "skipped": skipped,
            "collection_count": vector_store.count(),
            "dim": embedding.dim(),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "note": "主服务没有返回可建索引的 DONE 帖子（bookName 全为空？）",
        }

    texts = [compose_text(p) for p in usable]
    vectors = embedding.embed_documents(texts)

    vector_store.upsert(
        ids=[str(p["id"]) for p in usable],
        embeddings=vectors,
        documents=texts,
        metadatas=[
            {
                # 元数据只能存标量（str/int/float/bool），不能存 dict / list。
                # None 也不允许 —— Chroma 会报错，所以统一转成空串。
                # 这几个字段是「列表页直接能展示」的，避免为了显示一行结果
                # 再去主服务逐条查详情（N+1 次 HTTP）。
                "post_id": p["id"],
                "book_name": p.get("bookName") or "",
                "author": p.get("author") or "",
                "publisher": p.get("publisher") or "",
                "edition": p.get("edition") or "",
                "price": p.get("price") if p.get("price") is not None else -1.0,
                "has_notes": bool(p.get("hasNotes")),
                _EMBEDDING_MODEL_KEY: settings.embedding_model,
            }
            for p in usable
        ],
    )

    # BM25 索引不单独持久化，**从向量库重建**（理由见 lexical.py 坑 2）。
    # 放在 upsert 之后：这时候向量库才是这一批文档的最终状态。
    # 两者永远同源，不存在「改了一个忘了改另一个」。
    lexical.rebuild()

    return {
        "fetched": len(posts),
        "indexed": len(usable),
        "skipped": skipped,
        "collection_count": vector_store.count(),
        "dim": embedding.dim(),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度。

    需要它是因为 BM25 通道捞回来的候选**绕过了向量检索**，手里没有余弦分。
    但对外输出的 `score` 字段承诺是「余弦相似度」，得给它补上，
    否则同一个响应里两种含义的 score 混在一起，调用方没法用。

    （顺手归一化再点乘，不假设输入已经 L2 归一化 —— BGE 默认 encode 出来
    就没归一化，假设它归一化会得到一个偏大的假分数。）
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _rrf_fuse(ranked_lists: list[list[str]]) -> dict[str, float]:
    """把多个「按优劣排好序的 id 列表」融合成一个 {id: 融合分}。

    每个通道各贡献 1/(K + 名次)，同一篇文档被多个通道命中就累加 ——
    **所以「两个通道都排前面」的文档会明显胜出**，这正是混合检索想要的效果。
    """
    fused: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
    return fused


def search(
    query: str,
    top_k: int = 10,
    min_score: float | None = None,
    rerank_enabled: bool | None = None,
) -> list[dict]:
    """混合检索（向量 + BM25 → RRF 融合 → cross-encoder 精排）。

    :param query: 用户的自然语言查询，如「同济版高数」
    :param top_k: 最多返回几条
    :param min_score: **稠密通道**的余弦下限。⚠️ 它现在只管向量那一路，
                      BM25 那一路由 lexical.py 的覆盖率闸门负责 —— 两个通道的
                      分数不可比，没法用同一个数管（详见模块开头「两套闸门」）。
    :param rerank_enabled: 是否走精排。None = 用配置里的默认值（settings.rerank_enabled）。
                     显式传 False 能拿到「纯召回」的结果 —— eval_search.py 靠它做 A/B，
                     不然「rerank 到底有没有净增益」就只能靠嘴说。

    返回的列表每项：
        {"id": "42", "score": 0.83, "rrf_score": 0.0323,
         "rerank_score": 0.97, "text": "...", "metadata": {...}}

        score        —— 余弦相似度，给人看的（BM25 捞回来的现算，可能低于 min_score）
        rrf_score    —— 粗排的融合分，**数值大小本身没有意义**，只看相对顺序
        rerank_score —— 精排分（sigmoid 到 0~1）。**只在开了 rerank 时才有这个键**，
                        且它和 score 不可比（两个模型的输出，别混着排序）
    """
    if min_score is None:
        min_score = _DEFAULT_MIN_SCORE

    # 召回池比要返回的多 —— 理由见 _RECALL_N 的注释。
    # 但也不能少于 top_k，否则用户要 50 条却只拿到 20 条。
    recall_n = max(top_k, _RECALL_N)

    query_vector = embedding.embed_query(query)

    # ── 通道① 稠密：过余弦闸门 ──────────────────────────────────────────
    dense_kept = [
        h for h in vector_store.query(query_vector, top_k=recall_n)
        if h["score"] >= min_score
    ]

    # ── 通道② 稀疏：闸门在 lexical.search 内部（分数>0 且覆盖率≥0.6）────
    lexical_hits = lexical.search(query, top_k=recall_n)

    # ── 融合 ────────────────────────────────────────────────────────────
    fused = _rrf_fuse([[h["id"] for h in dense_kept], [h["id"] for h in lexical_hits]])
    if not fused:
        return []

    # 稠密通道的候选自带 text/metadata，直接用；BM25 独有的那些要补一次。
    by_id = {h["id"]: dict(h) for h in dense_kept}
    missing = [i for i in fused if i not in by_id]
    for row in vector_store.get_by_ids(missing):
        by_id[row["id"]] = {
            "id": row["id"],
            "score": _cosine(query_vector, row["embedding"]),
            "text": row["text"],
            "metadata": row["metadata"],
        }

    ordered = sorted(fused, key=fused.get, reverse=True)
    candidates = []
    for doc_id in ordered:
        hit = by_id.get(doc_id)
        if hit is None:
            # BM25 索引是**快照**，向量库是**活的**，两者之间有窗口：
            # 期间有条目被删（或刚 rebuild 过）就会出现「BM25 认得它、
            # 向量库查不到」。跳过去就行 —— 一条脏数据不该让整个 /search 抛 500。
            continue
        hit["rrf_score"] = fused[doc_id]
        candidates.append(hit)

    # 粗排到此为止。⚠️ 这里**不再截到 top_k** —— 截了的话精排能看到的池子
    # 就等于要返回的条数，「重排」退化成「原地不动」，精排彻底白做。
    # 池子大小由 rerank_pool 控制，不是 top_k。
    if rerank_enabled is None:
        rerank_enabled = settings.rerank_enabled
    if not rerank_enabled:
        return candidates[:top_k]

    # 精排只重排、不过滤（rerank_min_score 默认 None）—— 为什么这么定，
    # 见 config.py 里 rerank_min_score 的注释（一句话：正例那一侧只有 0.03 宽，切不起）。
    return rerank.rerank(query, candidates, top_k, min_score=settings.rerank_min_score)

