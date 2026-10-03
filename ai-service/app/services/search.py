"""语义检索链路（M2）。

RAG 里的那个 R（Retrieval）。整条链路：

    建索引：主服务 API 取 DONE 的帖子
            → compose_text() 拼成检索文本      ← 你写
            → BGE 向量化
            → 存进 Chroma（id = 帖子主键）

    查询：  用户 query
            → BGE 向量化（同样的模型，必须）
            → Chroma 算余弦相似度，取 top_k
            → filter_hits() 过滤掉不够像的      ← 你写
            → 返回

**为什么能做「同济版高数」匹配到「高等数学 同济大学出版社」**：
两个句子没有任何一个词完全相同，但 BGE 把它们映射到向量空间里相近的位置。
这是关键词搜索（`LIKE '%高数%'`）做不到的事 —— 也是这个项目相对 CRUD 的价值所在。
"""

import time

from app.clients import book_service
from app.config import settings
from app.services import embedding, vector_store

# BGE 模型名，写进元数据方便以后排查「这批向量是哪个模型生成的」。
# 换模型必须重建索引 —— 不同模型的向量空间没有任何可换算关系。
_EMBEDDING_MODEL_KEY = "embedding_model"
#阈值定 0.60 的依据：scripts/eval_runs/阈值0.60.json —— 16 条用例（12 正 / 4 反）全过。
#为什么不留在 0.58：难反例「法学概论」原始分 0.5902 会漏进来。它和
#  「毛泽东思想和中国特色社会主义理论体系概论」共享「概论」二字 —— 这是**字面重叠**，
#  不是语义相近。反例挑得不狠的时候，阈值看着很安全（见面试素材第 14 条）。
#余量：最高负例 0.5902，最低正例 0.6095（「我想找本线代」）→ 缝只有 0.019。
#  缝薄是测试集还太弱的症状，不代表语料稳定，别把这个数当安全带。
#⚠️ 这个常量只兜底 search() 的 min_score=None。eval_search.py 的 --min-score 会覆盖它——
#  改这里之后，评测要用不带 --min-score 的跑法验证，否则测的不是线上行为。
#更稳的做法可能是与 top-1 的相对阈值（还没做）。
_DEFAULT_MIN_SCORE = 0.60


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


def build_index(limit: int = 500, rebuild: bool = False) -> dict:
    """从主服务拉 DONE 的帖子，向量化后写进 Chroma。

    :param limit: 最多拉多少条（主服务会截断到 500）
    :param rebuild: True 时先清空整个 collection 再写。
                    平时用 False —— upsert 是幂等的，重跑不会产生重复。

    返回本次执行的统计，给接口直接当响应体。
    """
    started = time.perf_counter()

    if rebuild:
        vector_store.reset()

    posts = book_service.fetch_posts(extract_status="DONE", limit=limit)

    # 抽不出书名的帖子跳过：拼出来的文本只有成色和价格，
    # 跟「这是哪本书」完全无关，放进向量库只会变成噪音、抢走真实命中。
    usable = [p for p in posts if p.get("bookName")]
    skipped = len(posts) - len(usable)

    if not usable:
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

    return {
        "fetched": len(posts),
        "indexed": len(usable),
        "skipped": skipped,
        "collection_count": vector_store.count(),
        "dim": embedding.dim(),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def search(query: str, top_k: int = 10, min_score: float | None = None) -> list[dict]:
    """语义检索。

    :param query: 用户的自然语言查询，如「同济版高数」
    :param top_k: 最多返回几条
    :param min_score: 相似度下限，低于它的丢掉

    返回按 score 降序的命中列表，每项：
        {"id": "42", "score": 0.83, "text": "...", "metadata": {...}}
    """
    query_vector = embedding.embed_query(query)
    hits = vector_store.query(query_vector, top_k=top_k)
    if min_score is None:
        min_score = _DEFAULT_MIN_SCORE
    results = []
    for v in hits:
        score=v["score"]
        if score <min_score:
            continue
        results.append(v)
    return results

