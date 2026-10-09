"""向量库封装（Chroma）。

## 为什么用 Chroma

它是**嵌入式**库，跟 SQLite 一样跑在进程内 —— 不用起 Docker、不用起服务，
`pip install` 完就能用，数据存成一个目录。这个项目 200 多条数据，
上 Milvus / Qdrant 属于杀鸡用牛刀。

代价是它不适合多副本水平扩展（每个进程一个独立的库文件）。
生产环境会换成独立的向量数据库服务，但**接口形状是一样的**：
写进去 (id, 向量, 原文, 元数据)，查出来 (id, 距离, 原文, 元数据)。
所以这里把 Chroma 的细节都关在这个文件里，换库时只改这一个文件。

## 一个必须转换的东西：distance 不是 score

Chroma 查询返回的是 **距离**（越小越相似），不是相似度（越大越相似）。
cosine 空间下：`distance = 1 - cosine_similarity`，取值范围 [0, 2]。

如果直接把 distance 当分数用，排序会整个反过来；如果拿它跟 "0.5 以上算命中"
这种阈值比，也会全错。所以这个封装**统一把 distance 换算成 score 再往外给**，
外面看到的永远是「越大越相似」。

（这个细节是我实测出来的：往库里塞两个正交向量 [1,0] 和 [0,1]，
查 [1,0] 得到的 distance 是 [0.0, 1.0] —— 自己那条 0.0，正交那条 1.0，符合预期。）
"""

import chromadb

from app.config import settings

_client: chromadb.ClientAPI | None = None
_collection = None


def get_client() -> chromadb.ClientAPI:
    """拿到全局唯一的 Chroma 客户端。"""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client


def get_collection():
    """拿到全局唯一的 collection，第一次调用时创建。"""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=settings.chroma_collection,
            # 必须显式指定余弦空间。
            # 不指定的话 Chroma 默认用 L2（欧氏距离）—— 对归一化向量来说
            # 排序结果跟余弦其实等价，但 distance 的数值范围完全不同，
            # 阈值就没法定了。显式写出来，别依赖默认值。
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# 一次 upsert 最多写多少条。
#
# 为什么必须切块：Chroma 底层（Rust 绑定）有硬上限，超了直接抛
#     InternalError: Batch size of 9997 is greater than max batch size of 5461
# ——**这不是"数据太大性能不好"，是直接失败**。220 条语料时永远碰不到，
# 扩到 1 万条第一次建索引就撞上了。
#
# 取 2000 而不是贴着 5461：留一半余量。这个上限是 Chroma 内部定的，
# 不同版本会变（甚至跟 embedding 维度有关），贴着写等于把版本升级变成线上事故。
_UPSERT_CHUNK = 2000


def upsert(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """写入或覆盖一批向量。**参数长度可以任意大**（内部自动切块）。

    用 upsert 而不是 add：同一个 id 重复写会覆盖，不会重复插入。
    所以**重复调建索引接口是幂等的**，不会把库写成一堆重复条目。

    id 用帖子的数据库主键转成字符串 —— 这样向量库里的每条记录
    都能直接对回主服务里那一行。

    切块放在这一层，而不是让调用方自己切：能不能一次写 1 万条是
    **Chroma 的实现细节**，不该泄漏到 search.py 去（那里的代码是「把帖子写进去」，
    不该操心底下这批库的批次上限）。换向量库时也只需要改这个文件。
    """
    collection = get_collection()
    for start in range(0, len(ids), _UPSERT_CHUNK):
        end = start + _UPSERT_CHUNK
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def query(query_embedding: list[float], top_k: int) -> list[dict]:
    """向量检索，返回按相似度降序的命中列表。

    每项形如：
        {"id": "42", "score": 0.83, "text": "高等数学 ...", "metadata": {...}}

    score 已经换算成**余弦相似度**（越大越相似，理论上限 1.0）。
    """
    result = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma 的返回是「批量」形状：每个 key 都是 list[list[...]]，外层对应每个 query。
    # 我们只发了一个 query，所以取 [0]。
    ids = result["ids"][0]
    distances = result["distances"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    return [
        {
            "id": doc_id,
            "score": 1.0 - float(distance),  # distance -> 相似度，见模块开头说明
            "text": document,
            "metadata": metadata or {},
        }
        for doc_id, distance, document, metadata in zip(
            ids, distances, documents, metadatas
        )
    ]


def all_documents() -> tuple[list[str], list[str], list[dict]]:
    """取出库里全部 (id, 文档文本, 元数据)。

    给 BM25 建索引用 —— BM25 不单独持久化，而是**从向量库重建**，
    这样两个索引的语料永远是同一份，不会出现「改了一个忘了改另一个」。
    （见 app/services/lexical.py 开头的说明。）
    """
    result = get_collection().get(include=["documents", "metadatas"])
    return (
        result["ids"],
        [d or "" for d in result["documents"]],
        [m or {} for m in result["metadatas"]],
    )


def get_by_ids(ids: list[str]) -> list[dict]:
    """按 id 批量取回记录（含向量）。

    用途：BM25 通道召回的候选**绕过了向量检索**，手里没有它的余弦分。
    但对外输出需要一个统一的 score 字段，所以这里把它的向量捞回来现算 ——
    一次批量调用，比逐条查便宜得多。
    """
    if not ids:
        return []
    result = get_collection().get(
        ids=ids, include=["documents", "metadatas", "embeddings"]
    )
    return [
        {
            "id": doc_id,
            "text": doc or "",
            "metadata": meta or {},
            "embedding": [float(x) for x in emb],
        }
        for doc_id, doc, meta, emb in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
            result["embeddings"],
        )
    ]


def count() -> int:
    """库里有几条。"""
    return get_collection().count()


def reset() -> None:
    """删掉整个 collection 再重建 —— 重建索引时用，保证不留旧数据的残渣。"""
    global _collection
    try:
        get_client().delete_collection(settings.chroma_collection)
    except Exception:
        # 不存在就没什么可删的，正常继续
        pass
    _collection = None
    get_collection()
