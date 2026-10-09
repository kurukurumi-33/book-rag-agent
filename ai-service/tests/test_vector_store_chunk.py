"""`vector_store.upsert` 的分块写入。

起因是一个**真炸过的 bug**：语料扩到 1 万条后第一次建索引，
`POST /index/build` 直接回 500 ——

    chromadb.errors.InternalError:
    ValueError: Batch size of 9997 is greater than max batch size of 5461

Chroma 底层（Rust 绑定）单批写不下 5461 条以上。220 条语料时永远碰不到，
所以这个坑一直藏着，直到语料涨了 45 倍才露出来。

这里不测 Chroma 本身，测的是**我们这一层有没有把大批次切开**：
参数长度任意大时，实际发出去的每一批都不能超过上限，且**一条都不能丢**。
"""

import pytest

from app.services import vector_store


class _FakeCollection:
    """假 collection：只记下每次 upsert 收到的条数，不真写库。"""

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.received_ids: list[str] = []

    def upsert(self, ids, embeddings, documents, metadatas):
        # 断言四个参数是**等长**的 —— 切块时最容易切歪的地方就是漏切其中一个，
        # 那样 Chroma 会报「长度不一致」，但如果是我们自己造的数据就容易看花眼。
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        self.batch_sizes.append(len(ids))
        self.received_ids.extend(ids)


@pytest.fixture
def fake_collection(monkeypatch):
    fake = _FakeCollection()
    monkeypatch.setattr(vector_store, "get_collection", lambda: fake)
    return fake


def _payload(n: int):
    return (
        [str(i) for i in range(n)],
        [[0.1, 0.2] for _ in range(n)],
        [f"文档{i}" for i in range(n)],
        [{"post_id": i} for i in range(n)],
    )


def test_超过上限会自动切块(fake_collection):
    """9997 条 —— 就是线上那次实际炸掉的数字。"""
    ids, emb, docs, metas = _payload(9997)
    vector_store.upsert(ids, emb, docs, metas)

    assert len(fake_collection.batch_sizes) > 1, "9997 条必须被切开，不能一批发出去"
    assert max(fake_collection.batch_sizes) <= vector_store._UPSERT_CHUNK


def test_切块不丢也不重(fake_collection):
    """切完以后收到的 id 必须和传进去的**一模一样**（顺序也要一致）。

    这是分块最经典的 bug：`range(0, n, size)` 写对了但切片写成 `ids[start:size]`
    （第二个参数是**结束下标**不是长度），结果每批都从 start 切到固定的 size，
    前面的数据被反复写、后面的永远写不到 —— 而且不会报错。
    """
    n = 5000
    ids, emb, docs, metas = _payload(n)
    vector_store.upsert(ids, emb, docs, metas)

    assert fake_collection.received_ids == ids


def test_刚好等于上限时只发一批(fake_collection):
    size = vector_store._UPSERT_CHUNK
    ids, emb, docs, metas = _payload(size)
    vector_store.upsert(ids, emb, docs, metas)

    assert fake_collection.batch_sizes == [size]


def test_空列表不发请求(fake_collection):
    """零条不该产生一次空 upsert —— 某些版本会对空批直接抛错。"""
    vector_store.upsert([], [], [], [])
    assert fake_collection.batch_sizes == []


def test_小批量仍然是一批(fake_collection):
    """别为了修 1 万条把 220 条的小场景改成多次往返。"""
    ids, emb, docs, metas = _payload(220)
    vector_store.upsert(ids, emb, docs, metas)
    assert fake_collection.batch_sizes == [220]
