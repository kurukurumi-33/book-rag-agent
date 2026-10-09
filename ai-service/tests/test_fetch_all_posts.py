"""`book_service.fetch_all_posts` 的翻页逻辑。

为什么要单独测它：这个函数是**为了修一个静默 bug** 才存在的。
主服务把 `limit` 夹在 500，之前 `build_index` 直接调 `fetch_posts(limit=...)`，
语料涨到 1 万条以后只会索引到最新的 500 条 —— **接口照样返回 200，
`indexed=500` 看着完全正常**。只有哪天发现「大半的书搜不出来」才会怀疑到这儿。

所以这里不测「能翻页」，专门测那些**翻错了还不报错**的情况：
翻页没前进、offset 被忽略、最后一页刚好是整页。
"""

import pytest

from app.clients import book_service


def _pages(total: int, page_size: int, *, honors_offset: bool = True):
    """假的主服务：按 id 倒序，每页最多 page_size 条。

    honors_offset=False 用来模拟「主服务回退到旧版本、不认识 offset 参数」——
    它在真实世界里会表现为：每页都返回同样的前 page_size 条。
    """
    rows = [{"id": i, "rawText": f"帖子{i}", "extractStatus": "DONE"}
            for i in range(total, 0, -1)]  # id 倒序，跟主服务一致

    calls = []

    def fake(extract_status=None, limit=500, offset=0):
        calls.append(offset)
        start = offset if honors_offset else 0
        return rows[start : start + limit]

    fake.calls = calls
    return fake


def test_翻页能拿全量(monkeypatch):
    monkeypatch.setattr(book_service, "fetch_posts", _pages(1200, 500))
    posts = book_service.fetch_all_posts(extract_status="DONE", page_size=500)
    assert len(posts) == 1200
    assert [p["id"] for p in posts] == list(range(1200, 0, -1))


def test_最后一页刚好是整页也不会少(monkeypatch):
    """1000 条 / 每页 500 → 翻两页正好取满。

    这是最容易写错的地方：如果循环条件写成「拿到空页才停」，
    就会多打一次接口；如果写成「按总数算页数」，除数一变就错。
    这里靠「短页即止」，第二页正好整页、不是短页，所以还得再请求一次才能确认到头 ——
    这次请求返回空列表，不误伤数据。
    """
    fake = _pages(1000, 500)
    monkeypatch.setattr(book_service, "fetch_posts", fake)
    posts = book_service.fetch_all_posts(extract_status="DONE", page_size=500)
    assert len(posts) == 1000
    assert fake.calls == [0, 500, 1000]


def test_offset_被忽略时不死循环(monkeypatch):
    """主服务不认识 offset 的话，每页都返回同样的 500 条。

    「短页即止」在这里永远不会触发（每页都是满的），所以必须有第二道保险：
    整页全是重复 → 立刻停。不然就是个无声的死循环，而且每轮都在打 HTTP。
    """
    monkeypatch.setattr(book_service, "fetch_posts", _pages(5000, 500, honors_offset=False))
    posts = book_service.fetch_all_posts(extract_status="DONE", page_size=500)
    assert len(posts) == 500  # 拿到的就是同一批，去重后只剩 500
    assert [p["id"] for p in posts] == list(range(5000, 4500, -1))


def test_limit_限制总量且不多拉(monkeypatch):
    """limit=700 → 第一页 500，第二页只要 200，不去拉满 500。"""
    fake = _pages(5000, 500)
    monkeypatch.setattr(book_service, "fetch_posts", fake)
    posts = book_service.fetch_all_posts(extract_status="DONE", limit=700, page_size=500)
    assert len(posts) == 700
    assert fake.calls == [0, 500]


def test_空库直接返回空(monkeypatch):
    monkeypatch.setattr(book_service, "fetch_posts", lambda **kw: [])
    assert book_service.fetch_all_posts() == []


def test_extractStatus_透传(monkeypatch):
    seen = {}

    def fake(extract_status=None, limit=500, offset=0):
        seen["status"] = extract_status
        return []

    monkeypatch.setattr(book_service, "fetch_posts", fake)
    book_service.fetch_all_posts(extract_status="PENDING")
    assert seen["status"] == "PENDING"


def test_单页接口会把_offset_带上(monkeypatch):
    """锁住 fetch_posts 真的把 offset 发出去了。

    漏传的话表现和「服务端忽略 offset」一模一样（每页都是同一批），
    但原因完全不同 —— 这个测试让两种情况的锅甩不到彼此头上。
    """
    seen = {}

    def fake_request(method, path, *, params=None, json_body=None):
        seen.update(params or {})
        return []

    monkeypatch.setattr(book_service, "_request", fake_request)
    book_service.fetch_posts(extract_status="DONE", limit=500, offset=1000)
    assert seen["offset"] == 1000
    assert seen["limit"] == 500
    assert seen["extractStatus"] == "DONE"
