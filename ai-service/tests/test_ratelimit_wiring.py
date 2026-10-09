"""限流**接进接口以后**的行为（`-m integration`，要真 Redis）。

## 为什么单独一层

`test_ratelimit.py` 测的是「算法对不对」，`test_redis_integration.py` 测的是
「Lua 在真 Redis 上对不对」。但这两层都**没有回答一个更基本的问题**：

    这个限流器，真的挂在 /extract 上了吗？

漏挂的后果特别隐蔽 —— 单测全绿、代码看起来也全对，只是闸门根本没接上，
**而「没被限流」这件事本身不会报错**。所以要有一组用例从 HTTP 层打进去。

## 用 TestClient，但不跑 lifespan

`TestClient(app)` 不用 `with` 的话不会触发 lifespan —— 正好，
lifespan 里要加载 embedding（13 秒）和 rerank（1.1GB），
这一层根本不测模型，没必要等。

⚠️ 抽取函数是**打桩**的（monkeypatch 掉 `extract_book_info`）：
这些用例要验证的是「限流在 LLM 之前生效」，不是为了抽帖子。
而且这样跑**不花一分钱** —— 被限掉的请求连 LLM 都不会碰。
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import redis_client

pytestmark = pytest.mark.integration


@pytest.fixture
def client_and_app():
    """TestClient + **清干净限流桶**。

    ⚠️ 这个清理不是洁癖，是必须的：限流计数存在 Redis 里（**进程外**），
       TestClient 的客户端 IP 永远叫 "testclient"，所以**所有用例共用同一个桶**。
       不清的话，前一个用例把桶用满，后一个用例的第一个请求就已经是 429 ——
       表现成「限流接错了」，实际是测试之间互相污染。
       （这正是「把状态挪到 Redis」的副作用第一次咬到我：测试不再天然隔离。）
    """
    client = redis_client.get_redis()
    if client is None:
        pytest.skip("本机 Redis 没起 —— 跳过，不算失败")

    for key in client.scan_iter(match=redis_client.key("rl", "*"), count=100):
        client.delete(key)

    return TestClient(main.app)


@pytest.fixture
def free_extract(monkeypatch):
    """把抽取打桩，避免真调 LLM。"""
    from app.schemas import BookInfo

    monkeypatch.setattr(
        main, "extract_book_info", lambda raw: BookInfo(book_name="高等数学")
    )


def test_超过限额返回429并且带RetryAfter(client_and_app, free_extract, monkeypatch):
    """⚠️ 这条是这个文件里最重要的一个断言：**闸门真的接上了**。"""
    monkeypatch.setattr(main.settings, "ratelimit_extract_per_min", 2)

    codes = []
    last = None
    for _ in range(3):
        last = client_and_app.post("/extract", json={"raw_text": "出高数一本"})
        codes.append(last.status_code)

    assert codes[:2] == [200, 200], f"前两次该放行，实际 {codes}"
    assert codes[2] == 429, f"第三次该被限流，实际 {codes}"

    # Retry-After 必须是个正整数 —— 给 0 等于让客户端立刻重试
    retry_after = last.headers.get("Retry-After")
    assert retry_after is not None, "429 必须带 Retry-After"
    assert int(retry_after) > 0
    assert last.headers.get("X-RateLimit-Limit") == "2"


def test_被限流时不会走到LLM(client_and_app, monkeypatch):
    """这是限流的**全部意义**：把请求挡在花钱那一步之前。

    如果限流写在调用 LLM 之后，那它只是「事后记一笔」—— 钱已经花了。
    所以这里数的是**打桩函数被调用了几次**，不是状态码。
    """
    from app.schemas import BookInfo

    calls = []
    monkeypatch.setattr(main, "extract_book_info", lambda raw: calls.append(raw) or BookInfo(book_name="高等数学"))
    monkeypatch.setattr(main.settings, "ratelimit_extract_per_min", 1)

    assert client_and_app.post("/extract", json={"raw_text": "a"}).status_code == 200
    assert len(calls) == 1, "第一次该正常抽取"

    assert client_and_app.post("/extract", json={"raw_text": "b"}).status_code == 429
    assert len(calls) == 1, "⚠️ 被限流的请求**一次 LLM 都不该调**"


def test_extract和extract_batch共用一个桶(client_and_app, monkeypatch):
    """⚠️ 分开算的话，「单条打满 + 批量打满」就是两倍上限。

    它们烧的是同一份额度，就该共用一个计数器。
    """
    from app.schemas import BookInfo

    monkeypatch.setattr(main.settings, "ratelimit_extract_per_min", 2)
    # 两条路径都要打桩。**只打批量那条是不够的** —— 漏掉 /extract 用的
    # extract_book_info 的话，它会真去调 LLM（conftest 里塞的是假 key），
    # 报 500 而不是 200，看起来像限流接错了。（第一版就是这么挂的。）
    monkeypatch.setattr(main, "extract_book_info", lambda raw: BookInfo(book_name="x"))
    monkeypatch.setattr(
        main,
        "extract_book_info_many",
        lambda texts, **kw: [BookInfo(book_name="x")] * len(texts),
    )

    assert client_and_app.post("/extract", json={"raw_text": "a"}).status_code == 200
    assert (
        client_and_app.post("/extract/batch", json={"texts": ["a"]}).status_code == 200
    )
    # 桶已经用满 2 次 → 无论打哪个接口都该被拦
    assert client_and_app.post("/extract", json={"raw_text": "a"}).status_code == 429
    assert (
        client_and_app.post("/extract/batch", json={"texts": ["a"]}).status_code == 429
    )


def test_health里能看到redis状态(client_and_app):
    """降级必须**可见** —— 否则你会以为有闸门，实际敞着。"""
    body = client_and_app.get("/health").json()
    assert "redis" in body, "/health 必须暴露 Redis 状态"
    assert body["redis"]["enabled"] is True
    assert body["redis"]["connected"] is True
    # 密码不许出现在 /health 里
    assert "123456" not in str(body), "/health 泄漏了 Redis 密码"


def test_检索接口不限流(client_and_app):
    """`/search` 不烧 key，不该被限 —— 限流限的是成本，不是流量。

    这条同时是个**防呆**：将来有人图省事给所有接口加限流中间件，
    演示时狂点搜索框就会被自己拦住。
    """
    for _ in range(25):
        r = client_and_app.post("/search", json={"query": "高等数学", "top_k": 1})
        assert r.status_code == 200, "检索被限流了 —— 它不烧模型调用，不该限"


# ============================================================================
# 以图搜书：限流 + 多本扇出
#
# 这个接口**长得像检索、其实是成本接口**（第一步就调视觉模型），所以它同时
# 踩两个坑：闸门容易漏挂，扇出容易默默截断。两条都在这儿钉住。
# ============================================================================


@pytest.fixture
def free_vision(monkeypatch):
    """把视觉识别打桩成「就认出这几本」，不碰真模型、不花钱、不要 key。"""

    def _install(books):
        monkeypatch.setattr(
            main.vision, "extract_books_from_image", lambda image: list(books)
        )

    return _install


@pytest.fixture
def counting_search(monkeypatch):
    """把检索打桩并**记账**：返回每一本被拿去搜的书名。

    断言「搜了几本」比断言 HTTP 状态码有信息量得多 ——
    截断这类 bug 不会改状态码。
    """
    calls: list[str] = []

    def _fake_search(**kwargs):
        calls.append(kwargs["query"])
        return []

    monkeypatch.setattr(main.search_service, "search", _fake_search)
    return calls


def test_以图搜书也被限流(client_and_app, free_vision, monkeypatch):
    """⚠️ 这条补的是**漏了一整个里程碑**的闸门。

    它走 /search 那条路径，很容易被归进「检索接口不烧模型、不限流」那类 ——
    但它第一步就是一次视觉调用，比文本模型慢一个数量级。
    """
    monkeypatch.setattr(main.settings, "ratelimit_image_per_min", 2)
    free_vision(["高等数学"])
    body = {"image": "https://example.com/a.jpg"}

    assert client_and_app.post("/search/by-image", json=body).status_code == 200
    assert client_and_app.post("/search/by-image", json=body).status_code == 200
    last = client_and_app.post("/search/by-image", json=body)
    assert last.status_code == 429
    assert int(last.headers["Retry-After"]) > 0


def test_书上限必须比搜上限宽():
    """两个上限要是设成同一个数，下面那条「差额要露出来」就无从谈起了 ——
    识别和检索的成本结构完全不同，拧成一股就是把贵的那头绑死在便宜的那头上。
    """
    assert main.vision._MAX_SEARCH_PER_IMAGE < main.vision._MAX_BOOKS_PER_IMAGE


def test_认出超过上限时只搜上限本但书名全给(
    client_and_app, free_vision, counting_search, monkeypatch
):
    """⚠️ 这个差额是**故意露出来**的，不能藏。

    识别是一次模型调用（认 1 本和认 30 本一样贵），检索是每本一次完整链路
    （真算力），所以两个上限分开设。但少搜的那几本不能凭空消失 ——
    调用方要能从 `len(results) < len(recognized_books)` 看出来，
    而不是以为「图里就那么几本」。

    这里把上限**临时改小**再用，而不是照着生产值写死 15/10 这种数字：
    生产值是要被调优的，测试不该跟着一起改（上一版就是这么脆的，
    上限从 10 提到 20 时它直接红了）。
    """
    monkeypatch.setattr(main.settings, "ratelimit_image_per_min", 100)
    monkeypatch.setattr(main.vision, "_MAX_SEARCH_PER_IMAGE", 3)
    books = [f"第{i}本书" for i in range(8)]
    free_vision(books)

    body = client_and_app.post(
        "/search/by-image", json={"image": "https://example.com/a.jpg"}
    ).json()

    assert body["recognized_books"] == books, "认出来的 8 本一本都不能少"
    assert len(body["results"]) == 3
    assert counting_search == books[:3], "搜的应该正好是前 3 本，顺序不能乱"
    # 前缀性质：results 与 recognized_books 同序一一对应
    assert [r["book"] for r in body["results"]] == body["recognized_books"][:3]


def test_认不出书时一次检索都不该发生(
    client_and_app, free_vision, counting_search, monkeypatch
):
    """拿一句敷衍的词去检索比返回空更糟：会命中一堆书名带那个词的书，
    用户还以为是「识别错了」，其实是我们在乱搜。
    """
    monkeypatch.setattr(main.settings, "ratelimit_image_per_min", 100)
    free_vision([])

    body = client_and_app.post(
        "/search/by-image", json={"image": "https://example.com/a.jpg"}
    ).json()

    assert body["recognized_books"] == []
    assert body["results"] == []
    assert counting_search == [], "没认出书，检索压根不该被调用"
