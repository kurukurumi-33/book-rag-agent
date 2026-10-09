"""跑在**真 Redis** 上的用例（`-m integration`）。

## 为什么必须有这一组

`test_ratelimit.py` 里那个假 Redis 是**用 Python 复刻了一遍滑动窗口的语义**。
这意味着一件很尴尬的事：**Lua 脚本本身一行都没被测过** ——
如果 Lua 写错了（漏了 ZREMRANGEBYSCORE、EXPIRE 没设、返回值顺序反了），
只要我的 Python 复刻版是对的，单测就会全绿。**测试和一个错误实现达成了一致。**

这类「测试替身把被测逻辑也一起替换掉」的坑，本质是**测试和真实现共享了同一个错误假设**。
唯一的解法是让真东西跑一遍。所以这几个用例直连真 Redis。

## 怎么跑

    # 先起 Redis（本机是 D:/develop/redis/... 那个）
    cd ai-service
    ./.venv/Scripts/python.exe -m pytest -m integration tests/test_redis_integration.py -q

没起 Redis 的话会**跳过**而不是失败 —— 这几条不是「代码对不对」的判据，
是「环境配好没有」的判据，混在一起会让「本机没开 Redis」看起来像是代码坏了。

## 用独立的 key 前缀，跑完自己清干净

不碰任何已有的 key（`zhimi:rl:*` / `zhimi:hist:*` 是线上在用的）。
这几个用例只读写 `zhimi:test:*`，结束即删。
"""

import time
import uuid

import pytest

from app.services import redis_client, session_cache

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """真 Redis 客户端；连不上就跳过整组。"""
    c = redis_client.get_redis()
    if c is None:
        pytest.skip("本机 Redis 没起 / REDIS_URL 不对 —— 跳过，不算失败")
    # 先探活：get_redis() 会缓存「连上了」这个结论，但服务中途挂掉时它不会自己知道
    try:
        c.ping()
    except Exception as e:
        pytest.skip(f"Redis 探活失败：{e}")
    return c


@pytest.fixture
def probe_key(client):
    """一个独属于本次用例的 key，用完删掉。"""
    k = f"{redis_client.KEY_PREFIX}:test:{uuid.uuid4().hex[:8]}"
    yield k
    client.delete(k)


# ---- Lua 脚本本身 -----------------------------------------------------------


def test_Lua滑动窗口_到限额就拦(client, probe_key):
    """真的把 Lua 发到 Redis 上跑 —— 这是这组用例存在的唯一理由。"""
    from app.services.ratelimit import _LUA_SLIDING_WINDOW

    now = int(time.time() * 1000)
    results = [
        client.eval(_LUA_SLIDING_WINDOW, 1, probe_key, now + i, 60_000, 3, f"m{i}")
        for i in range(5)
    ]

    flags = [int(r[0]) for r in results]
    assert flags == [1, 1, 1, 0, 0], f"限额 3 应该是前三次放行、后两次拦住，实际 {flags}"


def test_Lua超限时回的是窗口内最早的时间戳(client, probe_key):
    """调用方要靠这个值算 Retry-After。回错了（比如回成条数、或回最新那条）
    会让 Retry-After 算成一个荒谬的数字。"""
    from app.services.ratelimit import _LUA_SLIDING_WINDOW

    now = int(time.time() * 1000)
    for i in range(2):
        client.eval(_LUA_SLIDING_WINDOW, 1, probe_key, now + i, 60_000, 2, f"m{i}")
    denied = client.eval(_LUA_SLIDING_WINDOW, 1, probe_key, now + 2, 60_000, 2, "m2")

    assert int(denied[0]) == 0
    oldest = int(denied[1])
    # 必须落在「最早那次的时刻」附近，而不是条数(2)、也不是现在的时刻
    assert now <= oldest <= now + 1, f"应是窗口内最早的时间戳，拿到 {oldest}"


def test_Lua会设置过期时间(client, probe_key):
    """⚠️ 漏了 EXPIRE 的话，每个访问过的 IP 都会在 Redis 里留一个**永久** ZSET。

    这个 bug 在本地永远看不出来 —— 内存够大，也不会报错，
    只会在某天 `INFO memory` 里发现一堆不认识的 key。
    """
    from app.services.ratelimit import _LUA_SLIDING_WINDOW

    client.eval(_LUA_SLIDING_WINDOW, 1, probe_key, int(time.time() * 1000), 60_000, 5, "m")
    ttl = client.ttl(probe_key)
    assert 0 < ttl <= 61, f"TTL 应该是 61 秒左右，拿到 {ttl}（-1 = 永不过期，是 bug）"


def test_Lua里滑出窗口的记录会被真清掉(client, probe_key):
    """推进时间轴，让旧记录滑出 —— 这条**必须**用真 Redis 验，
    因为「清没清」的结果只存在于 ZSET 里。"""
    from app.services.ratelimit import _LUA_SLIDING_WINDOW

    now = int(time.time() * 1000)
    for i in range(3):
        client.eval(_LUA_SLIDING_WINDOW, 1, probe_key, now + i, 60_000, 3, f"m{i}")
    assert client.zcard(probe_key) == 3

    # 61 秒之后：三条全滑出，应该又能打
    later = now + 61_000
    r = client.eval(_LUA_SLIDING_WINDOW, 1, probe_key, later, 60_000, 3, "m-late")
    assert int(r[0]) == 1, "窗口外记录没被清 —— 那窗口就是「永久」，不是「滑动」"
    assert int(r[1]) == 1, "清完应该只剩刚打进去的这条"


# ---- 会话缓存 ---------------------------------------------------------------


def test_会话缓存_存取与失效(client):
    """走完整的 cache-aside 一轮：写 → 读得到 → 失效 → 读不到。

    ⚠️ 这里**不碰主服务**：只测 session_cache 自己的 get/set/invalidate，
    不测 load_history_cached（那个要连 Spring Boot，属于另一组）。
    """
    sid = f"test-{uuid.uuid4().hex[:8]}"
    history = [{"role": "human", "content": "有高数吗"}]

    session_cache.invalidate(sid)  # 保证起点干净（上一轮可能留了东西）
    assert session_cache.get(sid, 20) is None, "起点应该没有缓存"

    session_cache.set(sid, 20, history)
    assert session_cache.get(sid, 20) == history

    session_cache.invalidate(sid)
    assert session_cache.get(sid, 20) is None, "失效之后必须读不到"


def test_空历史是一个有效结果不是未命中(client):
    """新会话的历史是 `[]`。

    ⚠️ 如果把 `[]` 当成「没缓存」处理，逻辑上也还转得动（会回源）；
    但如果把「未命中」返回成 `[]`，调用方就**永远走不到回源那条路**了 ——
    agent 会永远以为自己没有历史。这两种表示必须分得开。
    """
    sid = f"test-empty-{uuid.uuid4().hex[:8]}"
    session_cache.set(sid, 20, [])
    cached = session_cache.get(sid, 20)
    assert cached == []
    assert cached is not None, "空列表是「命中且为空」，不是「未命中」"
    session_cache.invalidate(sid)


def test_不同limit是不同缓存(client):
    """⚠️ 这个用例锁的是一个**很难查的 bug**：

    若 key 里不带 limit，先读 limit=2 再读 limit=20 会拿到那份 2 条的缓存 ——
    表现成「agent 突然失忆」，而且**只有改变调用顺序时才复现**。
    """
    sid = f"test-limit-{uuid.uuid4().hex[:8]}"
    short = [{"role": "human", "content": "1"}, {"role": "ai", "content": "2"}]
    long = short + [{"role": "human", "content": "3"}]

    session_cache.set(sid, 2, short)
    session_cache.set(sid, 20, long)

    assert session_cache.get(sid, 2) == short
    assert session_cache.get(sid, 20) == long, "不同 limit 串味了 —— key 里必须带 limit"
    session_cache.invalidate(sid)


def test_失效会删掉该会话所有limit的缓存(client):
    """invalidate(sid) 不带 limit 时必须删全部。

    只删一个的话：写完新消息，limit=20 的缓存删了、limit=2 的还留着旧数据，
    下一次带 limit=2 的读就会拿到**上一个回合**的历史。
    """
    sid = f"test-inv-{uuid.uuid4().hex[:8]}"
    session_cache.set(sid, 2, [{"role": "human", "content": "old"}])
    session_cache.set(sid, 20, [{"role": "human", "content": "old"}])

    session_cache.invalidate(sid)

    assert session_cache.get(sid, 2) is None
    assert session_cache.get(sid, 20) is None


def test_缓存损坏时删掉并回源(client):
    """缓存里被塞了非 JSON 的东西（人工改过 / 老版本格式）。

    必须**删掉**再回源 —— 留着的话每次读都解析失败，这个 key 等于永久中毒，
    而且表现成「这个会话永远记不住事」，查起来要命。
    """
    sid = f"test-corrupt-{uuid.uuid4().hex[:8]}"
    client.set(session_cache._key(sid, 20), "这不是 JSON{{{", ex=60)

    assert session_cache.get(sid, 20) is None, "损坏的缓存要当未命中"
    assert not client.exists(session_cache._key(sid, 20)), "而且必须把它删掉"
