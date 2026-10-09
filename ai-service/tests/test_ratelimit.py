"""限流（M4.5）的行为契约。

## 这里锁死的三件事

1. **Redis 不可用时 fail-open** —— 放行，不抛。Redis 是可选依赖，
   为了限流把 `/chat` 打死是本末倒置。
2. **窗口边界**：滑出窗口的记录要被清掉（这是「滑动」两个字的全部含义），
   固定窗口那个「0:59 打满、1:00 再打满」的漏洞不能有。
3. **超过限额返回 Retry-After**，且它是**正数** —— 给 0 或负数
   等于让客户端立刻重试，反而更容易把服务压垮。

## 怎么做到不连真 Redis

`redis_client.get_redis` 换成返回一个**内存版假客户端**。
被测的是滑动窗口的算法和降级分支，不是 Redis 本身 ——
后者是 redis-py 的活，不是我们的。
"""

import time

import pytest

from app.services import ratelimit, redis_client


class _FakeRedis:
    """内存版 Redis，只实现限流用到的那几个命令 + eval。

    `eval` 直接调我们自己的 Lua 脚本？不行 —— 那需要真 Lua 解释器。
    所以这里**用 Python 复刻一遍滑动窗口的语义**，但只复刻 lupa 之外的部分：
    真正被测的是「调用方怎么解读返回值」，以及降级分支。
    """

    def __init__(self, limit_behavior=None):
        self.store: dict[str, list[tuple[float, str]]] = {}
        self.fail = False
        self.eval_calls: list[tuple] = []
        self._behavior = limit_behavior

    def eval(self, script, numkeys, key, now, window, limit, member):
        if self.fail:
            raise ConnectionError("假的：Redis 挂了")
        self.eval_calls.append((key, now, window, limit, member))

        bucket = self.store.setdefault(key, [])
        # ① 清掉滑出窗口的
        bucket[:] = [(t, m) for t, m in bucket if t > now - window]
        # ② 数
        if len(bucket) >= limit:
            oldest = min(t for t, _ in bucket)
            self._behavior = ("denied", int(oldest))
            return [0, int(oldest)]
        bucket.append((now, member))
        return [1, len(bucket)]


@pytest.fixture
def fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(redis_client, "get_redis", lambda: fake)
    return fake


@pytest.fixture
def no_redis(monkeypatch):
    """模拟「Redis 连不上」：get_redis 返回 None。"""
    monkeypatch.setattr(redis_client, "get_redis", lambda: None)


# ---- fail-open --------------------------------------------------------------


def test_redis不可用时放行(no_redis):
    """⚠️ 核心降级契约。Redis 是可选依赖，它挂了不该把 /chat 也拖下水。"""
    result = ratelimit.check("chat", "1.2.3.4", limit=1, window_s=60)
    assert result.allowed is True
    assert result.retry_after == 0


def test_redis不可用时连续调用也一直放行(no_redis):
    """降级不是「放过第一次」，是**完全不限**。"""
    for _ in range(50):
        assert ratelimit.check("chat", "1.2.3.4", limit=1, window_s=60).allowed


def test_eval抛异常时也放行并留痕(fake_redis):
    """「连上过又掉线」和「一开始就没连上」是两回事，都要放行，
    但这条路径要打 warning（这里只断言行为，日志由 logging 模块负责）。"""
    fake_redis.fail = True
    result = ratelimit.check("chat", "1.2.3.4", limit=5, window_s=60)
    assert result.allowed is True


# ---- 限额判定 ---------------------------------------------------------------


def test_限额内放行(fake_redis):
    for i in range(3):
        r = ratelimit.check("chat", "1.2.3.4", limit=3, window_s=60)
        assert r.allowed, f"第 {i + 1} 次不该被拦"
        assert r.remaining == 2 - i


def test_到限额就开始拦(fake_redis):
    for _ in range(3):
        ratelimit.check("chat", "1.2.3.4", limit=3, window_s=60)
    r = ratelimit.check("chat", "1.2.3.4", limit=3, window_s=60)
    assert r.allowed is False
    assert r.remaining == 0


def test_超限时RetryAfter是正数(fake_redis):
    """⚠️ 给 0 等于让客户端立刻重试 —— 那是 429 的经典写法错误。"""
    for _ in range(2):
        ratelimit.check("chat", "1.2.3.4", limit=2, window_s=60)
    r = ratelimit.check("chat", "1.2.3.4", limit=2, window_s=60)
    assert r.allowed is False
    assert r.retry_after > 0, f"Retry-After 必须是正数，拿到 {r.retry_after}"


def test_不同IP各算各的(fake_redis):
    """限流是**按身份**的。共用桶的话，一个用户刷满会把所有人挡在门外。"""
    for _ in range(2):
        ratelimit.check("chat", "1.1.1.1", limit=2, window_s=60)
    assert ratelimit.check("chat", "1.1.1.1", limit=2, window_s=60).allowed is False
    assert ratelimit.check("chat", "2.2.2.2", limit=2, window_s=60).allowed is True


def test_不同scope各算各的(fake_redis):
    """⚠️ /chat 和 /extract 必须分开算：一次 batch 抽 20 条，
    共用桶会让「批量抽取时把对话配额吃光」。"""
    for _ in range(2):
        ratelimit.check("chat", "1.1.1.1", limit=2, window_s=60)
    assert ratelimit.check("chat", "1.1.1.1", limit=2, window_s=60).allowed is False
    assert ratelimit.check("extract", "1.1.1.1", limit=2, window_s=60).allowed is True


def test_key里带scope和identity(fake_redis):
    ratelimit.check("chat", "9.9.9.9", limit=5, window_s=60)
    key = fake_redis.eval_calls[0][0]
    assert "chat" in key and "9.9.9.9" in key
    assert key.startswith(redis_client.KEY_PREFIX), "key 必须带统一前缀，否则会和别的项目撞"


# ---- 滑动窗口的语义 ---------------------------------------------------------


def test_滑出窗口的记录会被清掉(fake_redis):
    """这条是「滑动窗口」区别于「固定窗口」的全部意义所在。

    固定窗口的经典漏洞：0:59 打满 → 1:00 计数器归零 → 再打满，
    两秒内实际放行了两倍。滑动窗口不会有这个洞：**时间往前走，
    旧记录就自动滑出去了**。

    ⚠️ 用**真实时钟**（time.time）来喂旧记录，别用自造的数字 ——
    第一版我图省事写了个 now=1000.0，那些时间戳离真实时钟有几十年，
    于是「被清掉」是被清掉了，但测的是「几十年前的记录会被清」这种废话，
    跟「70 秒前刚好滑出窗口」不是一回事。
    """
    key = redis_client.key("rl", "chat", "1.2.3.4")
    real_now_ms = int(time.time() * 1000)

    # 两条**已经滑出** 60 秒窗口的记录，一条**仍在窗口内**的
    fake_redis.store[key] = [
        (real_now_ms - 70_000, "old-out"),   # 70 秒前 → 窗口外，该被清
        (real_now_ms - 65_000, "old-out2"),  # 65 秒前 → 窗口外，该被清
        (real_now_ms - 5_000, "fresh"),      # 5 秒前  → 窗口内，**要算数**
    ]

    # limit=2：如果只剩那条 fresh，就还差一格 → 应该**放行**
    r = ratelimit.check("chat", "1.2.3.4", limit=2, window_s=60)
    assert r.allowed is True, "窗口外的旧记录必须被清掉，否则窗口就变成「永久」了"
    assert r.remaining == 0, "清完只剩 1 条 + 这一次 = 2 条，正好用满"


def test_窗口内的记录不会被清(fake_redis):
    """上面那条的**正对照** —— 少了它，一个「每次都把桶清空」的实现也能过测试。"""
    key = redis_client.key("rl", "chat", "5.5.5.5")
    real_now_ms = int(time.time() * 1000)
    fake_redis.store[key] = [
        (real_now_ms - 1_000, "a"),
        (real_now_ms - 2_000, "b"),
        (real_now_ms - 3_000, "c"),
    ]
    # limit=3，窗口内已经 3 条 → 该拦
    r = ratelimit.check("chat", "5.5.5.5", limit=3, window_s=60)
    assert r.allowed is False, "窗口内的记录被误清了 —— 那限流就形同虚设"


def test_member必须唯一否则同毫秒请求只算一次(fake_redis):
    """⚠️ 同一毫秒进来的多个请求，如果 member 用时间戳，ZSET 会当成同一条。

    表现是：高并发下 N 个请求被算成 1 次，**限流悄悄失效**且很难发现。
    """
    seen = set()
    for _ in range(5):
        ratelimit.check("chat", "1.2.3.4", limit=10, window_s=60)
    for _, _, _, _, member in fake_redis.eval_calls:
        seen.add(member)
    assert len(seen) == 5, "每个请求的 member 都必须不同"


# ---- 限额配比 ---------------------------------------------------------------


def test_extract的限额比chat宽():
    """有意的设计：批量抽取是内部调用（建索引时连着打几百次），
    限得和对话一样紧会把正常运维动作限死。但**也不能不限**。"""
    assert ratelimit.limit_for("extract") > ratelimit.limit_for("chat")
    assert ratelimit.limit_for("extract") > 0


def test_未知scope有兜底限额():
    """新增接口忘了登记时，不能变成「限额 0」把请求全拦死。"""
    assert ratelimit.limit_for("某个还没登记的接口") > 0
