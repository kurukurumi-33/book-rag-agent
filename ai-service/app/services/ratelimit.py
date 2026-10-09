"""限流（M4.5）—— 滑动窗口，计数存在 Redis 里。

## 为什么现在要做这个

`/chat` `/extract` `/extract/batch` `/search/by-image`
**一次请求就烧一次（或 N 次）模型调用**，而在此之前它们**一个闸门都没有** ——
谁能访问到 8000 端口，谁就能把 API key 的额度烧光。
这不是「体验优化」，是**成本保护**。

⚠️ 注意 `/search/by-image` 是这份名单里**唯一一个检索接口**。
    `/search` 和 `/index/build` 烧的是本地 CPU，不限；
    但以图搜书的第一步是一次视觉模型调用（比文本模型慢得多），
    后面还跟着最多 10 次完整检索 —— 它按检索归类是错的，按成本归类才对。
    这是「按成本限流，不按 URL 前缀限流」的一个具体例子。

## 为什么是「滑动窗口」而不是「固定窗口」

固定窗口（每分钟一个计数器，整点清零）有个著名的边界漏洞：

    限额 20/分钟
    在某分钟的 0:59 打 20 次 → 计数器清零
    在下分钟的 1:00 再打 20 次 → 又全部通过
    **两秒内实际放行了 40 次**，是限额的两倍

滑动窗口用 ZSET 存「每次请求的时间戳」，每次都先清掉窗口外的记录再数，
所以任意时刻往前看 60 秒，都不会超过限额。

## 为什么用 Lua 脚本

「先清理 → 再计数 → 判断 → 再写入」这四步**必须原子**。

分开发四条命令的话，两个并发请求会同时读到 count=19、同时判定「还没到 20」、
同时写入 —— 结果放行 21 次。**这类 bug 只在并发下出现，本地怎么点都复现不了。**
Redis 单线程执行 Lua，整个脚本期间不会插入别的命令，天然原子。

（另一个常见做法是 INCR + EXPIRE，但那是固定窗口；滑动窗口只能靠 ZSET。）

## fail-open 的分工

`check()` 里 Redis 不可用时**放行**（返回 allowed=True），并打 warning。
理由写在 `redis_client.py` 的开头。这里只强调一条：
**放行不是静默的** —— `redis_client.get_redis()` 那条 warning 就是痕迹，
`/health` 里的 `redis.connected` 也是。
"""

import logging
import time
import uuid
from dataclasses import dataclass

from app.config import settings
from app.services import redis_client

log = logging.getLogger("uvicorn.error")

# 滑动窗口的标准写法：ZSET 的 score = 请求时刻（毫秒），member = 唯一 id。
#
# ⚠️ member 必须是**唯一**的，不能直接用时间戳 —— 同一毫秒进来的两个请求
#    时间戳相同，ZSET 会把它们当成同一个 member，只留一条。
#    结果就是「同一毫秒打进来的请求只算一次」，高并发下限流悄悄失效。
#    加个随机后缀即可（这就是下面那个 uuid4().hex[:8] 的全部作用）。
_LUA_SLIDING_WINDOW = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])   -- 毫秒
local window = tonumber(ARGV[2])   -- 毫秒
local limit  = tonumber(ARGV[3])
local member = ARGV[4]

-- ① 把滑出窗口的记录删掉
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
-- ② 数一下窗口内还剩几次
local count = redis.call('ZCARD', key)

if count >= limit then
    -- 超了。返回窗口内**最早**那次请求的时间，让调用方算出还要等多久。
    -- （不给这个值的话调用方只能盲等或暴力重试。）
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    return {0, math.floor(oldest[2])}
end

redis.call('ZADD', key, now, member)
-- ③ 给整个 key 一个过期时间，防止冷 key 永久占内存。
--    ⚠️ 这一步不能省：滑动窗口的 key 不会自己消失，
--       没有 EXPIRE 的话，每个访问过的 IP 都会在 Redis 里留一个永久 ZSET。
redis.call('EXPIRE', key, math.ceil(window / 1000) + 1)
return {1, count + 1}
"""


@dataclass(frozen=True)
class RateLimitResult:
    """一次限流判定的结果。

    带上 `remaining` / `retry_after` 是为了能拼出标准的响应头 ——
    只回一个 429 而不告诉调用方「还要等几秒」是很不友好的接口设计
    （客户端只能盲目重试，反而更容易把服务打死）。
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # 秒；allowed=True 时是 0


def check(scope: str, identity: str, limit: int, window_s: int) -> RateLimitResult:
    """判断 `identity` 在 `scope` 上这一分钟还能不能再打。

    :param scope: 限流维度，如 "chat" / "extract"。**必须区分** ——
                  /chat 一次烧一次模型调用，/extract/batch 一次烧 20 次，
                  用同一个桶会让「批量抽取时把对话配额吃光」。
    :param identity: 谁在打，通常是客户端 IP（见 main.py 的 _client_id）
    :param limit: 窗口内允许几次
    :param window_s: 窗口长度（秒）

    Redis 不可用时**放行**（fail-open），见模块开头。
    """
    client = redis_client.get_redis()
    if client is None:
        # 降级放行。不在这里打 warning（get_redis 已经打过一次了，
        # 每个请求再打一条会把日志刷爆），但要如实反映在 remaining 上。
        return RateLimitResult(allowed=True, limit=limit, remaining=limit, retry_after=0)

    now_ms = int(time.time() * 1000)
    window_ms = window_s * 1000
    member = f"{now_ms}-{uuid.uuid4().hex[:8]}"
    k = redis_client.key("rl", scope, identity)

    try:
        allowed_flag, value = client.eval(
            _LUA_SLIDING_WINDOW,
            1,
            k,
            now_ms,
            window_ms,
            limit,
            member,
        )
    except Exception:
        # 连上过、但这条命令失败了（Redis 中途挂了 / 主从切换）——
        # 仍然 fail-open，但要**吵**，因为这跟「一开始就没连上」不是一回事。
        log.warning("限流判定失败，本次放行（scope=%s）", scope, exc_info=True)
        return RateLimitResult(allowed=True, limit=limit, remaining=limit, retry_after=0)

    if int(allowed_flag) == 1:
        return RateLimitResult(
            allowed=True,
            limit=limit,
            remaining=max(0, limit - int(value)),
            retry_after=0,
        )

    # 超限：value 是窗口内最早那次请求的时间戳（毫秒）
    oldest_ms = int(value)
    retry_after = max(1, int((oldest_ms + window_ms - now_ms) / 1000) + 1)
    return RateLimitResult(
        allowed=False, limit=limit, remaining=0, retry_after=retry_after
    )


def limit_for(scope: str) -> int:
    """各接口的限额。集中在一处，方便对照着看「哪个接口更贵」。

    ⚠️ `/extract` 和 `/extract/batch` 的限额（60）**比 `/chat`（20）宽得多**，
        这个不对称是故意的，原因有两层：

        1. 批量抽取是**内部调用** —— 建索引时主服务会连着打几百次。
           限得太紧会把「重建索引」这个正常运维动作给限死。
        2. 但**也不能不限**：这两个接口同样烧 key，而且一次 batch 是 20 次调用。
           60 次/分钟 × 20 = 1200 次 LLM 调用/分钟，已经是很宽松的天花板了。

    ⚠️ `image`（以图搜书）是**最紧的一档**（10），比 chat 还紧。
        它不是文本接口：一次请求 = 一张要等几秒的视觉调用 + 最多 10 次完整检索。
        烧的钱和占的 CPU 都不在同一个量级，限额也就不该在同一档。

    ⚠️ 这个字典是**唯一的登记处**：`_enforce_limit` 传进来的 scope 拼错了不会报错，
        会静默拿到 fallback 的 extract 限额。加了新接口记得来这里补一行。
    """
    return {
        "chat": settings.ratelimit_chat_per_min,
        "extract": settings.ratelimit_extract_per_min,
        "image": settings.ratelimit_image_per_min,
    }.get(scope, settings.ratelimit_extract_per_min)
