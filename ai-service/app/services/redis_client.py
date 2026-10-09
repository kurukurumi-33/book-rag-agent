"""Redis 客户端（M4.5：限流 + 会话历史缓存）。

## 为什么 Redis 放 AI 服务，而不是主服务

铁律是「**AI 服务不直连业务数据库**」。Redis 在这里装的东西**不是业务数据**：

    业务数据：帖子、会话历史、订单 —— 真相在 MySQL，丢了不能重建
    运维态：  限流计数、缓存 —— 可丢、可重建、进程重启后从零开始也没什么大不了

限流计数就是这个项目最典型的「运维态」。它天然属于「谁提供服务谁负责」，
而 `/chat` `/extract` 是 AI 服务提供的，所以计数器跟着 AI 服务走。
（放主服务也能做，但那样 AI 服务的限流要跨进程问主服务要配额 —— 多一次往返，
而且主服务一挂，AI 服务的保护也跟着没了。）

## 为什么不用「进程内一个 dict 计数」

这是面试一定会追问的点，答案很短：**多副本会各算各的**。

    limit = 20/分钟，起了 3 个副本
    → 把请求分散到 3 个副本，实际能打进去 60 次/分钟
    → 限流形同虚设，而且**不会报错**，你只会觉得「怎么配额超了」

Redis 是进程外的，三个副本看到的是同一个计数器。这也是「无状态服务」
能真正水平扩容的前提之一 —— 状态必须挪到共享存储里，
**要么不存，要么存在所有副本都看得见的地方**。

## 连接失败怎么办：fail-open（放行），但必须吵

`get_redis()` 连不上就返回 None，调用方**放行并打一条 warning**。

为什么是 fail-open 而不是 fail-closed：

- 这是个**演示项目**，Redis 只是个可选依赖。为了限流把整个 `/chat` 打死，
  代价远大于「这一分钟多放了几个请求」。
- fail-closed 更常见的场景是**计费/风控**（宁可拒绝服务也不能超支）——
  我们不是那个场景。

⚠️ 但「放行」不等于「静默」：**每次降级都会打 warning**，
且 `/health` 里能直接看到 `redis.connected`。
**能降级，但降级必须可见** —— 这是本项目一贯的立场
（同 rerank 加载失败、vision 未配置的处理）。
"""

import logging
import threading

import redis

from app.config import settings

log = logging.getLogger("uvicorn.error")

# 所有 key 统一前缀。Redis 是**共享**的 —— 本机还跑着苍穹外卖那个项目
# （它用的是 db10），加前缀是为了在同一个 db 里也不会撞名。
KEY_PREFIX = "zhimi"

_client: "redis.Redis | None" = None
_failed = False  # 连不上就置位，进程内不再反复重试（避免每个请求都卡一次连接超时）
_lock = threading.Lock()


def get_redis() -> "redis.Redis | None":
    """返回 Redis 客户端；没开或连不上时返回 **None**（不抛异常）。

    返回 None 而不是抛，是为了让「Redis 不可用」变成调用方**必须显式处理**的一条分支 ——
    想 fail-open 就得自己写出来，想 fail-closed 也得自己写出来。
    抛异常的话，这个决定就被藏在某处的 try/except 里了。
    """
    global _client, _failed

    if not settings.redis_enabled:
        return None
    if _client is not None or _failed:
        return _client

    with _lock:
        if _client is not None or _failed:
            return _client
        try:
            client = redis.from_url(
                settings.redis_url,
                decode_responses=True,  # 回 str 而不是 bytes，省掉到处 .decode()
                socket_timeout=settings.redis_timeout,
                socket_connect_timeout=settings.redis_timeout,
            )
            client.ping()
            _client = client
            log.info("Redis 已连接：%s", _mask(settings.redis_url))
        except Exception:
            _failed = True
            log.warning(
                "Redis 连不上（%s），限流与会话缓存将**降级为放行/直读**；"
                "服务本身不受影响。启动 Redis 后需重启本服务才能恢复。",
                _mask(settings.redis_url),
                exc_info=True,
            )
    return _client


def _mask(url: str) -> str:
    """把 URL 里的密码换成 ***，免得密码进日志。

    redis://:123456@127.0.0.1:6379/0  →  redis://:***@127.0.0.1:6379/0
    """
    if "@" not in url:
        return url
    head, tail = url.rsplit("@", 1)
    if ":" not in head:
        return url
    scheme, _, _ = head.partition("://")
    return f"{scheme}://:***@{tail}"


def key(*parts: str) -> str:
    """拼一个带统一前缀的 key。

    统一走这个函数，是为了让「本项目的 key 长什么样」只有一处定义 ——
    排查的时候 `KEYS zhimi:*` 一把梭，不会漏。
    """
    return ":".join([KEY_PREFIX, *parts])


def status() -> dict:
    """给 /health 看的状态：**连着、没连上、还是根本没开**，三态要能分开。"""
    client = get_redis()
    if client is None:
        return {
            "enabled": settings.redis_enabled,
            "connected": False,
            "error": "未启用" if not settings.redis_enabled else "连接失败（见启动日志）",
        }
    try:
        info = client.info()
        return {
            "enabled": True,
            "connected": True,
            "version": info.get("redis_version"),
            "url": _mask(settings.redis_url),
        }
    except Exception as e:  # 连上了又掉线
        return {"enabled": True, "connected": False, "error": str(e)[:120]}
