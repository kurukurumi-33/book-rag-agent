"""会话历史缓存（M4.5）—— cache-aside + **写时失效**。

## 要解决什么

`/chat` 每轮都要把整个会话的历史读回来喂给模型（`load_history`），
而那是一次跨进程 HTTP + MySQL 查询。**同一个会话连着聊十句，
这十次读的是同一份正在变长的数据** —— 典型的读放大。

## 为什么是「写时失效」，不是「写穿更新」

我在计划里写的是写穿（写完顺手把新消息追加进缓存），**实现的时候改成了失效**，
因为写穿在这里有个不容易发现的坑：

    load_history(limit=20) 的缓存里只有**最近 20 条**
    新写进来一轮（2 条），写穿的做法是 append 到那个 20 条的列表
    → 列表变成 22 条，但**下次 load_history 只会返回 20 条**

也就是说写穿要求缓存**复刻主服务的截断逻辑**（按时间倒序取 20 条再正序返回）。
一旦哪天主服务改了排序或截断规则，缓存里就会留下一份「和接口不一样的真相」——
这正是这个项目一直在避免的「第二份业务逻辑」。

失效就没这个问题：**缓存里要么是「主服务某次真实返回过的东西」，要么不存在。**
缓存永远不可能比主服务更聪明，也就永远不会不一致。

实现上就是 `save_messages` 之后 `delete` 掉这个 key，下一次读自然回源。

## ⚠️ 失效是「每次写都删」，不是「删了再更新」——这是刻意的

删掉之后**不马上回填**。回填需要再发一次 HTTP 去拉，而那一次拉的请求
马上就要用自己算好的消息（`loop.py` 已经把新历史拼好了），
回填等于白读一次。**让下一次真正需要的人去读**，这才是 cache-aside。

## 谁在写

只有 AI 服务自己（`POST /api/chat/{sid}/messages` 是我们发的）。
**没有第三方写入者**，所以「删掉缓存」就足以保证一致 ——
不需要版本号、不需要订阅失效通知、不需要 TTL 兜底一致性。
如果哪天主服务那边也能写消息，这个结论立刻失效，那时要换成别的方案
（这条记下来，因为它是这个缓存「敢做得这么简单」的唯一前提）。
"""

import json
import logging

from app.clients import book_service
from app.config import settings
from app.services import redis_client

log = logging.getLogger("uvicorn.error")


def _key(session_id: str, limit: int) -> str:
    """缓存 key 里**必须带上 limit**。

    同一个会话，limit=20 和 limit=2 拿到的**不是同一份数据**（是前缀关系）。
    只用 session_id 做 key 的话，先用 limit=2 读一次，再用 limit=20 读，
    会拿到那个 2 条的缓存 —— 表现成「agent 突然失忆了」，
    而且**只有改变 limit 的调用顺序时才复现**，非常难查。
    """
    return redis_client.key("hist", session_id, str(limit))


def get(session_id: str, limit: int) -> list[dict] | None:
    """读缓存。**返回 None = 未命中**（要和「命中但结果是空列表」区分开）。

    这两种情况必须分开：空列表是一个**有效结果**（新会话确实没有历史），
    把它当成未命中会每次都回源；把它当成命中又会… 其实没事。
    但反过来更糟：未命中返回 `[]` 的话，调用方就永远走不到「回源」那条路了。
    """
    client = redis_client.get_redis()
    if client is None:
        return None
    try:
        raw = client.get(_key(session_id, limit))
    except Exception:
        log.warning("会话缓存读取失败，回源（session=%s）", session_id, exc_info=True)
        return None

    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        # 缓存里是坏数据（人工改过 / 老版本格式）。**删掉它**并回源 ——
        # 留着的话每次读都解析失败，等于这个 key 永久中毒。
        log.warning("会话缓存内容损坏，已删除并回源（session=%s）", session_id)
        invalidate(session_id)
        return None


def set(session_id: str, limit: int, history: list[dict]) -> None:
    """写缓存。失败就算了 —— 缓存写不进去不该影响这次对话。"""
    client = redis_client.get_redis()
    if client is None:
        return
    try:
        # 用 set(..., ex=) 而不是 setex()：redis-py 8.x 已把 setex 标为 deprecated
        # （实测会打 DeprecationWarning）。功能一样，但别在警告里养出一堆噪音 ——
        # 噪音多了之后，真正重要的那条警告就没人看了。
        client.set(
            _key(session_id, limit),
            json.dumps(history, ensure_ascii=False, default=str),
            ex=settings.session_cache_ttl_s,
        )
    except Exception:
        log.warning("会话缓存写入失败（session=%s）", session_id, exc_info=True)


def invalidate(session_id: str, limit: int | None = None) -> None:
    """删缓存 —— **写完一轮必须调这个**。

    :param limit: 只删某一个 limit 的缓存；**None = 删这个会话的所有 limit**

    默认删全部（用 SCAN 而不是 KEYS —— KEYS 会阻塞 Redis，
    生产上是要出事的；虽然这儿是本地演示，但习惯要从一开始就对）。

    ⚠️ 这个函数**不能因为「缓存里可能没有」就跳过** ——
       它必须在每次写完消息之后无条件执行。理由：我们并不知道
       之前有没有人读过（可能被 TTL 清掉了，也可能被 eviction 挤掉了），
       「先查再删」省不了什么，却多了一次往返和一次竞态窗口。
    """
    client = redis_client.get_redis()
    if client is None:
        return

    pattern = (
        _key(session_id, "*") if limit is None else _key(session_id, limit)
    )
    try:
        if limit is not None:
            client.delete(pattern)
            return
        # 一个会话正常只会有一两个 limit 的缓存，scan 一次就完
        for k in client.scan_iter(match=pattern, count=100):
            client.delete(k)
    except Exception:
        log.warning("会话缓存失效失败（session=%s）", session_id, exc_info=True)


# ============================================================================
# 给调用方用的两个包装（cache-aside 的完整形状）
#
# ⚠️ 为什么这两个函数在这里，而不在 clients/book_service.py：
#    clients/ 的职责是「怎么把数据拿过来」（HTTP、超时、错误码），
#    services/ 才是「拿到数据之后做什么」。缓存属于后者。
#    book_service.py 自己的模块注释就是这么划的 —— 把 Redis 塞进去
#    等于一边说「换 gRPC 只动这一个文件」，一边在里面加缓存，自相矛盾。
# ============================================================================


def load_history_cached(session_id: str, limit: int = 20) -> list[dict]:
    """load_history 的 cache-aside 版本：**先查缓存，没有再回源并回填**。

    回填（`set`）只在这里做，不在 `save_messages_cached` 里做 ——
    **缓存里只该有「主服务真的返回过的东西」**，这样它永远不可能比主服务更新。
    """
    hit = get(session_id, limit)
    if hit is not None:
        return hit

    rows = book_service.load_history(session_id, limit)
    set(session_id, limit, rows)
    return rows


def save_messages_cached(session_id: str, messages: list[dict]) -> list[dict]:
    """save_messages + **写时失效**。顺序不能反。

    ⚠️ 必须是「先写主服务、再删缓存」。反过来的话，两步之间如果有并发读，
    会读到旧缓存并**把它回填**，于是刚删掉的脏数据又被写了回去 ——
    这就是经典的 cache-aside 竞态。先写后删把这个窗口压到最小
    （严格地说仍不是零窗口，要根除得用延迟双删或版本号；
    但我们的写者是唯一且串行的（只有 AI 服务自己），这个窗口实际不存在）。
    """
    result = book_service.save_messages(session_id, messages)
    invalidate(session_id)  # 删该会话**所有** limit 的缓存，见 invalidate 的注释
    return result
