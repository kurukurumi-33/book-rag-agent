"""调用 Spring Boot 主服务的 HTTP 客户端。

**这是铁律一的落点**：AI 服务不直连业务数据库，要帖子数据就调主服务的 REST API。
（唯一的豁免是 scripts/ 下的开发期脚本，它们图省事可以直连库。）

面试里「为什么拆两个服务」的答案就在这里：
AI 服务是**无状态的纯函数**——不持有数据库连接、不在本地存业务数据、
重启不丢任何东西，所以可以随便扩成多副本。

## 一个设计取舍：向量库里不存业务数据的副本

建索引时只把「帖子 id + 几个能直接展示的字段」写进向量库，
详情仍然回主服务按 id 取（`get_post`）。

这样避免了「同一份数据存两处、要保证两边一致」的经典麻烦。
代价是查详情多一次 HTTP —— 220 条数据的场景完全划算。
"""

import httpx

from app.config import settings

# 主服务是本地进程，正常 10ms 内就返回。
# 给 30 秒是因为建索引时可能一次拉 500 条，走 MySQL + 序列化会慢一些。
DEFAULT_TIMEOUT = 30.0


class BookServiceError(RuntimeError):
    """主服务连不上 / 超时 / 返回了 4xx、5xx。

    单独定义异常类型，是为了让调用方能区分「主服务挂了」和「这段代码写错了」——
    前者该返回 502/503 让上游重试，后者是 500 该修 bug。
    """


def _get(path: str, params: dict | None = None):
    url = f"{settings.book_service_url}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise BookServiceError(
            f"主服务返回 {e.response.status_code}：{url}"
            f"　响应体：{e.response.text[:300]}"
        ) from e
    except httpx.HTTPError as e:
        # ConnectError / ReadTimeout / ... 都归到这里
        raise BookServiceError(
            f"连不上主服务 {url}（{type(e).__name__}: {e}）。"
            f"确认 Spring Boot 已启动、端口是 8080、book_service_url 用的是 127.0.0.1。"
        ) from e
    return resp.json()


def fetch_posts(extract_status: str | None = "DONE", limit: int = 500) -> list[dict]:
    """按抽取状态拉帖子列表（建索引用）。

    默认只要 DONE：PENDING 的还没抽取、结构化字段是空的，拼出来的检索文本没有意义；
    FAILED 的抽取本身就没成功，拼出来是原文，会污染向量空间。

    :param extract_status: PENDING / DONE / FAILED；传 None 表示不过滤
    :param limit: 主服务那边会截断到 500
    """
    params: dict = {"limit": limit}
    if extract_status:
        params["extractStatus"] = extract_status
    return _get("/api/posts", params)


def get_post(post_id: int) -> dict | None:
    """按 id 取单条帖子详情。查不到时主服务返回 JSON null。"""
    return _get(f"/api/posts/{post_id}")
