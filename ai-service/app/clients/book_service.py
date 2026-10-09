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
代价是查详情多一次 HTTP —— 本地进程间调用 10ms 量级，换来的是「改字段不用重建索引」。
"""

import re

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


def _request(method: str, path: str, *, params: dict | None = None, json_body=None):
    """给主服务发一个请求，统一处理错误。

    GET / POST 共用这一份错误处理。分开写的话，每加一个动词就要把
    「区分 HTTPStatusError 和 HTTPError」那段抄一遍 —— 抄漏一处就少一种错误提示，
    而且报错文案会各写各的。
    """
    url = f"{settings.book_service_url}{path}"
    try:
        resp = httpx.request(
            method, url, params=params, json=json_body, timeout=DEFAULT_TIMEOUT
        )
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

    # ⚠️ Spring 的 @RestController 方法返回 null 时，响应是 **200 + 空体**，
    #    不是 JSON 的 `null`（实测：Content-Length: 0）。直接 resp.json() 会抛
    #    JSONDecodeError —— 那是 ValueError 的子类，**不被上面的 httpx.HTTPError 兜住**，
    #    会一路炸到调用方。所以这里必须先判空。
    if not resp.content.strip():
        return None

    return resp.json()


def _get(path: str, params: dict | None = None):
    return _request("GET", path, params=params)


def _post(path: str, json_body):
    return _request("POST", path, json_body=json_body)


def ping(timeout: float = 1.5) -> dict:
    """探一下主服务活着没有。给 `/health` 用。**永远不抛异常。**

    ## 为什么单独写一个，而不是让 /health 直接调 fetch_posts

    1. **超时必须短。** `DEFAULT_TIMEOUT` 是 30 秒（迁就建索引拉 500 条），
       拿它来探活，主服务挂掉时 `/health` 会卡 30 秒才回 —— 探活接口自己先死了。
    2. **不能抛。** /health 的语义是「报告状态」，主服务挂了正是**它要报告的内容**，
       不是让它也跟着 500。所以这里把异常收成返回值。
    3. **不烧模型。** 只打一个轻量的 GET。LLM 那条链路**故意不实测** ——
       探活每次烧一次 token 太贵，而且 key 有效性在真调用时自然暴露。

    Returns:
        {"reachable": bool, "url": str, "status_code": int | None, "error": str | None}
    """
    url = f"{settings.book_service_url}/api/posts"
    try:
        # 只取 1 条：目的是「通不通」，不是「数据对不对」
        resp = httpx.get(
            url, params={"limit": 1}, timeout=timeout
        )
    except httpx.HTTPError as e:
        return {
            "reachable": False,
            "url": url,
            "status_code": None,
            "error": f"{type(e).__name__}: {e}",
        }
    # 4xx/5xx 也算「到了但不对」—— reachable 描述的是**网络可达**，
    # 所以这里看状态码而不是直接判 True。两个字段各说各的，不混在一起。
    return {
        "reachable": resp.status_code < 500,
        "url": url,
        "status_code": resp.status_code,
        "error": None if resp.status_code < 500 else f"HTTP {resp.status_code}",
    }


def fetch_posts(
    extract_status: str | None = "DONE", limit: int = 500, offset: int = 0
) -> list[dict]:
    """按抽取状态拉帖子列表（建索引用）。**单页**，上限 500 条。

    默认只要 DONE：PENDING 的还没抽取、结构化字段是空的，拼出来的检索文本没有意义；
    FAILED 的抽取本身就没成功，拼出来是原文，会污染向量空间。

    :param extract_status: PENDING / DONE / FAILED；传 None 表示不过滤
    :param limit: 主服务那边会截断到 500
    :param offset: 跳过前多少条。要拿全量得用 fetch_all_posts 翻页，别指望调大 limit

    ⚠️ 别用 `fetch_posts(limit=10000)` 去「一次拉完」：主服务把 limit 夹在 500，
    传 10000 进去只会拿回 500 条，**而且不报错** —— 静默少数据，最难查。
    """
    params: dict = {"limit": limit, "offset": offset}
    if extract_status:
        params["extractStatus"] = extract_status
    return _get("/api/posts", params)


def fetch_all_posts(
    extract_status: str | None = "DONE",
    limit: int | None = None,
    page_size: int = 500,
) -> list[dict]:
    """翻页拉全量帖子。

    为什么要有这个函数：主服务单次最多给 500 条（故意的，防一次拉爆内存），
    而语料有 1 万条。`fetch_posts` 是单页接口，拿全量必须自己循环。

    :param limit: 最多总共要多少条；None = 不设上限，一直翻到翻完
    :param page_size: 每页多少条。默认贴着主服务 500 的上限，翻页次数最少

    两道防死循环的保险（都遇到过类似的事）：

      1. **短页即终止** —— 返回条数 < page_size 说明是最后一页了。
         不要去「预先查总数」再算页数：那要多一个接口、多一次往返，
         而且两次调用之间数据可能变。
      2. **去重 + 上限** —— 万一主服务忽略了 offset（比如回退到旧版本），
         它会每页都返回同样的 500 条。短页判断永远不会触发，就死循环了。
         所以按 id 去重：某一页**全是重复**就说明页没翻动，立刻停。
    """
    all_posts: list[dict] = []
    seen: set = set()
    offset = 0

    while True:
        # 最后一页只取需要的量，别多拉
        size = page_size
        if limit is not None:
            remaining = limit - len(all_posts)
            if remaining <= 0:
                break
            size = min(page_size, remaining)

        page = fetch_posts(extract_status=extract_status, limit=size, offset=offset)

        fresh = [p for p in page if p.get("id") not in seen]
        seen.update(p.get("id") for p in page)
        all_posts.extend(fresh)

        # 保险 1：短页 = 到头了
        if len(page) < size:
            break
        # 保险 2：整页都是见过的 —— offset 没起作用，再翻下去是死循环
        if not fresh:
            break

        offset += len(page)

    return all_posts


def get_post(post_id: int) -> dict | None:
    """按 id 取单条帖子详情。查不到时返回 None（主服务回的是 200 + 空体）。"""
    return _get(f"/api/posts/{post_id}")


# ============================================================================
# 会话历史（M3 第 4 步）
#
# 这两个函数是「会话记忆存在 MySQL」这件事在 Python 侧的全部。
# loop.py 里的 chat_once **不知道**它们存在 —— 它只负责算出新历史，
# 存哪、怎么存是调用方（POST /chat 那个接口）的事。
# ============================================================================


# 允许的 session_id 形状：字母 / 数字 / 下划线 / 连字符，最长 64。
# 跟主服务那张表的 VARCHAR(64) 对齐。
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _chat_path(session_id: str) -> str:
    """拼会话的 URL 路径，顺便把 session_id 挡在合法字符集内。

    session_id 会拼进 URL 的**路径段**，而路径段装不下任意用户输入：

        "a/b c"  ->  编码成 "a%2Fb%20c"  ->  Tomcat 直接回 400
                     （容器默认拒绝路径里的 %2F，防目录穿越，Spring 根本看不到）
        "a/b"    ->  不编码的话路由被拆成 /api/chat/a/b/messages，匹配不上

    试过 quote() 转义，没用 —— 转义后照样被容器拒。所以这里的结论是
    **约束字符集**，而不是「想办法转义」。我们自己的会话 ID 是 uuid4().hex，
    本来就只含字母数字；约束住它，这一类问题就整体消失了。

    主服务那边有同样的正则做二道防线（这里没挡住的话，那边回 400）。
    """
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            f"session_id 只能是字母/数字/下划线/连字符，最长 64 位，收到 {session_id!r}"
        )
    return f"/api/chat/{session_id}/messages"


def load_history(session_id: str, limit: int = 20) -> list[dict]:
    """取某个会话最近的消息，**按时间正序**（最老的在前），可以直接拼进 messages。

    返回形如 [{"role": "human", "content": "有高数吗", ...}, ...]

    取多少条由 limit 决定，主服务那边会截断到 200。这是**有意的截断**：
    会话可以无限长，全量读回来会把模型的上下文窗口撑爆。
    """
    return _get(_chat_path(session_id), {"limit": limit}) or []


def save_messages(session_id: str, messages: list[dict]) -> list[dict]:
    """把**一轮**产生的消息一次写进去。

    messages 形如 [{"role": "human", "content": "..."}, {"role": "ai", "content": "..."}]

    ⚠️ 一定要**一次提交整轮**，不要拆成两次调用。主服务那边用同一个事务写，
    拆开的话两个请求之间进程一挂，库里就只剩问、没有答，下一轮接不上话。
    """
    return _post(_chat_path(session_id), messages) or []
