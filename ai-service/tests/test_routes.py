"""路由**挂对了没有** —— 一组不需要任何外部依赖的防呆用例（秒级）。

## 为什么要有这个文件

写 M2.6 的时候我把一个辅助函数（`_to_hits`）插在了

    @app.post("/search", ...)     ← 装饰器
    def _to_hits(...): ...        ← 被装饰器抓走的其实是这个
    def search(...): ...          ← 没被注册

中间。结果 `/search` 这条路由绑到了 `_to_hits` 上，
每个请求都回 **422 `Input should be a valid list`**。

它为什么能溜过当时所有的检查：

    ✅ 服务正常启动        —— 装饰器语法完全合法
    ✅ /docs 正常打开      —— OpenAPI 里 /search 照样在
    ✅ 78 条单测全绿       —— 没有一条从 HTTP 层打 /search
    ✅ 之前的实测结果有效  —— 那是**改动之前**跑的服务进程

也就是说，一个「接口全挂」的改动可以是**完全静默**的。
这个文件就是为这一件事存在的：**用最便宜的方式确认每条路由后面站着的是谁。**

⚠️ 它测的不是「业务对不对」，是「线接对了没有」——
   所以永远不该有依赖、永远该秒过。
"""

from app import main

# 期望的路由表：路径 -> 处理函数名
#
# 新增接口时**必须来这里补一行**。忘了补的后果是这条用例不会失败，
# 但也正是这个文件存在的理由 —— 至少当下次有人插错函数时，
# 现有的这些行会立刻变红。
_EXPECTED = {
    "/health": "health",
    "/usage": "usage_stats",
    "/extract": "extract",
    "/extract/batch": "extract_batch",
    "/index/build": "index_build",
    "/search": "search",
    "/search/by-image": "search_by_image",
    "/chat": "chat",
}


def _actual() -> dict[str, str]:
    """从 app 里把「路径 -> 处理函数名」抓出来。"""
    out = {}
    for route in main.app.routes:
        if hasattr(route, "methods") and not route.path.startswith(("/openapi", "/docs", "/redoc")):
            out[route.path] = route.name
    return out


def test_每条路由都挂在对的函数上():
    """⚠️ 本文件的核心断言。

    名字对不上 = 装饰器抓到别的函数上去了（就是 _to_hits 那次）。
    这种错误不会让服务起不来，只会让接口静默地全坏。
    """
    actual = _actual()
    for path, expected_name in _EXPECTED.items():
        assert path in actual, f"路由 {path} 不见了"
        assert actual[path] == expected_name, (
            f"{path} 挂到了 `{actual[path]}` 上，应该是 `{expected_name}` —— "
            f"多半是装饰器和 def 之间被插进了别的函数"
        )


def test_没有多余的路由():
    """反过来查一遍：出现了没登记的路由，说明有人加了接口却没更新上面的表。

    （这条不是为了「删掉」它，是为了**逼着人来看一眼** ——
    新接口该不该被限流、该不该出现在 /health 里，都是看的时候才会想到的问题。）
    """
    extra = set(_actual()) - set(_EXPECTED)
    assert not extra, f"这些路由没登记进 _EXPECTED：{sorted(extra)}"


def test_要烧模型的接口都接了限流():
    """漏接限流的后果和漏挂路由一样隐蔽：**「没被限流」本身不会报错**。

    这里不真的发请求（那要 Redis），只看函数的签名里有没有 `Request` ——
    因为 `_enforce_limit(request, scope)` 必须拿到它才能取客户端 IP。
    """
    import inspect

    for path in ("/extract", "/extract/batch", "/chat", "/search/by-image"):
        route = next(r for r in main.app.routes if getattr(r, "path", None) == path)
        params = inspect.signature(route.endpoint).parameters
        assert "request" in params, (
            f"{path} 的处理函数没有 request 参数 —— 那它就没法调 _enforce_limit，"
            f"这个接口现在是**不限流**的"
        )


def test_以图搜书没有被漏在限流外面():
    """⚠️ 这条是补上去的，因为它**确实漏了一整个里程碑**。

    `/search/by-image` 走的是 `/search` 那条检索路径，容易被顺手归到
    「检索接口，烧本地 CPU，不限流」那一类里 —— 但它第一步就是一次
    视觉模型调用，比文本模型慢一个数量级，后面还跟着最多 10 次完整检索。

    「没被限流」不会报错、不会变慢、测试也不会挂，只会安静地烧钱。
    所以这条专门把「它是成本接口」钉住，顺便锁死 scope 的名字 ——
    scope 拼错不会抛异常，会静默 fallback 到 extract 的限额（见 ratelimit.limit_for）。
    """
    import inspect

    route = next(
        r for r in main.app.routes if getattr(r, "path", None) == "/search/by-image"
    )
    params = inspect.signature(route.endpoint).parameters
    assert "request" in params

    src = inspect.getsource(route.endpoint)
    assert '_enforce_limit(request, "image")' in src, (
        "以图搜书必须显式用 image 这个 scope，不能靠 fallback"
    )


def test_不烧模型的接口没有被接上限流():
    """限流限的是**成本**，不是流量。

    `/search` 和 `/index/build` 用的是本地 BGE（只烧 CPU），
    给它们加限流只会让演示时点快一点就被自己拦住。

    ⚠️ `/search/by-image` **不在这条名单里** —— 名字长得像检索，但它烧模型。
        以前这里是个 `if path == "/search/by-image": continue` 的豁免，
        带着一句「将来若要限流，这里就是第一个该改的地方」。
        那个「将来」来了，豁免已经删掉，改成上面那条真正的断言。
    """
    import inspect

    for path in ("/search", "/index/build"):
        route = next(r for r in main.app.routes if getattr(r, "path", None) == path)
        params = inspect.signature(route.endpoint).parameters
        assert "request" not in params, (
            f"{path} 不该被限流（它不烧模型调用），但它现在接了 request —— "
            f"确认一下是不是有人顺手给它也加了闸门"
        )


def test_openapi能生成():
    """/docs 打得开 —— 生成 schema 时炸掉的话，整个文档页和白盒测试都没了。

    这条特别便宜（纯本地），但能挡住「Pydantic 模型写错了导致 swagger 挂掉」
    这类只在打开文档页时才暴露的问题。
    """
    schema = main.app.openapi()
    assert "/search" in schema["paths"]
    assert "/search/by-image" in schema["paths"]
    assert "post" in schema["paths"]["/search/by-image"]
    assert "/usage" in schema["paths"]
