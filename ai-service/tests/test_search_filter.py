"""混合检索的闸门与融合行为。

## 为什么值得单测

阈值不是「优化项」，是**正确性的一部分**（见 `search.py` 里那段注解）：
向量检索**永远**返回 top_k 条，哪怕库里根本没有这本书 ——
不设阈值的话，搜「量子力学」也会返回一堆《高等数学》，agent 拿这堆垃圾去答用户。

所以「低于阈值的必须被丢掉」是一条必须锁死的契约，不是可选项。

## 怎么做到不依赖真模型、真向量库、真 BM25

`search()` 里有五个外部调用：

    embedding.embed_query()     要加载 BGE 模型（几百 MB）
    vector_store.query()        要读 Chroma
    lexical.search()            要读 Chroma 重建 BM25 索引
    vector_store.get_by_ids()   要读 Chroma
    rerank.rerank()             要加载 cross-encoder（**1.1GB**）

全换成假的（或关掉），整条逻辑就能在**毫秒级、无副作用**下验证。

⚠️ `lexical.search` 必须**显式**换成假的，不能指望它「反正返回空」。
   之前漏了这一步，测试碰巧还全绿 —— 因为本机 Chroma 恰好是空的。
   换台机器（或者本机建过索引）就会有额外命中混进结果，测试莫名其妙地挂。
   **测试依赖外部世界的状态 = 迟早变成玄学失败。**

⚠️ `rerank` 同理，而且后果更重：它是**真的 1.1GB 模型**，忘了关的话
   单测会老老实实把它 load 进内存、再按真实语义重排 —— 结果是
   「测试跑 40 秒」「断言按模型的脾气时绿时红」「CI 上没下模型直接挂」。
   这份假 backend 一律把精排**关掉**：这里要验的是融合和闸门，
   精排自己的行为由 tests/test_rerank.py 用假模型单独验。
"""

import pytest

from app.services import search as search_service

# 查询向量。只要不是零向量就行（零向量算余弦会除零）。
_QUERY_VEC = [1.0, 0.0]


def _hits(*scores: float) -> list[dict]:
    """按给定分数造一批假的向量检索结果（按分数降序，模拟真实返回）。"""
    return [
        {"id": str(i), "score": s, "text": f"doc{i}", "metadata": {}}
        for i, s in enumerate(scores)
    ]


def _lex(*ids: str) -> list[dict]:
    """造一批假的 BM25 命中。search.py 只用到 id，分数/覆盖率是 lexical 内部的事。"""
    return [{"id": i, "bm25": 10.0 - n, "coverage": 1.0} for n, i in enumerate(ids)]


@pytest.fixture
def fake_backend(monkeypatch):
    """把 search() 的四个外部依赖全换成假的，返回一个「装填结果」的函数。

    注意 patch 的位置：`search.py` 里写的是 `from app.services import embedding`，
    所以它拿到的是**模块对象**，要用 `search_service.embedding` 去改。
    """

    def _install(
        dense: list[dict],
        lexical: list[dict] | None = None,
        extra_rows: list[dict] | None = None,
    ) -> None:
        monkeypatch.setattr(search_service.embedding, "embed_query", lambda _q: _QUERY_VEC)
        monkeypatch.setattr(
            search_service.vector_store, "query", lambda _vec, top_k: dense[:top_k]
        )
        monkeypatch.setattr(
            search_service.lexical,
            "search",
            lambda _q, top_k: (lexical or [])[:top_k],
        )
        # BM25 独有的候选要补 text/metadata/向量，这里给一份假的
        rows = extra_rows or []
        monkeypatch.setattr(
            search_service.vector_store,
            "get_by_ids",
            lambda ids: [r for r in rows if r["id"] in ids],
        )
        # 关掉精排。**这一步不能省** —— 见模块开头那段：不关的话
        # 测试会去 load 那个 1.1GB 的真模型，慢、还按它的脾气给结果。
        monkeypatch.setattr(search_service.settings, "rerank_enabled", False)

    return _install


# ============================================================================
# 稠密通道：余弦闸门
# ============================================================================


def test_默认阈值是扫描出来的那个值():
    """这个常量是**在 1 万条语料上扫出来的**，不是拍脑袋定的：
    scripts/eval_search.py 的阈值扫描（0.58~0.65）里，0.61 是「正例 16/16 全活」
    的上限 —— 抬到 0.62，「同济高数」就挂了。

    ⚠️ 这里**故意写死字面量**，不要改成 `== search_service._DEFAULT_MIN_SCORE`
    （那样等于什么都没测）。改这个数必须是有意识的：它意味着整个评测集要重跑、
    快照要重存。
    """
    assert search_service._DEFAULT_MIN_SCORE == 0.61


def test_低于默认阈值的被丢弃(fake_backend):
    """比阈值低一点点的必须被丢掉。

    分数用**相对**写法（`t - 0.01`）而不是写死 0.55：这条测的是
    「低于阈值就丢」这个逻辑，不是那个具体的数。写死的话每次重标阈值
    都要来改一遍测试 —— 改测试比改代码更容易改错，等于给自己埋雷。
    """
    t = search_service._DEFAULT_MIN_SCORE
    fake_backend(_hits(0.95, t - 0.01, 0.30))
    results = search_service.search("任意查询", top_k=10)
    assert [h["score"] for h in results] == [0.95]


def test_恰好等于阈值保留(fake_backend):
    """边界：实现里是 `score < min_score` 才丢，所以等于阈值应当**保留**。
    这种 ±1 的边界是最容易写错的地方，专门测。"""
    fake_backend(_hits(search_service._DEFAULT_MIN_SCORE))
    assert len(search_service.search("q", top_k=10)) == 1


def test_正好卡在缝上的两个分数(fake_backend):
    """把实测里那对关键分数塞进来（9997 条语料测的）：

        0.6034  「核工程」→《Java核心技术》  最高的**能拦住**的越界查询 → 该丢
        0.6187  「同济高数」                最低正例                 → 该留
        缝 = 0.0153，阈值 0.61 卡在中间

    ⚠️ 别把这条读成「阈值很安全」。缝的另一侧是拦不住的那一批 ——
    教育学 0.6158 / 环境工程 0.6199 / 国际关系 0.6233 / 纺织 0.6959，
    **全落在正例区间里**。所以真实情况是「正负样本已经重叠，只是刚好还剩一条能拦」，
    这也是必须上 rerank 的证据（见 scripts/eval_search.py 的 KNOWN_LEAKS）。
    """
    fake_backend(_hits(0.6187, 0.6034))
    scores = [h["score"] for h in search_service.search("q", top_k=10)]
    assert scores == [0.6187]


def test_全部低于阈值时返回空列表(fake_backend):
    """负样本场景：库里确实没有的查询应当返回**空**。
    空列表是正常结果，不是错误 —— agent 据此回答「没找到」。

    ⚠️ 别拿「量子力学」当这个例子：它在 220 条的老语料里没有，
    1 万条的语料里**有 12 条** —— 语料一变，负样本就不是负样本了。
    """
    fake_backend(_hits(0.48, 0.45, 0.31))
    assert search_service.search("考古", top_k=10) == []


def test_显式min_score覆盖默认值(fake_backend):
    """eval_search.py 的 --min-score 走的就是这条路径。
    调参时必须能覆盖服务端默认值，否则测的不是你想测的阈值。"""
    fake_backend(_hits(0.90, 0.70, 0.20))
    results = search_service.search("q", top_k=10, min_score=0.15)
    assert [h["score"] for h in results] == [0.90, 0.70, 0.20]


# ============================================================================
# 稀疏通道：BM25 独有的候选怎么进来
# ============================================================================


def test_bm25能捞回余弦不够高的书(fake_backend):
    """⚠️ 这条是「BM25 到底有没有用」的核心证据，别删。

    id=9 的余弦只有 0.42（远低于阈值），**稠密通道亲手把它扔了**。
    但 BM25 靠精确词面把它捞了回来（比如查「汤家凤1800」—— BGE 不认识
    "1800" 这个编号，余弦给不高，但字面完全匹配）。

    它必须出现在最终结果里，否则加 BM25 就白加了。

    同时验证它的 score 是**现算的余弦**（0.0，因为假向量正交），
    而不是什么 BM25 分数 —— 对外只有一个 score 语义。
    """
    fake_backend(
        _hits(0.90),
        _lex("9"),
        extra_rows=[{"id": "9", "text": "汤家凤1800题", "metadata": {}, "embedding": [0.0, 1.0]}],
    )
    results = search_service.search("汤家凤1800", top_k=10)
    # 两个通道各出一个，融合分都是 1/61，打平 —— 这里只断言「都在」，
    # 平局时的先后由 test_两个通道都命中的排最前 那条单独锁（不靠打平来测排序）
    assert {h["id"] for h in results} == {"0", "9"}
    only_bm25 = next(h for h in results if h["id"] == "9")
    assert only_bm25["text"] == "汤家凤1800题"
    assert only_bm25["score"] == pytest.approx(0.0)  # 正交 → 余弦 0


def test_两个通道都命中的排最前(fake_backend):
    """RRF 的意义就在这里：**排名相加**。

        id=0  稠密第 1 + 稀疏第 2   →  1/61 + 1/62 ≈ 0.03252   ← 最前
        id=5  只有稀疏第 1          →  1/61       ≈ 0.01639
        id=1  只有稠密第 2          →  1/62       ≈ 0.01613   ← 最后

    「两边都靠前」胜过「单边第一」，这正是融合想要的效果。
    """
    fake_backend(
        _hits(0.90, 0.80),
        _lex("5", "0"),
        extra_rows=[{"id": "5", "text": "doc5", "metadata": {}, "embedding": [0.0, 1.0]}],
    )
    results = search_service.search("q", top_k=10)
    assert [h["id"] for h in results] == ["0", "5", "1"]


def test_索引快照与向量库不一致时不炸(fake_backend):
    """BM25 索引建好之后，向量库里那条被删了 ——
    BM25 通道仍然认得它，但 get_by_ids 查不回来。

    这时必须**跳过**它继续返回别人，而不是抛 KeyError 把整个 /search 打成 500。
    一条脏数据毁掉整个接口是最典型的可用性事故。
    """
    fake_backend(_hits(0.90), _lex("999"), extra_rows=[])  # 999 查不回来
    results = search_service.search("q", top_k=10)
    assert [h["id"] for h in results] == ["0"]


def test_没有BM25命中时退化成纯向量检索(fake_backend):
    """BM25 索引没建好 / 查询全是口语说法时，lexical 返回空 ——
    这时结果必须和加 BM25 之前一模一样，不能因为引入新通道把老行为搞坏。"""
    fake_backend(_hits(0.90, 0.80, 0.70))
    results = search_service.search("q", top_k=10)
    assert [h["id"] for h in results] == ["0", "1", "2"]


def test_两个通道都空时返回空(fake_backend):
    fake_backend(_hits(0.30), _lex())  # 全部低于阈值、BM25 也没命中
    assert search_service.search("量子力学", top_k=10) == []


# ============================================================================
# 返回结构与 top_k
# ============================================================================


def test_返回带rrf_score(fake_backend):
    """rrf_score 是**真正决定排序**的那个字段。
    它数值本身没有意义（都是零点零几），只看相对大小 ——
    下游（前端、调试）不能拿它跟 score 比。"""
    fake_backend(_hits(0.90))
    hit = search_service.search("q", top_k=10)[0]
    assert hit["rrf_score"] == pytest.approx(1 / 61)
    assert set(hit) >= {"id", "score", "rrf_score", "text", "metadata"}


def test_top_k限制最终条数(fake_backend):
    """top_k 是**最终返回**几条。

    注意它现在和「向向量库要几条」解耦了：召回池固定取
    max(top_k, _RECALL_N)，比 top_k 大 —— 池子刚好等于 top_k 的话，
    两个通道看到的是同一批，融合就退化成排序抖动、没意义了。
    """
    # 五个分数都要**在阈值之上** —— 之前这里写的是 (0.9, 0.8, 0.7, 0.6, 0.5)，
    # 阈值从 0.60 抬到 0.61 之后那条 0.6 被闸门滤掉了，top_k=4 只能拿到 3 条，
    # 测试就挂了。挂的是**测试数据过期**，不是代码错 —— 这种假失败最容易带偏排查方向。
    fake_backend(_hits(0.95, 0.90, 0.85, 0.80, 0.75))
    assert len(search_service.search("q", top_k=2)) == 2
    assert len(search_service.search("q", top_k=4)) == 4


def test_召回池不小于top_k(monkeypatch):
    """用户要 50 条时不能只召 20 条 —— 否则明明库里有 30 条符合，
    却因为召回池只有 20 而少给。"""
    seen = {}

    def spy(_vec, top_k):
        seen["top_k"] = top_k
        return []

    monkeypatch.setattr(search_service.embedding, "embed_query", lambda _q: _QUERY_VEC)
    monkeypatch.setattr(search_service.vector_store, "query", spy)
    monkeypatch.setattr(search_service.lexical, "search", lambda _q, top_k: [])

    search_service.search("q", top_k=50)
    assert seen["top_k"] >= 50
