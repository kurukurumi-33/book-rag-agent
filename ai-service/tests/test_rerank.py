"""精排（cross-encoder）的行为契约。

## 这里锁死的三件事

1. **`score` 不被改写** —— 接口对外承诺 `score` 是余弦相似度。
   精排分走**新字段** `rerank_score`。混用会让调用方没法判断哪个数是哪个。
2. **关掉精排时零副作用** —— 原样返回、不加字段、不碰模型。
   这条是给「rerank 没配好时系统还能用」兜底的。
3. **加载失败必须降级、不能抛** —— rerank 是可选的锦上添花，
   它挂了不该把整条 `/search` 打挂。但**也不能悄悄降级**，
   所以 `status()` 要能把「其实没在工作」说出来（见 test_加载失败时状态可查）。

## 怎么做到不加载那个 1.1GB 的真模型

`rerank_service.get_model` 换成返回一个假对象（只要有 `.predict`）。
被测的是**排序、拷贝、池子边界、过滤**这些逻辑，不是 BERT 本身准不准 ——
后者靠 `scripts/tune_rerank.py` 在真语料上量，不是单测的活。
"""

import pytest

from app.services import rerank as rerank_service


class _FakeModel:
    """假 cross-encoder。

    按文本里的数字给分，方便断言顺序 —— 比「按内容语义」可预测得多。
    `predict` 收的是 [(query, doc), ...]，返回与之一一对应的原始 logit。
    """

    def __init__(self, logits: dict[str, float]) -> None:
        self._logits = logits
        self.seen: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.seen.append(list(pairs))
        return [self._logits.get(doc, 0.0) for _q, doc in pairs]


def _cands(*ids: str) -> list[dict]:
    """造一批粗排候选，text 就是 id（方便假模型查表）。"""
    return [
        {"id": i, "score": 0.9 - n * 0.01, "text": i, "metadata": {}, "rrf_score": 0.03}
        for n, i in enumerate(ids)
    ]


@pytest.fixture
def with_model(monkeypatch):
    """装上假模型、开着精排，返回那个假模型方便断言它收到了什么。"""

    def _install(logits: dict[str, float]) -> _FakeModel:
        model = _FakeModel(logits)
        monkeypatch.setattr(rerank_service, "get_model", lambda: model)
        monkeypatch.setattr(rerank_service.settings, "rerank_enabled", True)
        return model

    return _install


# ── sigmoid ──────────────────────────────────────────────────────────────


def test_sigmoid把logit压到0到1之间():
    assert rerank_service._sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < rerank_service._sigmoid(-5.0) < 0.5
    assert 0.5 < rerank_service._sigmoid(5.0) < 1.0


def test_sigmoid在极端值上不溢出():
    """直接写 exp(-x) 的话 x = -1000 会 OverflowError。

    而 bge-reranker 对**完全不相关**的输入真的会输出很负的 logit
    （库里那一堆「核工程 → Java核心技术」就是），所以这条不是理论洁癖。
    """
    assert rerank_service._sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-9)
    assert rerank_service._sigmoid(1000.0) == pytest.approx(1.0)
    assert 0.0 <= rerank_service._sigmoid(-1e9) <= 1.0


# ── 降级路径 ──────────────────────────────────────────────────────────────


def test_关掉精排时原样返回(monkeypatch):
    monkeypatch.setattr(rerank_service.settings, "rerank_enabled", False)
    # get_model 故意设成「一调用就炸」—— 关掉时**根本不该碰模型**
    monkeypatch.setattr(rerank_service, "get_model", lambda: pytest.fail("不该加载模型"))
    out = rerank_service.rerank("q", _cands("a", "b", "c"), top_k=2)
    assert [c["id"] for c in out] == ["a", "b"]
    assert all("rerank_score" not in c for c in out)


def test_模型加载失败时不抛异常只降级(monkeypatch):
    monkeypatch.setattr(rerank_service.settings, "rerank_enabled", True)
    monkeypatch.setattr(rerank_service, "get_model", lambda: None)
    out = rerank_service.rerank("q", _cands("a", "b", "c"), top_k=2)
    # 降级 = 拿粗排的顺序，不报错
    assert [c["id"] for c in out] == ["a", "b"]


def test_空候选返回空列表(monkeypatch):
    monkeypatch.setattr(rerank_service, "get_model", lambda: pytest.fail("不该加载模型"))
    assert rerank_service.rerank("q", [], top_k=5) == []


def test_加载失败时状态可查(monkeypatch):
    """降级必须**看得见** —— 这正是本项目反复批判的「静默失败」。

    服务照样返回 ok（检索没坏），但 /health 里的 rerank.failed 要说实话。
    """
    monkeypatch.setattr(rerank_service.settings, "rerank_enabled", True)
    monkeypatch.setattr(rerank_service, "_model", None)
    monkeypatch.setattr(rerank_service, "_load_failed", False)
    monkeypatch.setattr(
        rerank_service, "_model_source", lambda: "/definitely/not/a/model"
    )
    assert rerank_service.get_model() is None
    st = rerank_service.status()
    assert st["failed"] is True
    assert st["loaded"] is False


def test_加载失败后不再反复重试(monkeypatch):
    """失败一次就记住 —— 否则每个请求都要卡在「重新加载 → 又失败」上。"""
    monkeypatch.setattr(rerank_service, "_model", None)
    monkeypatch.setattr(rerank_service, "_load_failed", True)
    calls = []
    monkeypatch.setattr(rerank_service, "_model_source", lambda: calls.append(1))
    assert rerank_service.get_model() is None
    assert calls == []


# ── 精排本身 ──────────────────────────────────────────────────────────────


def test_按精排分重排(with_model):
    with_model({"a": -5.0, "b": 8.0, "c": 1.0})
    out = rerank_service.rerank("q", _cands("a", "b", "c"), top_k=3)
    assert [c["id"] for c in out] == ["b", "c", "a"]


def test_余弦score原样保留(with_model):
    """精排分走新字段，**绝不覆盖 score**。

    覆盖了的话，`/search` 响应的 score 就有时是余弦、有时是 logit ——
    调用方按 score 排序或设阈值都会出错，而且**不会报错**。
    """
    with_model({"a": -5.0, "b": 8.0})
    before = _cands("a", "b")
    out = rerank_service.rerank("q", before, top_k=2)
    by_id = {c["id"]: c for c in out}
    assert by_id["a"]["score"] == 0.9
    assert by_id["b"]["score"] == 0.89
    assert by_id["b"]["rerank_score"] == pytest.approx(rerank_service._sigmoid(8.0))


def test_不改动调用方手里的原对象(with_model):
    """返回的是浅拷贝。原地改的话，粗排那份候选列表会被悄悄污染 ——
    调试时前后两个变量指向同一块内存，最难查的那种 bug。"""
    with_model({"a": 8.0, "b": -8.0})
    original = _cands("a", "b")
    snapshot = [dict(c) for c in original]
    rerank_service.rerank("q", original, top_k=2)
    assert original == snapshot


def test_精排分是sigmoid后的值(with_model):
    with_model({"a": 0.0})
    out = rerank_service.rerank("q", _cands("a"), top_k=1)
    assert out[0]["rerank_score"] == pytest.approx(0.5)


# ── 候选池边界 ────────────────────────────────────────────────────────────


def test_只精排池子内的候选(with_model, monkeypatch):
    """池子大小由 rerank_pool 决定，不是 top_k。

    ⚠️ 这条和「粗排不能提前截到 top_k」是配套的：粗排截早了，
    精排能看到的就只剩最终要返回的条数，「重排」退化成「原地不动」。
    """
    monkeypatch.setattr(rerank_service.settings, "rerank_pool", 2)
    model = with_model({"a": -5.0, "b": 8.0, "c": 100.0, "d": 100.0})
    out = rerank_service.rerank("q", _cands("a", "b", "c", "d"), top_k=4)
    # 假模型只该收到前 2 条
    assert [doc for _q, doc in model.seen[0]] == ["a", "b"]
    # c/d 没被精排过，原样接在后面（它们的 logit 再高也不该冒头）
    assert [c["id"] for c in out] == ["b", "a", "c", "d"]
    assert "rerank_score" not in out[2]


def test_池子外的候选不会被丢掉(with_model, monkeypatch):
    """它们没被精排，但也没被否定 —— 直接扔掉会让「要 50 条」的拿不满。"""
    monkeypatch.setattr(rerank_service.settings, "rerank_pool", 1)
    with_model({"a": 8.0})
    out = rerank_service.rerank("q", _cands("a", "b", "c"), top_k=3)
    assert [c["id"] for c in out] == ["a", "b", "c"]


# ── 过滤（默认关，但逻辑要在）────────────────────────────────────────────


def test_min_score过滤掉低分候选(with_model):
    with_model({"a": 8.0, "b": -8.0, "c": 1.0})
    out = rerank_service.rerank("q", _cands("a", "b", "c"), top_k=3, min_score=0.6)
    assert [c["id"] for c in out] == ["a", "c"]
    assert all(c["rerank_score"] >= 0.6 for c in out)


def test_默认不过滤只重排(monkeypatch):
    """**默认 min_score=None 是标定过的决定，不是忘了传。**

    正例那一侧的 rerank_score 只有 0.6987~0.7311 这么宽，在里面切一刀
    就是过拟合（详见 config.py 的 rerank_min_score）。所以这里锁的是
    「不给 min_score 就一条都不许丢」。
    """
    monkeypatch.setattr(rerank_service.settings, "rerank_pool", 10)
    model = _FakeModel({"a": -100.0, "b": -100.0})
    monkeypatch.setattr(rerank_service, "get_model", lambda: model)
    monkeypatch.setattr(rerank_service.settings, "rerank_enabled", True)
    out = rerank_service.rerank("q", _cands("a", "b"), top_k=5)
    assert len(out) == 2
    assert all(c["rerank_score"] < 0.01 for c in out)
