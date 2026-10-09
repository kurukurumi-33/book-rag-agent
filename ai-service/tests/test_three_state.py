"""三态语义的出口契约（`get_post_detail`）。

## 这里锁的是简历上那条「6533 条误判」的代码级回归

源头有三种状态：**未提及 / 明确否定 / 明确肯定**。
数据库里它们分别是 `null / false / true`，是**三态**。
如果出口做一次 `bool()` 转换，`null` 和 `false` 会被压成同一个值 —— 三态变两态，
而且丢掉的正是「卖家没提」这一态。

后果（1 万条语料上实测）：**6533 条（65.3%）「未提及」会被误判成「没有笔记」**，
用户勾选「要无笔记的书」时会搜到一堆卖家其实啥都没说的书。
（220 条时代这个数是 99 条 / 45% —— 扩容后比例反而更高，说明不是小语料的偶然。）

复现：`SELECT SUM(has_notes=1), SUM(has_notes=0), SUM(has_notes IS NULL) FROM book_post;`

所以出口的口径是：`null → "未提及"`（一个**字符串**，不是 False）。
**字符串化是关键** —— 它让「未提及」在类型上就不可能跟 `false` 混淆。
"""

import pytest

from app.agent import tools


def _post(**overrides) -> dict:
    """主服务 `GET /api/posts/{id}` 返回的一条记录（字段名是 Java 的驼峰）。"""
    base = {
        "bookName": "线性代数",
        "edition": "第六版",
        "publisher": None,
        "author": None,
        "conditionDesc": "有笔记，划过重点",
        "price": 8.0,
        "hasNotes": True,
        "status": "ON_SALE",
        "rawText": "线代 同济六版 8块 有笔记 划重点",
    }
    base.update(overrides)
    return base


@pytest.fixture
def stub_post(monkeypatch):
    """把 tools 模块里的 get_post 换成假的（它是一次跨进程 HTTP 调用）。"""

    def _install(post):
        monkeypatch.setattr(tools, "get_post", lambda _pid: post)

    return _install


def test_价格null转成未标价而不是面议(stub_post):
    """「面议」是**主动的**商业表述（卖家说可以还价），
    而 null 的真实含义常常只是「这行没写价格」。
    取**更弱的断言** —— 别替卖家承诺可以还价。"""
    stub_post(_post(price=None))
    rows = tools.get_post_detail.invoke({"post_id": 209})
    assert rows[0]["price"] == "未标价"


def test_笔记null转成未提及而不是False(stub_post):
    """核心断言：**不能**变成 False。
    False 的含义是「明确说了没有笔记」，这是替卖家撒谎。"""
    stub_post(_post(hasNotes=None))
    rows = tools.get_post_detail.invoke({"post_id": 209})
    assert rows[0]["has_notes"] == "未提及"
    assert rows[0]["has_notes"] is not False


def test_明确否定保持False(stub_post):
    """卖家真的写了「无笔记」时，才给 False。三种状态各有各的出口。"""
    stub_post(_post(hasNotes=False))
    rows = tools.get_post_detail.invoke({"post_id": 1})
    assert rows[0]["has_notes"] is False


def test_明确肯定保持True(stub_post):
    stub_post(_post(hasNotes=True))
    rows = tools.get_post_detail.invoke({"post_id": 1})
    assert rows[0]["has_notes"] is True


def test_查不到返回空列表而不是None(stub_post):
    """返回 None 会让 loop.py 那句 `len(results)` 抛 TypeError，
    然后被 except 兜成一条**假的**「工具执行出错」——
    「其实是没查到」这个真相就被盖掉了。所以必须给空列表。"""
    stub_post(None)
    assert tools.get_post_detail.invoke({"post_id": 9999}) == []


def test_状态翻成中文(stub_post):
    stub_post(_post(status="SOLD"))
    assert tools.get_post_detail.invoke({"post_id": 1})[0]["status"] == "已卖出"


def test_未知状态不崩(stub_post):
    """主服务加了新状态（比如 RESERVED）而 AI 服务还没跟上时，
    不能 KeyError 崩掉，退化成「状态未知」。"""
    stub_post(_post(status="RESERVED"))
    assert tools.get_post_detail.invoke({"post_id": 1})[0]["status"] == "状态未知"
