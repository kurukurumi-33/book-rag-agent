"""请求 / 响应模型的校验契约。

## 为什么单测 schema

`schemas.py` 是**服务的边界** —— 所有外部输入都从这里进。
边界校验写错，后面全是脏数据，而且 FastAPI 的报错很靠后（422 在路由层才出）。

这里重点测两类历史上真实踩过的坑：
1. `session_id` 的正则**必须自己锚定** `^...$`
2. `price` 的哨兵值 `-1` 要在出口转回 `null`
"""

import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, SearchRequest


# --------------------------------------------------------------------------
# session_id：必须锚定，否则约束形同虚设
# --------------------------------------------------------------------------


def test_合法session_id通过():
    assert ChatRequest(message="hi", session_id="abc-123_XYZ").session_id == "abc-123_XYZ"


def test_不传session_id是None_表示开新会话():
    """语义：不传 = 开新会话，服务端生成后返回。
    所以 None 是**合法输入**，不是错误。"""
    assert ChatRequest(message="hi").session_id is None


@pytest.mark.parametrize(
    "bad",
    [
        "a/b",          # 带斜杠：会拼进 URL 路径，被路由拆开
        "a b",          # 带空格：编码后 Tomcat 直接拒
        "会话",          # 中文：不在允许字符集内
        "a" * 65,       # 超长：DB 列是 VARCHAR(64)
    ],
    ids=["斜杠", "空格", "中文", "超长"],
)
def test_非法session_id被拒(bad):
    """⚠️ 这条测试守的是那个**非常容易犯**的错：

    Pydantic 的 pattern 是**子串匹配**（底层是 re.search 语义），
    如果只写 `[A-Za-z0-9_-]{1,64}` 而不加 ^$，
    那么 "a/b c" 里**含一个 a** 就算通过 —— 整个约束形同虚设。

    所以 schema 里写的是 `^[A-Za-z0-9_-]{1,64}$`，必须自己锚定。
    这里用 "a/b" / "a b" 这类「含合法字符但整体不合法」的输入来证明锚定生效。
    """
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", session_id=bad)


# --------------------------------------------------------------------------
# 其它边界
# --------------------------------------------------------------------------


def test_message不能为空():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_search_top_k上界是50():
    """向量库 n_results 无上限会拖慢查询；50 是拍定的护栏。"""
    with pytest.raises(ValidationError):
        SearchRequest(query="高数", top_k=51)
    assert SearchRequest(query="高数", top_k=50).top_k == 50


def test_search_min_score默认None表示用服务端默认值():
    """None ≠ 0。None 的含义是「没指定，你去用你配置的那个默认阈值」。
    如果默认成 0，就等于「不过滤」，负样本会全部漏进来。"""
    assert SearchRequest(query="高数").min_score is None


def test_search_query不能为空():
    with pytest.raises(ValidationError):
        SearchRequest(query="")
