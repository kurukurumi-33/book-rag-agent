"""compose_text() 的行为契约。

## 为什么值得单测

`compose_text()` 是整条检索链路的地基：它决定**什么东西会被向量化**。
拼错一个字段，召回就被污染，而且这种污染很隐蔽 —— 分数只是低了一点点，
你不会立刻发现是拼法的问题（会以为是模型不行）。

这里的每一条断言都对应 `app/services/search.py` 注释里的一次**实测结论**，
不是凭空写的规则。锁住它们 = 防止以后有人「顺手把价格也拼上去」。
"""

from app.services.search import compose_text


def test_只拼四个身份字段():
    """书名 + 版次 + 作者 + 出版社 —— 这四个才回答『这是哪本书』。"""
    post = {
        "bookName": "高等数学",
        "edition": "第七版",
        "author": "同济大学数学系",
        "publisher": "高等教育出版社",
    }
    assert compose_text(post) == "高等数学 第七版 同济大学数学系 高等教育出版社"


def test_价格_成色_原文一律不进检索文本():
    """实测结论：书名在检索文本里重复出现反而**拉低**分数
    （0.6258 → 0.6054 → 0.5953），因为自注意力让重复的 token 互相 attend、
    向量方向被带偏。所以价格、成色、原文（rawText）都不拼。

    原文尤其危险：它包含书名之外的杂讯，会把其他书名的相似度抬上来。
    """
    post = {
        "bookName": "高等数学",
        "price": 40.0,
        "conditionDesc": "九成新，有笔记",
        "rawText": "高数同济七版 40 有笔记 可小刀",
        "hasNotes": True,
    }
    assert compose_text(post) == "高等数学"


def test_空字段跳过_不留占位符():
    """None 必须**直接跳过**，不能变成空串或 "None" 占位。

    占位会毒化检索：一个孤零零的 "None" token 会混进向量，
    让所有记录的向量都被同一份噪声污染。
    """
    post = {"bookName": "高等数学", "edition": None, "author": None, "publisher": None}
    assert compose_text(post) == "高等数学"


def test_全空返回空串():
    """整条都没有可用字段时返回空串（调用方靠它判断『这条不能进索引』）。"""
    assert compose_text({}) == ""


def test_只有版次没有书名也照拼():
    """抽取不出书名时 build_index 会跳过这条 —— 但那是调用方的决定，
    compose_text 本身不做判断，只做拼接。职责单一，好测。"""
    assert compose_text({"bookName": None, "edition": "第六版"}) == "第六版"
