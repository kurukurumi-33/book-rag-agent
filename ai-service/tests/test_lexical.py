"""BM25 通道的闸门测试。

## 为什么用**合成语料**而不是真库

真库要起 MySQL + Spring Boot + 建索引，跑一次十几秒，还是集成测试。
但这里要验的**根本不是「检索准不准」**，而是「闸门逻辑对不对」——
闸门是纯代码，喂什么语料它就该怎么判。所以自己造一份最小的语料，
把真库里那几个关键 case（法学概论 / 汤家凤1800 / 考研数学用的）复刻进来，
0.1 秒跑完，而且**断言是确定的**（真库里书多了，BM25 的 IDF 会变，
分数会漂，但闸门的判定不该漂）。

## 这份语料在防什么

| 用例 | 期望 | 防的是 |
|---|---|---|
| 法学概论 | 返回空 | **假阳性**：只共享「概论」二字就放行（最危险，它绕过余弦闸门） |
| 汤家凤1800 | 命中 | **假阴性**：真匹配被闸门误杀 |
| 考研数学用的 | 命中 | **假阴性**：单字虚词「用/的」把覆盖率分母撑大 |

后两条是同一类问题（闸门太严），但成因不同，所以分开锁。
"""

import pytest

from app.services import lexical, vector_store


@pytest.fixture
def corpus(monkeypatch):
    """把 vector_store.all_documents 换成一份内存语料，并重置 BM25 单例。

    这些文档就是 build_index 时喂给向量库的 compose_text() 结果
    （书名 + 版次 + 作者 + 出版社），所以形状是真实的。

    ⚠️ 必须重置模块级单例：_bm25 是**进程级**缓存，上一个用例建好的索引
    会原样留在下一个用例里。不重置的话第二个用例查的是第一个用例的语料，
    测试之间就互相污染了。
    """
    docs = [
        "高等数学 第七版 同济大学出版社",
        "毛泽东思想和中国特色社会主义理论体系概论",
        "考研数学复习全书",
        "汤家凤1800题",
        "数据结构 C语言版 严蔚敏 清华大学出版社",
    ]
    ids = [str(i) for i in range(1, len(docs) + 1)]

    def fake_all_documents():
        return ids, docs, [{} for _ in docs]

    monkeypatch.setattr(vector_store, "all_documents", fake_all_documents)

    # 重置单例 —— 下次 search() 会从这份假语料重建
    lexical._bm25 = None
    lexical._ids = []
    lexical._term_sets = []
    yield docs
    # 用例跑完再清一遍，别把假语料留给后面的测试文件
    lexical._bm25 = None
    lexical._ids = []
    lexical._term_sets = []


def names(hits, docs):
    """把命中的 id 换成人看得懂的书名。"""
    return [docs[int(h["id"]) - 1] for h in hits]


# ============================================================================
# 假阳性：闸门必须挡住的
# ============================================================================


def test_法学概论不泄漏到毛概(corpus):
    """全组最难的负例。

    「法学概论」切出 ['法学', '概论']，而库里那本毛概书含「概论」——
    BM25 会认为这是一次命中（分数 > 0）。但这是**字面重叠，不是语义相近**：
    用户要的是法学书，返回一本毛概书是纯噪声。

    向量通道靠余弦 0.5902 < 0.60 拦得住；BM25 通道没有绝对阈值可用
    （分数无上界），只能靠覆盖率：1/2 = 0.5 < 0.6 → 挡掉。
    """
    hits = lexical.search("法学概论", top_k=10)
    assert hits == [], f"假阳性！返回了 {names(hits, corpus)}"


def test_库里根本没提到的词返回空(corpus):
    """「量子力学」库里一个字都没有 → BM25 分数全是 0 → 闸门①挡掉。"""
    assert lexical.search("量子力学", top_k=10) == []


def test_覆盖率的分母是查询词数不是文档词数(corpus):
    """把「法学概论」的原理单独抽出来验。

    分子 = 命中的查询词数，分母 = **查询词总数**（不是文档词总数）。
    文档有一堆词、查询只有两个词，那就按两个词算。
    """
    # 两个词都命中 → 2/2 = 1.0，放行
    assert lexical.search("高等数学 第七版", top_k=10) != []
    # 只命中一个 → 1/2 = 0.5，挡掉
    assert lexical.search("高等数学 线性代数", top_k=10) == []


def test_同义词在词面上零重叠(corpus):
    """⚠️ 这条锁的是 BM25 的**先天缺陷**，不是 bug —— 写在这里是为了让
    「为什么 BM25 只能当补充、不能替代向量」有个可执行的证据。

    实测：jieba 把「同济大学」切成**一个** token，和单独的「同济」**不相等**。

        '同济 出版社'  → ['同济', '出版社']
        '同济大学出版社' → ['同济大学', '出版社']     ← 共享的只有「出版社」

    所以查「同济 出版社」的覆盖率是 1/2 = 0.5，被挡掉 —— 尽管人一眼就
    知道这两句话说的是同一个出版社。

    **这正是 BM25 的死穴**：它只认字面完全相同。人脑认为「同济」和
    「同济大学」是一回事，BM25 认为毫无关系。所以真实查询「同济版高数」
    在词法通道一定是 0 命中 —— 没关系，向量通道会把它捞回来。

    （如果哪天真要救这个，方向是给 jieba 加自定义词典把「同济大学」
    也切成「同济」，但那会引入别的副作用。现阶段不值得。）
    """
    assert lexical.search("同济 出版社", top_k=10) == []


# ============================================================================
# 假阴性：闸门不能误杀的
# ============================================================================


def test_汤家凤1800能命中(corpus):
    """BM25 存在的**理由**就是这种查询：

    「1800」是编号，在向量空间里没有区分度（模型没见过几次），
    但词面上它是唯一标识 —— 这正是向量检索的盲区、BM25 的主场。

    切出 ['汤家凤', '1800']，两个词都在同一本书里 → 覆盖率 1.0 → 放行。
    """
    hits = lexical.search("汤家凤1800", top_k=10)
    assert names(hits, corpus)[:1] == ["汤家凤1800题"]


def test_单字虚词不稀释覆盖率(corpus):
    """⚠️ 这条是第二版修正的回归测试，别删。

    「考研数学用的」切出 ['考研', '数学', '用', '的']。
    库里那本《考研数学复习全书》只含「考研」「数学」——
    如果拿**原始 token** 算覆盖率：2/4 = 0.5 < 0.6 → **误杀一个正例**。

    修正：算覆盖率前先剔掉单字（见 lexical._content_terms）。
    剔掉后 {考研, 数学} → 2/2 = 1.0 → 正确放行。

    这个 bug 是跑 jieba 分词探针时**实测发现的**，不是推演出来的 ——
    写检索代码时「以为分词完就能直接比」是最容易踩的坑。
    """
    hits = lexical.search("考研数学用的", top_k=10)
    assert names(hits, corpus)[:1] == ["考研数学复习全书"]


def test_全是单字时不会除零(corpus):
    """查询里一个多字词都没有（比如只查「书」）时，
    _content_terms 的分母会变成 0 —— 得退回用原始 token，不能抛 ZeroDivisionError。
    """
    lexical.search("书", top_k=10)  # 不抛就算过


# ============================================================================
# 边界
# ============================================================================


def test_空查询和纯空白返回空(corpus):
    assert lexical.search("", top_k=10) == []
    assert lexical.search("   ", top_k=10) == []


def test_top_k生效(corpus):
    """「数学」能命中高数/考研数学两本 → top_k=1 时只能回 1 条。"""
    hits = lexical.search("数学", top_k=1)
    assert len(hits) == 1


def test_返回结构对得上(corpus):
    """外部（search.py）要做 RRF 融合，只依赖 id 和自己的排名，
    外加 coverage 做排查。字段名写错会让融合静默失效。"""
    hits = lexical.search("汤家凤1800", top_k=10)
    assert hits
    for h in hits:
        assert set(h) == {"id", "bm25", "coverage"}
        assert isinstance(h["id"], str)
        assert isinstance(h["bm25"], float)
        assert 0.0 <= h["coverage"] <= 1.0


def test_结果按bm25降序(corpus):
    hits = lexical.search("数学", top_k=10)
    scores = [h["bm25"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_空语料不炸(monkeypatch):
    """库里一条数据都没有时（还没建索引），BM25Okapi 会抛。
    闸门得先判断再建，让 search() 安静地返回空 —— 不然 /search 接口直接 500。"""
    monkeypatch.setattr(vector_store, "all_documents", lambda: ([], [], []))
    lexical._bm25 = None
    lexical._ids = []
    lexical._term_sets = []
    assert lexical.search("高等数学", top_k=10) == []


# ============================================================================
# 分词与实词提取（直接测内部函数，因为这里是所有坑的源头）
# ============================================================================


def test_内容词提取剔掉单字():
    assert lexical._content_terms(["考研", "数学", "用", "的"]) == {"考研", "数学"}


def test_内容词全单字时退回原样():
    assert lexical._content_terms(["书"]) == {"书"}


def test_分词丢掉空白():
    """jieba 把空格也切成 token（'高等数学 第七版' → ['高等数学', ' ', '第七版']），
    不滤掉的话空 token 会进 _bm25 的语料，还可能污染覆盖率。"""
    tokens = lexical._tokenize("高等数学 第七版")
    assert " " not in tokens
    assert "" not in tokens
    assert "高等数学" in tokens
