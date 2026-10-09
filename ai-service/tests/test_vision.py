"""以图搜书（M2.6）的行为契约。

## 这里锁死的五件事

1. **没配 key 时是「可识别的降级」，不是崩溃。** 别人 clone 下来没填
   `VISION_API_KEY`，服务照样起、别的接口照样用，只有这个接口回 503 并说清缺什么。
2. **图片入参的校验**：空 / 超长 / 本地文件路径一律拒掉。
   最后一条是安全边界，不是格式洁癖 —— 收本地路径等于开了个读任意文件的口子。
3. **多行输出要变成一串书名，不能只留第一行。** 真实图片常常是一排书脊，
   一张图十几本书。旧版 `splitlines()[0]` 会把第 2 本往后全切掉 ——
   这条契约就是为了不让那个写法回来。
4. **脏行要淘汰，好行一本都不能丢。** 模型爱给每行加序号、爱在末尾补一行
   「作者：xxx」，这些要认出来；但「5年高考3年模拟」这种以数字开头的真书名
   不能被当成序号切坏。⚠️ 书脊场景下脏行的形态比想象的多 ——
   署名、出版社、「学生用书」这类标注、图上贴的水印，**都不带冒号**，
   所以只防「作者：」是防不住的（那批规则是被一张真实书架照片逼出来的，
   见文件后半段「书脊上的『不是书名』的东西」）。
5. **识别不出书时返回空列表，不能拿「无」去检索** —— 那会命中一堆书名里带
   「无」的书，比返回空更糟：用户会以为识别错了。

## 为什么全程不碰真模型

视觉调用要钱、要网、还要 key。被测的是**我们这个文件的逻辑**（校验、归一化、
降级分支），不是 GLM-4V 的识别准不准 —— 后者只有一个判断标准：
拿真书图跑一遍看结果，那是手工验收，不是单测。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.services import vision

_DATA_URL = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ=="


class _FakeReply:
    def __init__(self, content):
        self.content = content


class _FakeVisionLLM:
    """假视觉模型：记下收到的消息，回一段预设内容。"""

    def __init__(self, content) -> None:
        self._content = content
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return _FakeReply(self._content)


@pytest.fixture
def configured(monkeypatch):
    """把「已配 key」和「假模型」都装好，返回装假模型的钩子。"""
    monkeypatch.setattr(vision.settings, "vision_api_key", "test-vision-key")

    def _install(content):
        fake = _FakeVisionLLM(content)
        monkeypatch.setattr(vision, "get_vision_llm", lambda: fake)
        return fake

    return _install


# ---- 降级路径 ---------------------------------------------------------------


def test_没配key时返回None而不是抛异常(monkeypatch):
    """get_vision_llm() 必须返回 None，不能抛。

    抛的话调用方就只能靠 try/except 猜，而「没配 key」和「key 失效」
    是两种完全不同的故障，得能分开处理。
    """
    monkeypatch.setattr(vision.settings, "vision_api_key", "")
    assert vision.get_vision_llm() is None


def test_没配key时识别报RuntimeError(monkeypatch):
    """降级要**可见**：不能返回空列表让调用方以为「图里没有书」。"""
    monkeypatch.setattr(vision.settings, "vision_api_key", "")
    with pytest.raises(RuntimeError, match="VISION_API_KEY"):
        vision.extract_books_from_image(_DATA_URL)


def test_没配key时参数错误仍然报参数错(monkeypatch):
    """⚠️ 校验顺序的回归用例（我第一版写反了，被实测抓出来）。

    没配 key 时，一个**非法的 image** 必须报 ValueError（→ 接口 400），
    而不是 RuntimeError（→ 503）。因为非法参数是**这次请求自己的问题**，
    配不配 key 都改变不了这个判断；报成 503 会让调用方去查服务端配置，
    方向完全错了。
    """
    monkeypatch.setattr(vision.settings, "vision_api_key", "")
    with pytest.raises(ValueError, match="data URL|直链"):
        vision.extract_books_from_image("C:/Windows/win.ini")


def test_status能反映出没配key(monkeypatch):
    monkeypatch.setattr(vision.settings, "vision_api_key", "")
    assert vision.status()["configured"] is False

    monkeypatch.setattr(vision.settings, "vision_api_key", "k")
    assert vision.status()["configured"] is True


# ---- 入参校验 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_空图片直接拒掉(bad, configured):
    configured("高等数学")
    with pytest.raises(ValueError, match="不能为空"):
        vision.extract_books_from_image(bad)


def test_本地文件路径拒掉(configured):
    """🔒 安全边界：接受路径等于开了个读任意文件的口子。"""
    configured("高等数学")
    for path in ["C:/Windows/win.ini", "/etc/passwd", "./a.jpg", "file:///etc/passwd"]:
        with pytest.raises(ValueError, match="data URL|直链"):
            vision.extract_books_from_image(path)


def test_超大图片拒掉(configured, monkeypatch):
    configured("高等数学")
    monkeypatch.setattr(vision, "_MAX_IMAGE_CHARS", 100)
    with pytest.raises(ValueError, match="太大"):
        vision.extract_books_from_image(_DATA_URL + "A" * 200)


def test_http直链可以过(configured):
    fake = configured("高等数学")
    assert vision.extract_books_from_image("https://example.com/a.jpg") == ["高等数学"]
    assert len(fake.seen) == 1


# ---- 单本书名的归一化 --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("高等数学", ["高等数学"]),
        ("《高等数学》", ["高等数学"]),           # 带书名号
        ("  《线性代数》  ", ["线性代数"]),        # 带空白
        ("高等数学。", ["高等数学"]),             # 带句号
        ("书名：高等数学", ["高等数学"]),           # 标签后面就是书名，留着
        ("书名：《高等数学》", ["高等数学"]),        # 前缀 + 书名号一起来
        ('"高等数学"', ["高等数学"]),             # 带引号
        ("C++ Primer（第5版）", ["C++ Primer（第5版）"]),  # 中间括号是书名的一部分，不能动
    ],
)
def test_书名归一化(raw, expected, configured):
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == expected


@pytest.mark.parametrize("raw", ["无", "没有", "None", "null", "无法识别", " 无 ", ""])
def test_识别不出书时返回空列表(raw, configured):
    """图里没书 → []。

    ⚠️ 关键是**不能**把「无」原样返回 —— 它会被当 query 去检索，
    命中一堆书名含「无」的书，比空列表更误导人。
    """
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == []


def test_超长输出截断而不是丢掉(configured, monkeypatch):
    """模型把整页 OCR 结果吐出来了：截断比丢掉好，但要在日志里留痕。"""
    monkeypatch.setattr(vision, "_MAX_TITLE_LEN", 6)
    configured("高等数学" * 10)
    got = vision.extract_books_from_image(_DATA_URL)
    assert got == ["高等数学高等"]


def test_分块content也能处理(configured):
    """不同 provider 回的 content 形状不一样：有的给 str，有的给 [{'type':'text',...}]。"""
    configured([{"type": "text", "text": "《高等数学》"}])
    assert vision.extract_books_from_image(_DATA_URL) == ["高等数学"]


# ---- 多本书（书脊图的正面场景） ------------------------------------------------


def test_一排书脊要全部列出来不能只留第一本(configured):
    """⚠️ 这条是这次改动的核心回归用例。

    真实图片不是一张干净的封面照，是一排书脊。旧版取 `splitlines()[0]`，
    会安静地把第 2 本往后全部丢掉 —— 用户拿到 1/5 的结果，且毫无察觉。
    """
    configured("高等数学\n线性代数\n概率论与数理统计\n数据结构\n大学物理")
    assert vision.extract_books_from_image(_DATA_URL) == [
        "高等数学",
        "线性代数",
        "概率论与数理统计",
        "数据结构",
        "大学物理",
    ]


def test_带序号的列表也认(configured):
    """模型很喜欢输出「1. 高等数学」。序号是它加的，不是书名的一部分。"""
    configured("1. 高等数学\n2. 线性代数\n3、数据结构\n④ 大学物理\n- 概率论")
    assert vision.extract_books_from_image(_DATA_URL) == [
        "高等数学",
        "线性代数",
        "数据结构",
        "大学物理",
        "概率论",
    ]


@pytest.mark.parametrize("raw", ["5年高考3年模拟", "3D打印入门", "21世纪英语"])
def test_数字开头的真书名不能被当成序号切坏(raw, configured):
    """剥序号的正则必须要求数字后面跟分隔符。

    「5年高考3年模拟」是真实存在的书名，顺手写个 `^\\d+` 就把它切成
    「年高考3年模拟」了 —— 切完还是个词，检索不会报错，只会安静地搜错东西。
    """
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == [raw]


def test_属性行要被整行丢掉(configured):
    """模型常在一堆书名后面补一行「作者：同济大学」。

    留着的话「同济大学」会变成一个查询词，命中一堆同济大学出版社的书 ——
    用户看到结果还以为是识别对了。所以属性行整行丢，不是把标签切掉了事。
    """
    configured("高等数学\n作者：同济大学\n线性代数\n出版社：高等教育出版社")
    assert vision.extract_books_from_image(_DATA_URL) == ["高等数学", "线性代数"]


def test_重复的书名要去重(configured):
    """同一本书模型可能认两遍（书脊和封面各一次），大小写和空格也会飘。"""
    configured("《高等数学》\n高等数学\nC Primer Plus\nc  primer   plus")
    assert vision.extract_books_from_image(_DATA_URL) == ["高等数学", "C Primer Plus"]


def test_空行和纯符号行被跳过(configured):
    configured("高等数学\n\n---\n。。。\n线性代数")
    assert vision.extract_books_from_image(_DATA_URL) == ["高等数学", "线性代数"]


def test_超过上限时截断到上限(configured, monkeypatch):
    """上限挡的是「模型把整页 OCR 成 200 行」这种输出，不是省检索算力
    （省算力的那个上限在接口层，见 main.search_by_image）。"""
    monkeypatch.setattr(vision, "_MAX_BOOKS_PER_IMAGE", 3)
    configured("\n".join(f"书{i}" for i in range(10)))
    got = vision.extract_books_from_image(_DATA_URL)
    assert got == ["书0", "书1", "书2"]


# ---- 书脊上的「不是书名」的东西（真图跑出来的规则） ---------------------------
#
# 下面这几条不是想出来的，是拿一张真实书架照片跑出来的：
# test-images/wechat-shelf.jpg（肉眼能数出 17 本书脊），模型回了 20 行，
# 其中 6 行根本不是书名 —— 经济法学》编写组 / 学生用书 / （2021年版） /
# 王振友张丽丽李锋主编 / 北京大学出版社 / 吴传生主编。
#
# 而当时已有的 `_META_LABEL_RE` **一条都没拦住**：它要求行首是
# 「作者：」「出版社：」这种**带冒号**的标签，模型实际抄的是**裸的正文**。
# 记这一笔是为了提醒：这条规则是幸存者偏差的产物，只覆盖见过的 6 条，
# 没见过的不代表不存在。


@pytest.mark.parametrize(
    "raw",
    [
        "王振友张丽丽李锋主编",
        "经济法学》编写组",  # 模型把书名号切了一半，末尾只剩「编写组」
        "高等教育出版社",
        "中华书局",
    ],
)
def test_署名和出版单位行不是书名(raw, configured):
    """书脊上印着作者和出版社，模型会照抄。

    ⚠️ 留着特别危险：`吴传生主编` 拿去检索，库里真有一堆吴传生编的教材，
    能捞回看着完全合理的结果 —— 用户没法分辨这是识别对了还是把署名当书名了。
    """
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == []


@pytest.mark.parametrize("raw", ["（2021年版）", "(第三版)", "【2020】"])
def test_整行只是括号补充说明的要丢掉(raw, configured):
    """「（2021年版）」这种是从书名里掉出来的尾巴。

    ⚠️ 只丢**整行就是括号**的，不能对书名里的括号动手 ——
    「高等数学（上册）」是真书名，切了检索不到。
    """
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == []


@pytest.mark.parametrize("raw", ["学生用书", "教师用书", "智慧版", "上册", "下册"])
def test_不构成书名的通用标注要丢掉(raw, configured):
    """「学生用书」这类词单独出现时是教材的标注，不是书名。

    ⚠️ 实测确认过它的危害：`学生用书` 进检索命中了
    「走遍法国2 学生用书」「新视线俄语1 学生用书」—— 结果看着完全是真书，
    所以这个坏查询是**静默**的，用户只会觉得「搜出来的不太对」但说不出哪不对。
    """
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == []


@pytest.mark.parametrize(
    "raw",
    [
        "概率论与数理统计",
        "教育学基础",  # 以「基础」结尾，不能被「出版社」之类的后缀规则误伤
        "大学生思想道德修养与法律基础",
        "数据库系统概论（第5版）",  # 括号在中间，不是整行括号
        "C++ Primer Plus（中文版）",
    ],
)
def test_真书名不能被上面几条规则误伤(raw, configured):
    """⚠️ 这几条防线是**加法**，最容易的出事故方式是把真书名一起吃了。

    丢一个真书名的代价和认错一本是一样的：用户少看到一本，
    而且没有任何提示告诉他是被规则过滤掉的。
    """
    configured(raw)
    assert vision.extract_books_from_image(_DATA_URL) == [raw]


def test_不书名规则是顺序执行的不能互相短路(configured):
    """把脏行和干净行混在一起，确认脏行被剔掉、干净行一个不少。

    单条规则各自为政时容易过，合起来才暴露顺序问题
    （比如先截长度再判后缀，长署名行就会被截成一个像书名的东西）。
    """
    configured(
        "高等数学\n"
        "王振友张丽丽李锋主编\n"
        "线性代数\n"
        "（2021年版）\n"
        "学生用书\n"
        "概率论与数理统计\n"
        "北京大学出版社\n"
    )
    assert vision.extract_books_from_image(_DATA_URL) == [
        "高等数学",
        "线性代数",
        "概率论与数理统计",
    ]


# ---- 消息形状 ---------------------------------------------------------------


def test_发出去的是多模态消息(configured):
    """锁死协议形状：content 是列表，里面同时有文本块和图片块。

    这条挂了通常意味着换 provider 时协议对不上，是那种
    「代码看起来完全正常、但模型就是收不到图」的故障 —— 值得钉住。
    """
    fake = configured("高等数学")
    vision.extract_books_from_image(_DATA_URL)

    msg = fake.seen[0][0]
    assert isinstance(msg, HumanMessage)
    assert isinstance(msg.content, list)
    kinds = [p["type"] for p in msg.content]
    assert kinds == ["text", "image_url"]
    assert msg.content[1]["image_url"]["url"] == _DATA_URL


def test_prompt里明确要求只输出书名(configured):
    """prompt 不写清的话模型会回一整句解释，整句拿去检索会触发「条件词稀释」。"""
    fake = configured("高等数学")
    vision.extract_books_from_image(_DATA_URL)
    text = fake.seen[0][0].content[0]["text"]
    assert "只输出" in text


def test_prompt里要求一行一本并且不许猜(configured):
    """这两句是书脊场景的承重墙，删掉功能不会报错，只会悄悄变差：

    - 「一行一本」：不写的话模型把几本连成一句「图中可见高数、线代等教材」，
      整句进检索会触发条件词稀释。
    - 「不要猜」：书脊图必然有几本糊的。让它编一个像模像样的书名，
      那个名字照样能检索出几条看着挺像的结果，用户分不出来 ——
      这比漏掉一本危险得多。
    """
    fake = configured("高等数学")
    vision.extract_books_from_image(_DATA_URL)
    text = fake.seen[0][0].content[0]["text"]
    assert "一行一本" in text
    assert "不要猜" in text


# ---- 不越权的性质 -----------------------------------------------------------


def test_识别结果只作为字符串返回不掺进prompt(configured):
    """🔒 核心安全性质：图片是不可信输入，它的内容只能当检索词。

    这个用例把「模型输出的不是指令、是数据」钉成一条可回归的断言：
    整次调用里**只有一个** HumanMessage，且它的文本部分是我们自己的 prompt。
    图片内容出现在 image_url 里，**永远不出现在文本 prompt 里**。

    ⚠️ 现在返回的是一串书名而不是一个，这条性质**没有被削弱** ——
       越权的判断标准是「有没有回到 prompt 里」，不是「返回了几个字符串」。
       多本书只是多了几个查询词，每个走的是同一个出口。
    """
    fake = configured("《高等数学》\n《线性代数》")

    got = vision.extract_books_from_image(_DATA_URL)

    assert got == ["高等数学", "线性代数"]
    assert len(fake.seen) == 1, "只该发一轮，不该把识别结果再喂回去"
    prompt_text = fake.seen[0][0].content[0]["text"]
    assert "高等数学" not in prompt_text, "识别结果不许回灌进 prompt"
    assert "线性代数" not in prompt_text
