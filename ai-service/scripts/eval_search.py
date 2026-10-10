"""检索质量测试集（M2-3）：跑一批「查询 → 期望结果」，算命中率。

这个脚本回答一个问题：**改了一版检索（换拼法 / 调阈值 / 换模型），
效果是变好还是变坏？**

为什么必须有一个固定测试集：
  没有它，你每次调参都只能「感觉好像好了一点」。有了它，好坏是一个数。
  调优前后的这个对比，就是面试素材（见 docs/需求与接口.md §5.3）。

为什么要走 HTTP 而不是直接 import search()：
  走接口才是**真实的调用路径** —— 它会验证 min_score 的默认值有没有真的
  接上、响应结构对不对。直接调函数会绕过这些。（见面试素材第 12 条：
  验收用例必须能走到目标代码路径。）

用法（先确保 Spring Boot :8080 和 AI 服务 :8000 都起着，且索引已建）：

    .venv\\Scripts\\python.exe scripts\\eval_search.py

    # 存一份快照，用于调优前后对比
    .venv\\Scripts\\python.exe scripts\\eval_search.py --save 调阈值前

    # 和之前存的快照对比（看哪些用例翻盘了）
    .venv\\Scripts\\python.exe scripts\\eval_search.py --diff 调阈值前

    # 临时试别的阈值，不改代码
    .venv\\Scripts\\python.exe scripts\\eval_search.py --min-score 0.55

【你要改的只有 CASES】—— 见下面的说明。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 不是终端时 stdout 是块缓冲的，不加这行能几十秒一行不显示，看起来像卡死。
# 加 hasattr 保护：pytest 会把 sys.stdout 换成自己的捕获对象，那个没有 reconfigure，
# 不加判断的话本文件被 tests/ 里的集成测试 import 时会直接炸。
#
# errors="replace"：本机控制台是 cp936(GBK)，本文件打 ✅/⚠️/❌ 会 UnicodeEncodeError。
# 实测两个 venv 都是 gbk（`.venv` 和 `.venv-eval` 一样），所以这不是偶发。
# 之前跑成功是因为那个终端是 UTF-8 代码页；换个终端（比如双击 bat、IDE 内置
# 非 UTF-8 终端）就会在**汇总打印那一步**崩，前面的评测结果全白跑。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, errors="replace")

import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import httpx  # noqa: E402

AI_SERVICE = "http://127.0.0.1:8000"

# 问服务要几条。取大一点（10），这样即使 top-3 没中，你也能看到
# 「其实第 4 条是对的」—— 这是判断"拼法问题"还是"阈值问题"的关键线索。
TOP_K = 10

# 「返回的 top-N 里出现期望书名」算命中。文档 §5.3 定的目标是 top-3。
HIT_AT = 3
TARGET_RATE = 0.80

# 索引空了的话，探针查询用这个。选「线代」是因为它简短、肯定该有结果。
PROBE_QUERY = "线代"


@dataclass
class Case:
    """一条测试用例。

    :param query:  用户会怎么问
    :param expect: 哪些书名算答对。返回的 top-3 里出现**任意一个**就算命中。
                   写 [] 表示负样本 —— 要求服务返回**空列表**。
    :param note:   给人看的备注，不参与判定
    """

    query: str
    expect: list[str]
    note: str = ""

    @property
    def is_negative(self) -> bool:
        return not self.expect


# ============================================================================
# 【 测试用例 —— 你来填 】
# ============================================================================
#
# 规则：
#   - expect 写「哪些书名算答对」。匹配是**精确匹配**（去掉首尾空格后相等）。
#     要算多本就把名字都列上，例如 ["数据结构", "数据结构与算法分析"]。
#   - expect 写 [] = 负样本，要求返回空列表。
#
# 下面 3 条是**已经验过的回归基线，别删**。它们是保险丝：
# 保证你后面调参时，没把已经对的东西调坏。
#
# 往下面接着加，凑到 15~20 条，覆盖这五类（这才是测试集的价值所在 ——
# 光测「同济高数」这种干净查询，所有拼法都是 100% 命中，测不出任何东西）：
#
#   ① 精确书名          "计算机网络 谢希仁"
#   ② 口语说法          "有没有高数书" / "我想找本高数的" / "高数还有吗"
#   ③ 名字很像的书      库里同时有《数据结构》《数据结构与算法分析》
#                       《数据结构与算法分析 C语言描述》—— 搜「数据结构」该出哪本？
#   ④ 书名里带别人名字   "同济版高数" —— 线代也是同济版，「同济」这个词帮不上忙
#   ⑤ 只说用途          "考研数学用的" / "大二下学期要用的"
#
# 负样本别只写「量子力学」（太明显了，函数随便都能过）。挑**看起来该有、其实没有**的，
# 比如「有机化学」「法学概论」，甚至「同济版线性代数」—— 有同济版线代，但没有「同济版」这个说法。
#
# 想看库里到底有哪些书，跑：
#     curl -s "http://127.0.0.1:8080/api/posts?size=500"
# 或直接查库：
#     export MYSQL_PWD=<密码>   # 密码走环境变量，别写在命令行里（写进去会跟着文件进 git）
#     mysql -uroot -e "USE book_agent; SELECT DISTINCT book_name FROM book_post WHERE extract_status='DONE';"
# ============================================================================

# ⚠️⚠️ 这套用例是**照 1 万条语料重建的**（2026-10-09）。上一版是针对 220 条 /
#    37 个书种写的，语料一换就**整体作废**了 —— 最典型的是负样本：
#
#        量子力学    220 条语料里没有 → 现在 12 条
#        有机化学     同上 → 现在 43 条（《基础有机化学》等）
#        宏观经济学    同上 → 现在 11 条
#
#    这就暴露了「负样本」这个测法的本质问题：**它测的是语料，不是检索质量**。
#    语料一变，「库里没有」这个前提就没了，用例从「应该返回空」变成「答对了却判错」。
#    所以每次换语料，负样本必须**重新查一遍库**再定（下面的都是 `LIKE '%X%'` 数过、
#    **确认 0 条**才敢放上来的）。
#
#    至于正样本的 expect：判定是**精确匹配**书名，写错一个字就是一条假失败。
#    下面的书名全部是从库里 `GROUP BY book_name` 抄出来的原文。
CASES: list[Case] = [
    # ---- 回归基线（老库里就有，别删）----
    Case("同济高数", ["高等数学"], "精确简称"),
    Case("计算机网络", ["计算机网络"], "精确书名 + BM25 主场"),
    Case("软件工程", ["软件工程"], "精确书名"),

    # ---- 口语说法 ----
    Case("有没有高数书", ["高等数学"], "口语：带疑问句"),
    Case("高数还有吗", ["高等数学"], "口语：带疑问句"),
    Case("我想找本线代", ["线性代数", "线性代数应该这样学"],
         "口语。⚠️ expect 里必须有《线性代数应该这样学》—— 它是另一本**真实的**线代书"
         "（Sheldon Axler 那本），实测排在第 1~3 名。"
         "不把它列进来就是把「答对了」判成「答错了」"),
    Case("线代", ["线性代数", "线性代数应该这样学"], "极简口语"),

    # ---- 近义簇 ----
    Case("数据结构", ["数据结构", "数据结构与算法分析"],
         "近义簇：库里还有《数据结构与算法分析》(7 条)"),
    Case("毛概", ["毛概", "毛泽东思想和中国特色社会主义理论体系概论"],
         "近义簇：库里**两种写法都有** —— 有书名直接就是《毛概》的行，"
         "也有全称《毛泽东思想和中国特色社会主义理论体系概论》(8 条)。"
         "原始帖里学生写「毛概」，M1 抽取有时原样保留、有时归一成全称，"
         "所以两个都得算对（这就是「同一个词被抽成两个名字」，见面试素材第 6 条）"),
    Case("新视野大学英语",
         ["新视野大学英语", "新视野大学英语读写教程", "新视野大学英语综合教程",
          "新视野大学英语视听说教程", "新视野大学英语2", "新视野"],
         "近义簇：库里 8 个变体，抽取得很碎"),
    Case("计算机操作系统", ["操作系统", "计算机操作系统", "操作系统导论",
                            "操作系统设计与实现"],
         "近义簇：库里 4 种写法，还有《操作系统导论》这种不同作者的同名书"),

    # ---- 书名里带别人名字 ----
    Case("同济版高数", ["高等数学"],
         "「同济」是版本体系、不是书名的一部分 —— 实测库里**没有任何书名含「同济」**，"
         "所以这条只能靠语义，纯字面匹配一条都搜不到"),

    # ---- 只说用途 ----
    Case("考研数学用的", ["考研数学复习全书", "高等数学", "线性代数", "概率统计"],
         "只说用途。⚠️ expect 放宽了：1 万条语料里「考研数学用的」到底该命中哪本，"
         "已经没有唯一答案（库里同时有复习全书、高数、线代、概率统计）"),

    # ---- 精确词面（BM25 的主场，加它是为了**能量化**它有没有用）----
    Case("谢希仁", ["计算机网络"], "罕见作者名。词面唯一，理论上 BM25 的主场"),
    Case("严蔚敏", ["数据结构", "算法分析"],
         "罕见作者名，同上。⚠️ expect **必须**带上《算法分析》—— 查库确认过："
         "库里严蔚敏名下就是这两本（`SELECT book_name, author FROM book_post "
         "WHERE author LIKE '%严蔚敏%'` → 数据结构 / 算法分析）。"
         "上 rerank 之后它把《算法分析》提到了第 1 —— 那是**对的**，"
         "只写「数据结构」的话会把正确答案判成错的（实测踩过）"),

    # ---- 只有书本关键字 ----
    Case("java还有吗", ["Java核心技术"], "库里 10 条，原名就叫《Java核心技术》"),

    # ---- 负样本：要求返回空 ----
    #
    # ⚠️ 只放**能分得开**的。下面这几条实测 top-1 都低于阈值，阈值一调就能体现出来。
    #    分不开的那些挪到 KNOWN_LEAKS —— 不是删掉，是别让测试集常年见红：
    #    一个永远有红条的测试集，红条就失去信息量了。
    Case("考古", [], "0 条，实测返回空"),
    Case("葡萄酒", [], "0 条，实测返回空"),
    Case("区块链", [], "0 条，实测返回空"),
    Case("钢琴", [], "0 条，实测返回空"),
    Case("土木工程", [], "0 条，实测返回空"),
]

# ============================================================================
# 已知拦不住 —— 余弦口径下的原理性上限，以及**上完 rerank 之后的实测答复**
# ============================================================================
# 这些查询的共同点：余弦分**高过**正例的下界（0.6187，来自「同济高数」）。
# 也就是说**任何单一余弦阈值都救不了** —— 想把《Java核心技术》从「核工程」里过滤掉，
# 就得把线抬到 0.61 以上，可那会把「同济高数」一起误杀。
#
#   查询        top1 余弦   它返回了什么
#   ──────────────────────────────────────────────────
#   纺织        0.6959   这个词是怎么来的
#   农业经济学    0.7301   经济地理学
#   法学概论      0.7004   法学方法论
#   建筑学       0.6707   建筑物理
#   国际关系      0.6233   深度关系
#   环境工程      0.6199   Linux环境编程
#   教育学       0.6158   数学
#   核工程       0.6034   Java核心技术
#
# ── 上完 rerank 之后的实测答复（2026-10-09）──────────────────────────────
#
# ⚠️ 先说最容易被误读的那条：**这 8 条里仍然只有 1 条被拦下**（核工程，
#    它本来就低于余弦阈值）。也就是说 **rerank 并没有「修好」这张表**。
#
# 但这**不等于 rerank 没用** —— 它有用，只是用处不在这儿（见下面的两分类）。
# rerank 的真实收益在正例那一侧，是**排序**不是**过滤**：
# 正例 top-1 命中率 15/16 → **16/16**，3 条第 1 名变动且全部不劣
# （有没有高数书：数学→高等数学；新视野大学英语：英语2→英语；
#   严蔚敏：数据结构→算法分析，查库确认两本都是严蔚敏写的）。
# 对比脚本：`eval_search.py --no-rerank` 跑一遍再 `--diff`。
#
# ── 这 8 条其实分两类，混在一起谈是错的 ─────────────────────────────────
#
# **A 类：模型确实看穿了，只是掉得不够低**（精排分掉了，但仍高于安全线）
#
#   查询        余弦     精排      撞上的书
#   核工程      0.6034  0.5324    国际工程承包常用词汇
#   环境工程     0.6199  0.5473    Linux环境编程
#   教育学      0.6158  0.5794    数学
#   国际关系     0.6233  0.6299    深度关系
#
#   cross-encoder 把「工程≠核工程」「环境编程≠环境工程」读出来了，分数压到 0.53~0.63。
#   问题在于**正例那一侧只有 0.6987~0.7311 这么宽**（见 config.py 的 rerank_min_score），
#   安全阈值切不出「拦掉它们」又不碰正例的位置。16 条样本在 0.03 空隙里切一刀 = 过拟合。
#   所以现在**不按精排分过滤**，只重排。要动这个旋钮，先在更多正例上重扫。
#
# **B 类：压根不是「越界」，是主题相邻的真实书**
#
#   查询        余弦     精排      撞上的书        为什么它其实是对的
#   纺织        0.6959  0.6960   这个词是怎么来的   那本就是讲纺织词源的
#   法学概论     0.7004  0.7151   法律的概念       法学概论 ≈ 法理学入门
#   建筑学      0.6707  0.7100   建筑物理         同属建筑，是合理的近似
#   农业经济学    0.7301  0.6714   土地经济学论纲    农业经济学的近邻学科
#
#   精排给它们 0.67~0.72 **是正确判断** —— 这些书确实相关。
#   把它们叫「leak」，是**余弦时代留下的框架**：当时只看得到「分数高但库里没有对口书名」，
#   就一律当成模型出错。上了 cross-encoder 才看清其中有几本本来就该相关。
#
#   真正该修的是**产品定义**（用户搜「纺织」时，一本讲纺织词汇书源的书算不算命中？），
#   不是检索链路。这类问题不该靠调参解决。
#
# 为什么把这段留在代码里：面试问「你的阈值怎么定的」，能答
# 「我测出了余弦的上限，知道双塔为什么分不开，上了 cross-encoder，
#   用数据说明它改善的是排序、不是过滤，并说清哪些所谓 bad case 其实是产品定义问题」
# —— 比「调到一个看起来不错的数」强得多。
KNOWN_LEAKS: list[tuple[str, str, float]] = [
    # (查询, 实测返回的 top-1, 实测分数)   —— 分数是 2026-10-09 / 9997 条语料测的
    ("纺织", "这个词是怎么来的", 0.6959),
    ("农业经济学", "经济地理学", 0.7301),
    ("法学概论", "法学方法论", 0.7004),
    ("建筑学", "建筑物理", 0.6707),
    ("国际关系", "深度关系", 0.6233),
    ("环境工程", "Linux环境编程", 0.6199),
    ("教育学", "数学", 0.6158),
    ("核工程", "Java核心技术", 0.6034),
]

# ============================================================================
# BM25 到底有没有用：两轮实测的结论（**务必带上语料规模再说**）
# ============================================================================
#
# 复测用 `scripts/compare_bm25.py`（只切通道、同一份索引，逐条对比帖子 id 序列）。
#
# ── 第一轮：220 条 / 37 个书种 ─────────────────────────────────────────────
#   9 条查询，top-3 只有 1 条变（「计算机网络」）。命中率一条都没翻盘。
#   归因：① 语料小且**同质** —— 全是「书名+版次+作者+出版社」模板文本，BGE 本来就够用
#         ② BM25 的强项是长文档里的罕见词 / 编号；每本书才十来个字，IDF 拉不开
#
# ── 第二轮：9997 条 / 2869 个书名（2026-10-09）─────────────────────────────
#   16 条正例查询：**top-3 仍然只变了 1 条**，top-10 变了 5 条。
#
#   变了的那条（是**改善**）：
#     计算机操作系统   仅向量：操作系统导论 / 操作系统导论 / 计算机操作系统
#                     混  合：计算机操作系统 / 计算机操作系统 / 操作系统导论   ← 对的提到第 1
#
#   5 条 top-10 变动里最有说服力的一条：
#     计算机网络       混合把 6 条《大学计算机基础》挤出去，换成《计算机网络：自顶向下方法》
#                     —— 那 6 条是**同一本书的重复帖子**，占了 10 个位置里的 6 个
#
# ── 所以结论是什么（面试就这么答）──────────────────────────────────────────
#   **BM25 的收益在「清尾部」，不在「提命中率」。**
#   语料从 220 涨到 1 万之后，主要变化是 **top-10 里不再塞满无关/重复的条目**，
#   而不是「原本搜不到的书现在搜到了」。top-3 命中率几乎没动（1/16）。
#
#   为什么还是值得留：
#     ① 列表页给用户看 10 条，尾部质量直接影响观感 —— 6 条重复的《大学计算机基础》
#        排在《计算机网络》旁边，比命中率掉 1% 更伤体验
#     ② 「清掉同书重复」这件事纯向量做不到：重复帖子的向量几乎相同，余弦分一样高，
#        向量排序**天然分不开**它们；BM25 靠「这份文本里有没有这个词」能分
#
#   ⚠️ 但也别吹过：**它不是「提升了准确率」**。1/16 的 top-3 变动就是全部证据。
#   被问「加了 BM25 效果提升多少」，诚实答案是「top-3 命中率几乎没变，
#   主要价值是把 10 条列表里的重复项和无关项挤出去，因为……」
# ============================================================================


# ============================================================================
# 以下不用改
# ============================================================================


def call_search(
    client: httpx.Client,
    query: str,
    min_score: float | None,
    rerank: bool | None = None,
) -> list[dict]:
    """打一次 /search，返回 hits 列表。"""
    body: dict = {"query": query, "top_k": TOP_K}
    # 只有显式传了才带上 min_score —— 不传就让服务用它自己的默认常量。
    # 这样默认行为和服务真实默认值始终一致，不会出现「脚本和服务想的不是一回事」。
    if min_score is not None:
        body["min_score"] = min_score
    # 同理：不传就是服务配置里的默认（现在是开）。--no-rerank 才传 False。
    if rerank is not None:
        body["rerank"] = rerank

    resp = client.post(f"{AI_SERVICE}/search", json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["hits"]


def hit_names(hits: list[dict]) -> list[str]:
    """从 hits 里取书名，给判定和显示用。"""
    return [(h.get("book_name") or h.get("text") or "?") for h in hits]


def top_str(hits: list[dict]) -> str:
    """把前 HIT_AT 条的书名拼成一行展示，空的显示 (空)。"""
    return " / ".join(hit_names(hits[:HIT_AT])) or "(空)"


def judge(case: Case, hits: list[dict]) -> bool:
    """这条用例算不算过。"""
    if case.is_negative:
        # 负样本：要求一条都不返回
        return len(hits) == 0
    # 正例：top-HIT_AT 里出现任意一个期望书名
    top = hit_names(hits[:HIT_AT])
    return any(name in case.expect for name in top)


def preflight(client: httpx.Client) -> bool:
    """开跑前先确认服务和索引都活着。不然会得到一堆假失败。"""
    try:
        health = client.get(f"{AI_SERVICE}/health", timeout=5).json()
    except Exception as e:
        print(f"❌ 连不上 AI 服务 {AI_SERVICE}")
        print(f"   （{type(e).__name__}: {e}）")
        print("   起服务：.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000")
        return False

    print(f"AI 服务 OK：{health}")

    # 探针：索引空的话，后面所有用例都会「失败」，但根因跟检索质量无关。
    # 先在这里拦住，否则你会拿一份全是 ❌ 的报告去调拼法，白调半天。
    if not call_search(client, PROBE_QUERY, None):
        print()
        print(f"⚠️  探针查询「{PROBE_QUERY}」返回 0 条 —— 索引很可能是空的。")
        print("   先建索引（1 万条约 25 秒）：")
        print("     curl -X POST http://127.0.0.1:8000/index/build "
              "-H 'Content-Type: application/json' -d '{}'")
        return False

    return True


def run(client: httpx.Client, min_score: float | None,
        rerank: bool | None = None) -> list[dict]:
    """跑全部用例，返回每条的判定结果。"""
    rows = []
    for i, case in enumerate(CASES, 1):
        try:
            hits = call_search(client, case.query, min_score, rerank)
        except Exception as e:
            print(f"[{i:>2}] {case.query:<22} ⚠️  请求失败：{type(e).__name__}: {e}")
            rows.append({"case": case, "ok": False, "hits": [], "error": str(e)})
            continue

        ok = judge(case, hits)
        mark = "✅" if ok else "❌"
        # 两个分并排显示。余弦那一栏已经证明到顶了（见 analyze_threshold.py），
        # 精排那一栏才是现在真正决定顺序的东西 —— 出了问题一眼能看出是哪一栏的锅。
        if hits:
            top1 = hits[0]
            rr = top1.get("rerank_score")
            score_str = (f"cos {top1['score']:+.4f} / rr "
                         f"{rr:.4f}" if rr is not None else f"cos {top1['score']:+.4f} / rr   —  ")
        else:
            score_str = "          (空)         "

        print(f"[{i:>2}] {case.query:<22} {mark}  {score_str}  {top_str(hits)}")

        # 失败时把 top-K 全打出来 —— 常见情况是「第 1 名错了但第 4 名是对的」，
        # 这说明阈值把对的那条砍了，而不是拼法找不到它。两种问题的修法完全不同。
        if not ok and len(hits) > HIT_AT:
            print(f"      ↳ 第 {HIT_AT + 1}~{TOP_K} 名："
                  f"{' / '.join(hit_names(hits[HIT_AT:]))}")

        rows.append({"case": case, "ok": ok, "hits": hits})

    return rows


def report_known_leaks(client: httpx.Client, min_score: float | None,
                       rerank: bool | None = None) -> list[dict]:
    """把 KNOWN_LEAKS 实测一遍，看还有几条拦不住。

    **故意不参与及格率**：这些是当前方案的原理性上限，不是回归 bug。
    但每次都要实测而不是照抄上面那张表 —— 因为上完 rerank 之后，
    这里的分数会掉、甚至直接返回空，那时就该把它们搬回 CASES 了。
    """
    print()
    print("=" * 78)
    print("已知拦不住（不计入及格率）—— 单一余弦阈值的原理性上限")
    print("=" * 78)

    rows = []
    for query, expected_leak, before in KNOWN_LEAKS:
        try:
            hits = call_search(client, query, min_score, rerank)
        except Exception as e:
            print(f"  {query:<12} ⚠️ 请求失败 {type(e).__name__}")
            continue

        if not hits:
            # 被拦住了 —— 要么阈值抬了，要么 rerank 上了。两种都值得高兴，但要看是哪种。
            print(f"  {query:<12} ✅ 已拦下（原先会返回《{expected_leak}》{before:.4f}）"
                  f" ←rerank 或阈值生效了，可以把这条搬回 CASES")
            rows.append({"query": query, "blocked": True, "score": None,
                         "name": None, "before": before})
        else:
            now, name = hits[0]["score"], hits[0].get("book_name") or "?"
            rr = hits[0].get("rerank_score")
            rr_str = f"  rr {rr:.4f}" if rr is not None else ""
            if name != expected_leak:
                # top-1 被精排换掉了 —— 换成的那本往往**更相关**（见 KNOWN_LEAKS 的注释）。
                delta = f" ←精排把《{expected_leak}》换成了这本"
            else:
                delta = ""
            print(f"  {query:<12} ❌ 仍返回《{name}》{now:.4f}{rr_str}{delta}")
            rows.append({"query": query, "blocked": False, "score": now,
                         "name": name, "before": before, "rerank_score": rr})

    blocked = sum(1 for r in rows if r["blocked"])
    print(f"\n  {blocked}/{len(rows)} 条已被拦下")
    if blocked == 0:
        print("  全都没拦住 —— 这是**预期内的**，修法是把检索链路接上 rerank（Track 1.2），")
        print("  而不是继续抬阈值（抬到能拦住它们，正例就全死了，实测过）。")
    return rows


def top1_ok(case: Case, hits: list[dict]) -> bool:
    """第 1 名对不对。

    ⚠️ 为什么要单独统计这个：top-3 命中率**太宽松了**，已经饱和在 100%
    （1 万条语料上正例 16/16），任何改动都看不出差别 ——
    换句话说它已经没有分辨力了。而用户实际就看第 1 条，
    「对的排第 1 还是排第 3」是**体验差别**，top-3 命中率看不见它。
    上 rerank 前后的对比，差别全在这一栏。
    """
    if case.is_negative:
        return len(hits) == 0
    if not hits:
        return False
    return (hits[0].get("book_name") or "") in case.expect


def summarize(rows: list[dict]) -> dict:
    """算命中率并打印失败清单。"""
    pos = [r for r in rows if not r["case"].is_negative]
    neg = [r for r in rows if r["case"].is_negative]
    pos_ok = sum(1 for r in pos if r["ok"])
    neg_ok = sum(1 for r in neg if r["ok"])
    pos_top1 = sum(1 for r in pos if top1_ok(r["case"], r["hits"]))

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)

    if pos:
        rate = pos_ok / len(pos)
        flag = "✅ 达标" if rate >= TARGET_RATE else "❌ 未达标"
        print(f"  正例 top-{HIT_AT} 命中率： {pos_ok}/{len(pos)} = {rate:.1%}"
              f"   （目标 ≥ {TARGET_RATE:.0%}） {flag}")
        print(f"  正例 top-1 命中率： {pos_top1}/{len(pos)} = {pos_top1 / len(pos):.1%}"
              f"   ← 这一栏才有分辨力，top-{HIT_AT} 已经饱和了")
    else:
        rate = 0.0
        pos_top1 = 0
        print("  正例： 0 条 —— 你还没填用例呢")

    if neg:
        print(f"  反例通过率：        {neg_ok}/{len(neg)} = "
              f"{neg_ok / len(neg):.1%}   （要求 100%）")

    failed = [r for r in rows if not r["ok"]]
    if failed:
        print()
        print(f"  失败用例（{len(failed)} 条）—— 这些就是你要查的 bad case：")
        for r in failed:
            c = r["case"]
            expect = "空" if c.is_negative else " / ".join(c.expect)
            print(f"    - {c.query!r:<24} 期望 {expect:<20} 实际 {top_str(r['hits'])}")
            if c.note:
                print(f"      （{c.note}）")
    else:
        print()
        print("  全部通过 🎉")

    return {
        "positive_rate": rate if pos else None,
        "positive_ok": pos_ok,
        "positive_total": len(pos),
        "positive_top1_ok": pos_top1,
        "negative_ok": neg_ok,
        "negative_total": len(neg),
    }


def snapshot_path(name: str) -> Path:
    """快照存到 scripts/eval_runs/<name>.json。"""
    folder = Path(__file__).resolve().parent / "eval_runs"
    folder.mkdir(exist_ok=True)
    return folder / f"{name}.json"


def save_snapshot(name: str, rows: list[dict], summary: dict) -> None:
    path = snapshot_path(name)
    payload = {
        "summary": summary,
        "cases": [
            {
                "query": r["case"].query,
                "expect": r["case"].expect,
                "ok": r["ok"],
                "top1_ok": top1_ok(r["case"], r["hits"]),
                "top": hit_names(r["hits"][:HIT_AT]),
                "top1_score": r["hits"][0]["score"] if r["hits"] else None,
                "top1_rerank_score": (
                    r["hits"][0].get("rerank_score") if r["hits"] else None
                ),
            }
            for r in rows
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n快照已存：{path}")


def diff_snapshot(name: str, rows: list[dict], summary: dict) -> None:
    """和快照对比 —— 调优到底有没有用，看这里。"""
    path = snapshot_path(name)
    if not path.exists():
        print(f"\n⚠️  找不到快照 {path}，跳过对比")
        return

    old = json.loads(path.read_text(encoding="utf-8"))
    old_map = {c["query"]: c for c in old["cases"]}
    new_map = {r["case"].query: r for r in rows}

    print()
    print("=" * 78)
    print(f"对比快照：{name}")
    print("=" * 78)

    o = old["summary"]
    print(f"  正例 top-{HIT_AT} 命中率： {o['positive_ok']}/{o['positive_total']} "
          f"→ {summary['positive_ok']}/{summary['positive_total']}")
    # 老快照可能没有这一栏（top-1 统计是 2026-10-09 上 rerank 时才加的）
    if "positive_top1_ok" in o:
        print(f"  正例 top-1  命中率： {o['positive_top1_ok']}/{o['positive_total']} "
              f"→ {summary['positive_top1_ok']}/{summary['positive_total']}")
    else:
        print(f"  正例 top-1  命中率： （老快照没记） → "
              f"{summary['positive_top1_ok']}/{summary['positive_total']}")

    flipped_up = [q for q, r in new_map.items() if r["ok"] and q in old_map and not old_map[q]["ok"]]
    flipped_down = [q for q, r in new_map.items() if not r["ok"] and q in old_map and old_map[q]["ok"]]
    added = [q for q in new_map if q not in old_map]

    # ── 第 1 名的变动 ────────────────────────────────────────────────────
    # ⚠️ 必须单独看。top-3 命中率已经饱和在 100%，任何改动都**翻不动**它 ——
    # 只对比 ok 的话会得到「这次改动没有影响」的假结论。
    # 上 rerank 那次实测：top-3 一条没变，但第 1 名变了 3 条。
    def _old_top1(q: str) -> str | None:
        top = old_map[q]["top"]
        return top[0] if top else None

    top1_changed = [
        q for q, r in new_map.items()
        if q in old_map and _old_top1(q) is not None
        and (hit_names(r["hits"][:1]) or [None])[0] != _old_top1(q)
    ]

    def arrow(q: str) -> str:
        old_top = " / ".join(old_map[q]["top"]) or "(空)"
        return f"     {q!r}：{old_top}  →  {top_str(new_map[q]['hits'])}"

    if flipped_up:
        print(f"\n  ✅ 修好了（{len(flipped_up)} 条）：")
        for q in flipped_up:
            print(arrow(q))
    if flipped_down:
        print(f"\n  ❌ 调坏了（{len(flipped_down)} 条）—— 这才要命，别只看总命中率：")
        for q in flipped_down:
            print(arrow(q))
    if added:
        print(f"\n  ＋ 新增用例 {len(added)} 条（快照里没有，无法对比）："
              f"{'、'.join(repr(q) for q in added)}")

    # 这一栏经常是**唯一**有信息量的那一栏 —— 见上面 top1_changed 的注释。
    if top1_changed:
        print(f"\n  ◆ 第 1 名变了 {len(top1_changed)} 条"
              f"（top-{HIT_AT} 看不出来，但这才是用户实际看到的那一条）：")
        for q in top1_changed:
            print(f"     {q!r}：{_old_top1(q)}  →  "
                  f"{(hit_names(new_map[q]['hits'][:1]) or ['(空)'])[0]}")
    else:
        print("\n  ◆ 第 1 名一条没变")

    if not (flipped_up or flipped_down or top1_changed):
        print("\n  没有用例翻盘 —— 这次改动对测试集没有影响")


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量测试集（M2-3）")
    parser.add_argument("--save", metavar="名字", help="把本次结果存成快照")
    parser.add_argument("--diff", metavar="名字", help="和之前存的快照对比")
    parser.add_argument("--min-score", type=float, default=None,
                        help="临时指定阈值（不传则用服务默认值）")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                        help="强制开精排（服务默认已开，一般不用传）")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="强制关精排 —— 用它跑一遍再对比，才能量化 rerank 的净增益")
    args = parser.parse_args()

    if not CASES:
        print("CASES 是空的 —— 先在脚本里填测试用例")
        return

    n_pos = sum(1 for c in CASES if not c.is_negative)
    n_neg = len(CASES) - n_pos
    print("=" * 78)
    print(f"检索质量测试集 —— {len(CASES)} 条用例（正例 {n_pos} / 反例 {n_neg}）")
    print("=" * 78)
    print(f"AI 服务：{AI_SERVICE}   top_k={TOP_K}   命中判定：top-{HIT_AT}")
    if args.min_score is not None:
        print(f"⚠️  临时阈值：{args.min_score}（覆盖服务默认值）")
    rerank_label = {None: "按服务配置", True: "强制开", False: "强制关 --no-rerank"}[args.rerank]
    print(f"精排：{rerank_label}（rerank_score 只在开了精排时才有值）")
    print()

    with httpx.Client() as client:
        if not preflight(client):
            return
        print()
        print("-" * 78)
        rows = run(client, args.min_score, args.rerank)
        summary = summarize(rows)
        leaks = report_known_leaks(client, args.min_score, args.rerank)
        summary["leaks_blocked"] = sum(1 for r in leaks if r["blocked"])
        summary["leaks_total"] = len(leaks)

    if args.save:
        save_snapshot(args.save, rows, summary)
    if args.diff:
        diff_snapshot(args.diff, rows, summary)


if __name__ == "__main__":
    main()
