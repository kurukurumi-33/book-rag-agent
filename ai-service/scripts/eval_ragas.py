# -*- coding: utf-8 -*-
"""Ragas 评测（**生成侧**）：忠实度 / 上下文精确率 / 上下文召回率。

================================================================================
它补的是哪块空白（先说清，别误解成「把模糊的变清楚」）
================================================================================

现有指标**全部在检索侧**：top-1 命中率、阈值、召回变化、成本。生成侧一个都没有。
面试官问「你 Agent 回答的质量怎么保证」，现在答不上来 —— 这个脚本就是答案。

⚠️ 所以别把它当成「把模糊指标变清楚」：检索侧那些数字一点都不模糊
（`0.7147 → 0.5606`、`15/16 → 16/16` 都是实测）。它补的是**没有的那一类**。

================================================================================
跑之前必须知道的五件事（都是坑，不是注意事项）
================================================================================

1. **它装在另一个 venv（`.venv-eval`）里，不进服务运行环境。**
   ragas 依赖 pin 了 `langchain-openai < 1.1.10`，实测 `pip install ragas` 会把
   langchain-openai **从 1.6.6 降到 1.1.9**（复现：
   `.venv\\Scripts\\python.exe -m pip install --dry-run ragas`）。
   那个包是 AI 服务里**文本模型和视觉模型唯一的入口**，降 5 个次版本有把
   `with_structured_output` / 多模态消息打坏的风险。

   好在 eval_search.py 当初就定了「评测**走 HTTP，不 import 服务代码**」——
   评测工具本来就不需要和服务共享环境，那个决定现在正好用上。

2. **scikit-network 装不上，所以是 `--no-deps` + 手工装依赖。**
   它在 Python 3.14 / Windows 上**没有预编译 wheel**，源码编译要 MSVC C++ 生成工具
   （实测报错：`Microsoft Visual C++ 14.0 or greater is required`）。
   查过 ragas 源码：它只在 `ragas/testset/graph.py:312` **惰性 import**，
   属于「自动生成 QA 数据集」那部分功能。我们的评测集是**手工建的、直接复用
   eval_search.CASES**，根本不走那条路 —— 所以跳过它是安全的，不是权宜之计。

3. **ragas 0.4.3 引了一个 langchain-community 已经删掉的模块**，要垫片才能 import。
   见 `scripts/_ragas_compat.py`（含完整理由和边界）。

4. **用的是 `ragas.metrics.collections`（新版），不是 `ragas.metrics`（旧版）。**
   旧版 `Faithfulness()` 这类只吃 `BaseRagasLLM`，而 `llm_factory()` 返回的是
   `InstructorLLM` —— **实测 `isinstance(judge, BaseRagasLLM)` 是 False**，
   两者配不上（旧版还会打 deprecation 警告，v1.0 要删）。
   新版指标自带 `ascore()/score()`，要在**构造时**传 llm，逐条算，不走 `evaluate()`。
   所以下面的循环是自己写的，不是偷懒，是这套 API 本来就这么用。

5. **裁判是 DeepSeek，被测也是 DeepSeek。**
   **同族模型自评**，有已知的自我偏好偏差。所以这些数字看**趋势**，不看绝对值。
   面试时主动交代裁判是谁，比被问出来强。

================================================================================
口径（哪一栏是什么，别混着用）
================================================================================

    faithfulness                    回答里说的每一件事，上下文里都有吗
                                    → **防编造**：模型会不会编一个价格、
                                      编一句「有笔记」（工具结果里根本没有这一栏）
                                    入参：user_input / response / retrieved_contexts

    context_precision_with_reference 召回的内容里有多少是切题的
                                    → 候选池的噪声率，直接影响列表页观感
                                    入参：user_input / reference / retrieved_contexts

    context_recall                  该找到的书找到了吗
                                    → 我们**已有**确定性的 top-1 命中率，
                                      这一栏是 LLM 口径的旁证，不是主指标
                                    入参：user_input / retrieved_contexts / reference

⚠️ 后两个需要 `reference`，而我们能提供的 reference **只能是期望书名**
（没有「标准答案文本」）。拿书名当 reference 是**降格使用** ——
它反映得了「有没有找对书」，反映不了「答案写得对不对」。

⚠️ 检索侧的**主**指标仍然是 `eval_search.py` 的确定性命令率。这里是补充。
LLM 裁判的可重复性不如字符串匹配，同一份数据跑两遍结果会有出入。

⚠️ **三个指标的入参不是同一套**（上面已逐条列出，实测签名）。
   faithfulness 不吃 `reference`，另两个不吃 `response` ——
   拿一套 kwargs 挨个喂会直接 TypeError。`_judge_kwargs()` 按签名过滤。

================================================================================
用法（先起 Spring Boot :8080 + AI 服务 :8000，且索引已建）
================================================================================

    # 只采数据、不调裁判 —— **先跑这个**，确认数据收得对，且不花钱
    .venv-eval\\Scripts\\python.exe scripts\\eval_ragas.py --collect-only

    # 正式跑（会调 DeepSeek 当裁判，有 token 成本）
    .venv-eval\\Scripts\\python.exe scripts\\eval_ragas.py

    # 存快照，用于对比「改了 prompt / 换了模型之后有没有变好」
    .venv-eval\\Scripts\\python.exe scripts\\eval_ragas.py --save 上ragas基线

================================================================================
怎么把 .venv-eval 搭起来（**它被 .gitignore 挡着，clone 下来没有，得手动重建**）
================================================================================

    cd ai-service
    python -m venv .venv-eval
    .venv-eval\\Scripts\\python.exe -m pip install -U pip

    # ① 先按普通方式装 langchain 那一套（这几个在 py3.14 上都有 wheel）
    .venv-eval\\Scripts\\python.exe -m pip install ^
        langchain-core langchain langchain-community langchain-openai ^
        httpx python-dotenv

    # ② ragas 用 --no-deps 装，依赖手工补（**不能省 --no-deps**：
    #    它会连带拖 scikit-network，而那个在 py3.14/Windows 上要 MSVC 才能编译）
    .venv-eval\\Scripts\\python.exe -m pip install --no-deps ragas
    .venv-eval\\Scripts\\python.exe -m pip install ^
        numpy pandas datasets pyarrow openai instructor networkx tqdm ^
        pydantic typing-extensions orjson nest-asyncio

    # ③ 冒烟：能 import 出来就算成功（不需要 API key，不发请求）
    .venv-eval\\Scripts\\python.exe -c "import scripts._ragas_compat as c; c.install(); import ragas; print(ragas.__version__)"

⚠️ **不要 `pip install -r requirements.txt` 进这个环境。** ragas 会把
   langchain-openai 降到 1.1.x，那是服务里文本+视觉模型的唯一入口。
   两个环境就是为了隔开这件事（见开头理由 1）。

✅ **装完怎么判断装对了**（比背清单靠谱，直接查版本）：

        .venv-eval\\Scripts\\python.exe -m pip list --format=freeze | findstr /I "ragas langchain-core langchain-openai openai datasets"

    本机实测通过的一组（2026-10，Python 3.14.6 / Windows）：

        ragas==0.4.3          langchain-core==1.6.9
        langchain-openai==1.7.0   langchain-community==0.4.2
        openai==3.28.0        datasets==5.1.0

    **关键一条：langchain-openai 必须还是 1.7.x。**
    掉到 1.1.x 就说明被 ragas 的依赖解析带偏了 —— 那正是我们要避免的事，
    重装一遍。

⚠️ 上面 ② 的清单是**手工凑的，没有 lockfile**：那些包大部分是被 langchain
   和 ragas 连带拉进来的，这里列的是装完剩下需要手动补的。
   ragas 以后升级缺什么包，按 `ModuleNotFoundError` 补即可 ——
   环境是一次性的，不值得为它再引 poetry/uv 一层工具。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 让 `import eval_search` 能找到同级脚本。
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    # errors="replace" 是**兜底，不是装饰**：本机控制台是 cp936(GBK)，
    # 打 ✅/⚠️ 这类符号会 UnicodeEncodeError 直接崩。
    # 崩的位置很要命 —— 在**调完裁判、算完之后**的汇总打印里，
    # 等于钱花了、数算出来了、然后一句 print 把整轮结果带走。
    # UTF-8 终端下这些符号照常显示，"replace" 只在编码不了时把它降级成 ?。
    sys.stdout.reconfigure(line_buffering=True, errors="replace")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import httpx  # noqa: E402

# 必须在**第一次 import ragas 之前**打上（理由和边界见该文件开头）。
# 放在模块顶层而不是函数里，是为了保证「谁先 import ragas 都安全」。
import _ragas_compat  # noqa: E402

_ragas_compat.install()

# 评测集**直接复用** eval_search 的那一份，不在这里另建一套。
# 两套评测集意味着两处维护、迟早对不上；而且「检索侧和生成侧评的是同一批查询」
# 本身就是结论的一部分 —— 能直接对出「检索对了但回答编了」这种 case。
from eval_search import CASES, Case  # noqa: E402

AI_SERVICE = "http://127.0.0.1:8000"

# /chat 有限流（默认 20 次/分钟/IP，见 config.ratelimit_chat_per_min）。
# 21 条用例一次打完会贴着上限跑，网络抖动重试一下就 429 了 ——
# 所以默认每条之间 sleep 3 秒（= 20/分钟），可 --delay 调。
DEFAULT_DELAY = 3.0

# 指标名 → 打印用的中文列头。
# ⚠️ 这三个 key **必须**和 ragas collections 指标的 `.name` 完全一致
#    （实测：faithfulness / context_precision_with_reference / context_recall），
#    对不上的话快照里全是 None，而且不会报错 —— 只会安静地什么都没有。
METRICS = {
    "faithfulness": "忠实度",
    "context_precision_with_reference": "上下文精确率",
    "context_recall": "上下文召回率",
}


def evaluable(cases: list[Case]) -> list[Case]:
    """能评生成侧指标的用例。

    负样本不评：它要求返回**空列表**，此时「上下文」本来就是空的，
    拿空上下文算忠实度 / 精确率没有意义（分母是 0）。
    """
    return [c for c in cases if not c.is_negative]


def _env(key: str, default: str = "") -> str:
    """从环境变量或 ai-service/.env 取值（环境变量优先）。

    故意**不 import app.config**，这是刻意的解耦，不是因为 import 不了 ——
    实测 `from app.config import settings` 在 .venv-eval 里**是能跑通的**
    （app/__init__.py 是空的，config 只依赖 pydantic-settings，而那个被
    langchain 顺带装了）。所以别拿「装不上」当理由，那是不成立的。

    真正的理由：eval_search.py 开头就定了「评测走 HTTP，**不 import 服务代码**」。
    这个 eval 家族一旦开始 import app.*，就和服务内部的模块结构绑上了 ——
    哪天 app/__init__.py 加一行 import、或 config 多引一个重依赖，
    评测环境就跟着崩，而评测环境本来就该和服务环境无关。

    代价是自己解析 .env，多十几行。规则简单（KEY=VALUE、# 注释、
    首尾引号），够用；这份 .env 本来就是我们自己维护的。
    """
    val = os.environ.get(key)
    if val:
        return val
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return default


def call_chat(client: httpx.Client, query: str) -> dict:
    """打一次 /chat，拿到 reply / tool_calls / books。

    不传 session_id = 每次都是新会话。**这一点很重要**：
    复用会话会让上一轮的检索结果留在历史里，后面的用例就带了前面用例的上下文。
    """
    resp = client.post(f"{AI_SERVICE}/chat", json={"message": query}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _tool_price(card_price):
    """把卡片上的价签还原成 `search_books` 给模型的那个字符串。

    两处对「没标价」的措辞不一样，实测数据里同时出现了：
        模型看到的是 "未知"     （tools.py 里 _PRICE_UNKNOWN 的翻译）
        卡片上写的是 "未标价"   （回主服务取详情时 null 的翻译）
    回复里写「未知」而上下文写「未标价」，裁判会当成两回事。
    """
    return "未知" if card_price == "未标价" else card_price


def contexts_from(resp: dict) -> list[str]:
    """重建**模型真正看到**的工具结果，一条书一行。

    ⚠️ 这里最容易写错的一处：**不能直接把响应里的 `books` 拼进去。**
        `search_books` 返回给模型的只有 `post_id / book_name / price` 三栏
        （见 app/agent/tools.py 的返回值），而响应里的 `books` 是**服务端补全的卡片**，
        多了 `condition_desc / raw_text / status`。

        拿卡片当上下文会把**忠实度抬虚**：模型回答「有笔记、划过重点」，
        卡片里有这几栏，但工具结果里没有 —— 那其实是模型编的，
        用卡片评就会漏掉这个幻觉。评的是「回答 vs 它手里的材料」，
        材料必须是它真拿到的那些。

    `get_post_detail` 是真把全量字段给模型了，所以那个工具的结果按卡片还原。

    每行要写**工具返回的全部字段**，一个都不能少 —— 这是第一次真跑踩到的坑：
    漏掉 `post_id` 之后，模型回答里的编号（「| 209 | 高等数学 | 未知 |」）
    在上下文里找不到出处，裁判只能判成编的，16 条里有一半的忠实度被凭空压到 0.2。
    **它不是幻觉，是工具真给过的。** 少写一栏 = 自己给自己扣分。

    ⚠️ 还原**不是逐字节相等**，一处口径偏差要知道（来自「卡片是服务端翻译过的」）：

      价格：没标价时工具给模型的是字符串 `"未知"`（tools.py 的 `_PRICE_UNKNOWN`
      翻译），卡片里翻译成了 `"未标价"`。由 `_tool_price()` 还原回去。

    ⚠️ 另一处**不是偏差、是有意为之**：摘要里本来就没有「有没有笔记」这一栏
      （`search_books` 只回 post_id / book_name / price）。所以如果模型回答里说
      了「这本有笔记」而又没调 `get_post_detail`，这句话在上下文里**确实没有出处**，
      忠实度扣它是**对的**，不是评分器的毛病。

      注意别把它和「模型用 has_notes=True 筛过」搞混：筛过 ≠ 看得见。
      模型是拿「我筛了有笔记的」这个**自己的动作**在说话，而上下文里
      没有任何一条能证明。
    """
    cards = {b.get("post_id"): b for b in resp.get("books") or []}
    ctxs: list[str] = []

    for call in resp.get("tool_calls") or []:
        name = call.get("name")
        ids = call.get("post_ids") or []

        for pid in ids:
            card = cards.get(pid) or {}
            book = card.get("book_name") or "?"
            price = card.get("price")

            if name == "search_books":
                # ⚠️ **`编号 {pid}` 这一栏不能省** —— 第一次真跑就是漏了它，
                #    结果 16 条里有一半的忠实度被凭空压到 0.2。
                #
                #    模型看到的是 `str(results)`，即
                #    `[{'post_id': 209, 'book_name': '高等数学', 'price': '未知'}, ...]`
                #    —— **post_id 就在里面**，而且模型回答时很爱拿它当行号用
                #    （「| 209 | 高等数学 | 未知 |」）。上下文里不写编号，
                #    裁判就只能把「209」判成编的，可它恰恰是工具真给过的。
                #
                #    这是「上下文不等于工具真实输出」的一个实例，比价格那处
                #    更隐蔽：价格好歹还写了个值，编号是整栏都没写。
                ctxs.append(f"编号 {pid} 《{book}》 价格 {_tool_price(price)}")
            elif name == "get_post_detail":
                ctxs.append(
                    f"《{book}》 版次 {card.get('edition')} 价格 {price} "
                    f"成色 {card.get('condition_desc')} 状态 {card.get('status')} "
                    f"卖家原话：{card.get('raw_text')}"
                )
            # 其它工具（将来加的）先不收 —— 宁可漏一条上下文，
            # 也不要塞一段模型根本没见过的文本进去。
    return ctxs


def collect(client: httpx.Client, cases: list[Case], delay: float,
            limit: int | None) -> tuple[list, list]:
    """跑一遍 /chat，攒样本。返回 (样本, 被跳过的用例)。"""
    samples, skipped = [], []
    todo = cases[:limit] if limit else cases

    for i, case in enumerate(todo, 1):
        try:
            resp = call_chat(client, case.query)
        except Exception as e:
            print(f"[{i:>2}] {case.query:<20} ⚠️ /chat 失败：{type(e).__name__}: {e}")
            skipped.append((case, f"请求失败：{type(e).__name__}"))
            continue

        ctxs = contexts_from(resp)
        reply = (resp.get("reply") or "").strip()

        if not ctxs:
            # 模型这轮**没调检索工具**（直接答了，或者只问了一句澄清）。
            # 空上下文算不了忠实度，如实跳过并在报告里列出来 ——
            # 静默丢掉会让分母悄悄变小，最后那个平均分就不可信了。
            print(f"[{i:>2}] {case.query:<20} ⏭ 跳过：这轮没有检索上下文")
            skipped.append((case, "没有检索上下文（模型没调检索工具）"))
        else:
            print(f"[{i:>2}] {case.query:<20} ✅ 上下文 {len(ctxs)} 条 | "
                  f"回答 {len(reply)} 字 | 期望 {' / '.join(case.expect[:2])}")
            samples.append((case, resp, ctxs))

        if delay and i < len(todo):
            time.sleep(delay)

    return samples, skipped


def make_judge(model: str):
    """造裁判模型。

    走 OpenAI 兼容协议 —— DeepSeek 就是这个协议，所以 provider 仍是 "openai"，
    只是把 base_url 指过去。和 app/services/llm.py 里 get_llm() 的做法一致
    （那边用 ChatOpenAI(base_url=settings.llm_base_url)）。

    ⚠️ **必须传 `AsyncOpenAI`，不能传 `OpenAI`** —— 这是第一次真跑才暴露的坑，
       报错是 `TypeError: Cannot use agenerate() with a synchronous client`。

       原因在 ragas 内部：`InstructorLLM` 构造时会 `_check_client_async()` 判一次
       （base.py:780），把结果存成 `self.is_async`；同步客户端那就 `False`，
       然后 `agenerate()` 第一件事就是判它、不是异步直接抛（base.py:1092-1094）。
       而新版 collections 指标的 `score()` 是 `ascore()` 的同步包装，
       **内部走的正是 agenerate** —— 于是每一条都在这里撞死。

       为什么之前静态验证没发现：我只做了「构造指标 + 看签名」，那些都不触发调用。
       构造 `OpenAI` 和 `AsyncOpenAI` 都不会报错，**差别要真调一次才显形**。

       传异步客户端之后 `is_async=True`，`score()` 会走
       `_run_async_in_current_loop()`（base.py:1044）—— 那条路就是给同步调用方
       准备的，自己起 loop。所以下面的 `metric.score(**kwargs)` 写法不用改。
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key=_env("LLM_API_KEY"),
        base_url=_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    )
    # ⚠️ max_tokens 必须调大。ragas 的默认值是 **1024**（base.py:726），
    #    而评分 prompt 要求裁判逐条列出判断再给结论，上下文一多就写不完：
    #    第一次全量跑第 4 条（10 条上下文 + 10 行表格）直接
    #    `IncompleteOutputException` —— 输出被截断，整条判不出来。
    #    ragas 自己的提示就写着「结构化输出被截断就继续调大」（base.py:820-822）。
    #    4096 是它给的推荐值，也在 DeepSeek 的范围内。
    return llm_factory(model, provider="openai", client=client, max_tokens=4096)


def build_metrics(judge):
    """造三个指标。

    ⚠️ 从 `ragas.metrics.collections` 引（新版），不是 `ragas.metrics`（旧版）。
       旧版只吃 BaseRagasLLM，和 llm_factory 的 InstructorLLM 配不上（见开头理由 4）。

    ⚠️ 这三个**都不需要 embedding**（全是 LLM 裁判），所以 .venv-eval 里
       不用再装一份 torch + BGE。想加 answer_relevancy（那个要 embedding）的话，
       得先解决这件事。
    """
    from ragas.metrics.collections import (
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=judge),
        "context_precision_with_reference": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


def _judge_kwargs(metric, query: str, reply: str, ctxs: list, reference: str) -> dict:
    """按**这个指标实际接受的参数**组装入参。

    ⚠️ 这里有个实测踩到的坑：**三个指标的入参并不是同一套**。实测签名：

        faithfulness                      (user_input, response, retrieved_contexts)
        context_precision_with_reference  (user_input, reference, retrieved_contexts)
        context_recall                    (user_input, retrieved_contexts, reference)

    也就是说 `faithfulness` **不吃 reference**（它只问「有没有出处」，
    不问「该不该是这本书」），另外两个**不吃 response**（它们只看检索到的
    上下文，不看答案怎么写）。

    所以不能拿一套 kwargs 挨个喂 —— 会直接 TypeError（多余的关键字参数）。
    这里是**按签名过滤**，不是硬编码三份 if：ragas 以后改名或增减参数时，
    少传一个必填项照样会报错，不会静默算错。

    另外 `metric.score` 本身签名是 `(**kwargs)`（只是转发给 ascore），
    所以必须以 **ascore** 的签名为准，看 score 的签名等于什么都没看。
    """
    candidates = {
        "user_input": query,
        "response": reply,
        "retrieved_contexts": ctxs,
        "reference": reference,
    }
    accepted = inspect.signature(metric.ascore).parameters
    return {k: v for k, v in candidates.items() if k in accepted}


def score_all(samples: list, judge) -> list[dict]:
    """逐条算三个指标。★ **这一步会花钱**（DeepSeek 的 token）。

    自己写循环而不是调 `evaluate()`：新版 collections 指标**不是 `Metric` 实例**，
    进不了 `evaluate()` 的类型检查（实测 `isinstance(m, Metric)` 为 False）。
    这套 API 本来就是逐条 `ascore()` 的用法。

    单条失败**不中断整轮**，如实记成 None —— 一个指标判崩就让整轮白跑，
    是最浪费钱的失败方式。

    ⚠️ 用同步的 `score()` 而不是 `ascore()`：实测 `score()` 在**已经跑着
       asyncio 事件循环**的上下文里调会抛（它内部要起 loop）。
       本脚本是纯同步的，没问题；将来要嵌进 async 服务（比如加个评测接口），
       得换成 `await metric.ascore(...)`。
    """
    metrics = build_metrics(judge)
    rows = []

    for i, (case, resp, ctxs) in enumerate(samples, 1):
        reply = (resp.get("reply") or "").strip()
        reference = "；".join(f"《{n}》" for n in case.expect)
        scores: dict = {}
        reasons: dict = {}

        for key, metric in metrics.items():
            try:
                kwargs = _judge_kwargs(metric, case.query, reply, ctxs, reference)
                result = metric.score(**kwargs)
                scores[key] = float(result.value)
                reasons[key] = result.reason
            except Exception as e:
                scores[key] = None
                reasons[key] = f"判失败：{type(e).__name__}: {e}"

        shown = " ".join(
            f"{k.split('_')[0]}={'-' if v is None else format(v, '.3f')}"
            for k, v in scores.items()
        )
        print(f"[{i:>2}] {case.query:<20} {shown}")
        rows.append({"case": case, "reply": reply, "contexts": ctxs,
                     "scores": scores, "reasons": reasons})

    return rows


def _clip(text: str, n: int = 110) -> str:
    """回答太长了就截断 —— 汇总里只是给人一个定位的线索，不用全文。"""
    one_line = " ".join((text or "").split())
    return one_line if len(one_line) <= n else one_line[:n] + "…"


def report(rows: list[dict], skipped: list) -> dict:
    """打印汇总，返回汇总字典。"""
    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)

    summary: dict = {"n": len(rows)}
    for key, label in METRICS.items():
        vals = [r["scores"].get(key) for r in rows]
        ok = [v for v in vals if v is not None]
        # None 不算 0 分 —— 0.0 是「评了，得 0 分」，None 是「没评出来」，
        # 两者混在一起平均，得到的数谁也不敢用。所以分开报。
        summary[key] = round(sum(ok) / len(ok), 4) if ok else None
        summary[f"{key}_n"] = len(ok)
        print(f"  {label:<8}： {summary[key]}"
              f"   （{len(ok)}/{len(vals)} 条判出来了）")

    # ── 忠实度最低的几条 ────────────────────────────────────────────
    # 这一栏是**唯一**能看出「模型编了东西」的地方，所以要带 reason ——
    # 只看一个 0.6 的分数没法定位问题，得知道它认为哪句话没出处。
    # 只列**没满分**的：满分说明这一条没编造，列出来是噪音。
    # 全满分时不打这一段 —— 那本身就是好消息，不需要占版面。
    low = sorted(
        [r for r in rows
         if r["scores"].get("faithfulness") is not None
         and r["scores"]["faithfulness"] < 1.0],
        key=lambda r: r["scores"]["faithfulness"],
    )[:3]
    if low:
        print()
        print("  忠实度最低的几条（裁判认为有句子没出处的那几条）：")
        for r in low:
            # ⚠️ 实测 ragas collections 的 `MetricResult.reason` **经常是 None** ——
            #    别指望它给出「哪句话没出处」。没有就退回看回答本身：
            #    分数 + 回答一起看，人才判得出来是模型编的、还是上下文没还原全。
            why = r["reasons"].get("faithfulness")
            detail = why or f"（裁判没给 reason）回答：{_clip(r['reply'])}"
            print(f"    - {r['case'].query!r}（{r['scores']['faithfulness']:.3f}）{detail}")

    if skipped:
        print()
        print(f"  ⚠️ 跳过的用例（{len(skipped)} 条）—— **不计入上面的平均分**：")
        for case, why in skipped:
            print(f"    - {case.query!r}：{why}")

    print()
    print("  裁判 = DeepSeek，被测 = DeepSeek，**同族自评**。")
    print("  数字看趋势不看绝对值；检索侧的主指标仍以 eval_search.py 为准。")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Ragas 生成侧评测")
    ap.add_argument("--collect-only", action="store_true",
                    help="只采数据、不调裁判（不花钱）—— 先跑这个")
    ap.add_argument("--judge-model", default=_env("RAGAS_JUDGE_MODEL", "deepseek-chat"),
                    help="裁判模型（默认 deepseek-chat）")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help=f"每条之间的间隔秒数（默认 {DEFAULT_DELAY}，避开 /chat 限流）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    ap.add_argument("--save", default=None, help="把结果存成快照名")
    args = ap.parse_args()

    # ragas 在 import 时会打几条 deprecation 警告（它内部自己用了旧路径），
    # 看着像我们写错了。压掉，别让它盖住真正的输出。
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    cases = evaluable(CASES)
    print(f"可评用例 {len(cases)} 条"
          f"（负样本 {len(CASES) - len(cases)} 条不评生成侧指标，见脚本开头）")
    print()

    with httpx.Client() as client:
        try:
            health = client.get(f"{AI_SERVICE}/health", timeout=5).json()
            print(f"AI 服务 OK：{health}")
            # /chat 要读会话历史，历史在主服务里 —— 所以这个脚本比 eval_search.py
            # 多一个前置依赖：**Spring Boot 也必须起着**。
            if not health.get("book_service_reachable", True):
                print("⚠️  /health 报主服务不可达 —— /chat 会 503。先起 Spring Boot :8080。")
                return 1
        except Exception as e:
            print(f"❌ 连不上 AI 服务 {AI_SERVICE}（{type(e).__name__}: {e}）")
            print("   起服务：.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000")
            return 1

        samples, skipped = collect(client, cases, args.delay, args.limit)

        if not samples:
            print("\n❌ 一条样本都没采到，后面没得评。")
            return 1

        print()
        print(f"采到 {len(samples)} 条样本"
              f"{'（--collect-only，不调裁判）' if args.collect_only else ''}")

        # --collect-only 也要把样本打出来，否则「确认数据收得对」这件事没法做。
        if args.collect_only:
            print()
            for case, resp, ctxs in samples:
                print("=" * 78)
                print(f"查询：{case.query}   （期望：{' / '.join(case.expect)}）")
                print(f"回答：{(resp.get('reply') or '').strip()}")
                print("上下文（模型实际拿到的）：")
                for c in ctxs:
                    print(f"  - {c}")
            print()
            print("✅ 数据采集没问题的话，去掉 --collect-only 再跑一次（会调裁判）。")
            return 0

        rows = score_all(samples, make_judge(args.judge_model))
        summary = report(rows, skipped)

    if args.save:
        folder = ROOT / "scripts" / "eval_runs"
        folder.mkdir(exist_ok=True)
        payload = {
            "summary": summary,
            "skipped": [{"query": c.query, "why": w} for c, w in skipped],
            "cases": [
                {
                    "query": r["case"].query,
                    "expect": r["case"].expect,
                    "reply": r["reply"],
                    "contexts": r["contexts"],
                    "scores": r["scores"],
                    "reasons": r["reasons"],
                }
                for r in rows
            ],
        }
        path = folder / f"{args.save}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n快照已存：{path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
