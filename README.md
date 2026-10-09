<p align="center">
  <h1 align="center">智觅</h1>
  <p align="center">
    <b>一句话，找到你要的二手书</b><br>
    二手书群帖子 → 结构化抽取 → 混合检索（BM25 + 向量 + RRF）→ cross-encoder 精排 → 会自己调工具的对话 Agent<br>
    Java 24 / Spring Boot 4 主服务 + Python 3.14 / FastAPI AI 服务
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Java-24-orange" alt="Java">
  <img src="https://img.shields.io/badge/Spring_Boot-4.1.1-green" alt="Spring Boot">
  <img src="https://img.shields.io/badge/Python-3.14-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

**智觅**要解决的是：大学二手书群里的帖子都是这种一句话 ——
「带笔记的线代，15 出，有意私聊」。人话能看懂，机器读不了。
这个项目把这类句子抽成结构化字段（书名、价格、有没有笔记、成色），
建混合索引（BM25 + 向量）做语义检索，上面再套一个会自己决定查不查库的对话 agent。

技术上是两条链路：RAG（抽取 → 混合检索 → 精排）和 Agent（tool calling）。
CRUD 那部分没什么好讲的。

```
用户：「带笔记的线代，30 块以内」
  │
  ▼  模型自己拆参数，自己决定调哪个工具、调几次
  search_books(query="线性代数", has_notes=True, price_max=30)
  get_post_detail(post_id=...)
  │
  ▼
「《线性代数》有笔记，8 元，还在售。另外几本也有笔记，价格 8~15 元。」
```

---

## 目录

- [演示](#-演示)
- [架构](#-架构)
- [技术栈](#技术栈)
- [快速开始](#-快速开始)
- [接口一览](#接口一览)
- [关键设计决策](#关键设计决策)
- [文档](#文档)
- [目录结构](#目录结构)
- [开发日志](#开发日志)
- [常见问题](#-常见问题)

---

## 📸 演示

启动后打开 `http://localhost:8080`，是个聊天界面。

问「有笔记的线性代数，30 以内」，agent 自己去调 `search_books`，把书名、价格、成色列出来。
再追问「8 块那本还有吗，成色怎么样」——8 块的有两本，它会先反问要哪一本，确定了再去查。
这两句是实测能触发 `search_books` → `get_post_detail` 接力的问法。

页面上每次工具调用单独占一行，带工具名、参数、返回条数。下面第二张图是重点：
只放最终回答的话，看的人分不清「agent 自己决定查了库」和「硬编码了一段话」。

![聊天界面](docs/screenshots/01-chat.png)

![Agent 的工具调用与书卡](docs/screenshots/02-tool-calls.png)

![AI 服务不可用时的提示](docs/screenshots/03-offline.png)

完整操作步骤见 `docs/演示流程.md`。

---

## 🏗 架构

```
            浏览器  ── POST /api/chat ──▶  主服务 :8080
                                              │
                                              │ HTTP 转发
                                              ▼
                                        AI 服务 :8000
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                     ▼                     ▼
                   LLM 信息抽取          BGE 向量化            Agent 循环
                   （M1）                + Chroma（M2）        （M3）
                                              │                     │
                                              └──── 取业务数据 ─────┘
                                                       │
                                                       ▼
                                            回调主服务 REST API ──▶ MySQL
```

### 两个服务怎么分的

AI 服务不直连业务数据库。它要用帖子数据，就走主服务的 REST API。
这样它不持有任何状态，可以随便重启、随便扩副本，代价是每次多一跳 HTTP。

（LangChain 自带的会话历史有 InMemory / SQLite / Redis 三种，都要求自己存，
所以一个都没用上。历史存在主服务的 MySQL 里。）

反过来，校验、事务、状态流转也都归主服务。AI 服务只做「理解」这一件事：
文本进去，结构或者回答出来。

拆成两个服务不是必须的，但拆开之后每一边的职责都说得清：主服务是业务系统
（帖子、会话、事务），AI 服务是能力层（抽取、检索、对话），换模型或者换向量库
只动一边。

agent 的工具回调最能说明这个划分：代码上看是个普通函数，实际是一次跨进程 HTTP 调用。

---

## 技术栈

| | 技术 | 版本 | 备注 |
|---|---|---|---|
| 业务服务 | Java / Spring Boot | 24 / 4.1.1 | JDK 23+ 需显式配注解处理器（见 `pom.xml`） |
| | MyBatis-Plus | 3.5.17 | 必须用 `spring-boot4-starter`；`IService` 已从 `extension` 移到 `spring` 包 |
| | MySQL | 8.0 | |
| | springdoc | v3 | Boot 4 只能 v3，v2 起不来 |
| AI 服务 | Python | 3.14 | |
| | FastAPI + uvicorn | | |
| | LangChain | 1.x | 1.x 已移除 `AgentExecutor`，入口是 `create_agent` |
| | DeepSeek | `deepseek-chat` | OpenAI 兼容接口 |
| | BGE-small-zh-v1.5 | 512 维 | 本地跑，CPU 即可。双塔，负责**粗排召回** |
| | bge-reranker-base | | cross-encoder，负责**精排重排**。1.1GB，不进仓库（见 `.gitignore`） |
| | rank-bm25 + jieba | | 稀疏通道。中文没空格，必须先分词 |
| | GLM-4V-Flash | | 多模态「以图搜书」。**可选** —— 不填 key 就只有这一个接口不可用 |
| | Chroma | | 嵌入式向量库，跑在进程内 |
| | Redis | 8.10 | **可选**：限流计数 + 会话历史缓存。存的是运维态，不是业务数据 |
| | pytest | | 单测 168 条（秒级、不依赖外部服务）+ 集成 40 条 |
| | GitHub Actions | | 3 个 job：单测 / Maven 编译打包 / 真 Redis 集成测试。**不真调 LLM，不花钱** |
| 前端 | 原生 HTML + fetch | | 单文件，无构建步骤 |
| 部署 | Docker + Compose | | 一键起 MySQL + Redis + 双服务（⚠️ 见下方「关于 Docker」） |

**没有用到的**：消息队列。

> **关于 Docker**：`docker-compose.yml` 和两个 `Dockerfile` 都写好了，
> 但**没有在真机上验证过**（写的时候本机没装 Docker Desktop）。
> 不吹成「已支持容器化部署」——照着两边 Dockerfile 的实际内容推的配置，
> 第一次用请 `docker compose up --build` 前台起、盯着日志。
> 见 [docker-compose.yml](docker-compose.yml) 头部那两条「坑」。

---

## 🚀 快速开始

需要三个进程：**MySQL → 主服务 → AI 服务**，外加一个浏览器。

> 下面的命令以 **Windows** 为准（`net start MySQL80`、`.venv/Scripts/python.exe`）。
> macOS / Linux 只有这几处不同，其余照抄：
> - 起 MySQL：`brew services start mysql`（macOS）或 `sudo systemctl start mysql`（Linux）
> - 用 Python：`.venv/bin/python` 而不是 `.venv/Scripts/python.exe`

### 情况 A：数据库里已有数据（约 1 分钟）

```bash
# ① MySQL（Windows 上是手动启动的服务，重启机器后要重新起）
net start MySQL80

# ② 主服务
cd book-service && mvn spring-boot:run          # :8080，约 2 秒起来

# ③ AI 服务
cd ai-service
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000    # :8000

# ④ 打开浏览器
#    http://localhost:8080
```

### 情况 B：从零（首次，约 10 分钟）

```bash
# ① 建库建表
mysql -uroot -p < book-service/src/main/resources/db/schema.sql

# ② 填两个配置
cp book-service/src/main/resources/application-local.yml.example \
   book-service/src/main/resources/application-local.yml     # 数据库密码
cp ai-service/.env.example ai-service/.env                   # LLM_API_KEY

# ③ 装依赖（macOS/Linux 把下面两处 .venv/Scripts/python.exe 换成 .venv/bin/python）
cd ai-service
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
# 注意：torch 要装 CPU 版。默认 pip 会拉 CUDA 版（2GB+），本机没 N 卡就是白下。
#    命令见 requirements.txt 顶部注释。

# ④ 造种子数据（1 万条模拟帖子，要花模型调用；约 7 分钟）
#    --count 是「库里总共要有多少条」，不是「这次生成多少条」
#    —— 中途挂了直接重跑，它会自己补差额，不会灌重复数据
.venv/Scripts/python.exe scripts/gen_mock_posts.py --count 10000

# ⑤ 起两个服务（同情况 A 的 ②③）

# ⑥ 抽取：反复调这个接口直到 remaining 为 0（1 万条约 6 分钟）
curl -X POST http://localhost:8080/api/posts/extract-pending

# ⑦ 建向量索引（见下面的坑 1，这步不能跳；约 25 秒，不花 LLM token）
curl -X POST http://localhost:8000/index/build
```

#### 还要单独下一个模型：rerank 用的 cross-encoder

`models/` 在 `.gitignore` 里（1.1GB，GitHub 单文件上限 100MB，硬推会**整个仓库推不上去**）。
所以 clone 之后要自己下一次：

```bash
.venv/Scripts/python.exe scripts/download_reranker.py
```

⚠️ 这一步**不能像普通模型那样 `snapshot_download`** —— hf-mirror.com 不支持 HuggingFace
新的 Xet 存储协议，即使设了 `HF_HUB_DISABLE_XET=1` 也会被 302 到
`cas-bridge.xethub.hf.co` 然后 401 / 读超时（实测）。所以脚本走的是
`/api/models` 列文件 + `/resolve/main/` 逐个下，绕开 Xet。

**不下载也能跑**：rerank 加载失败会降级成「不精排」，检索照常工作，
只是排序退化回 RRF —— 而且这一点在 `GET /health` 的 `rerank.failed` 里看得见，
不会静默降级。

#### 三个可选依赖（不配也能跑，只是对应功能不可用）

| 依赖 | 不配会怎样 | 怎么配 |
|---|---|---|
| **Redis** | 限流放行、会话缓存直读 —— 服务照常，只是没闸门 | 起 Redis，`.env` 里填 `REDIS_URL`（带密码的完整 URL） |
| **视觉模型 key** | 只有 `POST /search/by-image` 回 503，其余全正常 | 注册智谱 GLM-4V-Flash（免费额度），`.env` 里填 `VISION_API_KEY` |
| **rerank 权重** | 降级为「不精排」，排序退回 RRF | 跑 `scripts/download_reranker.py`（见上） |

三个都是**可降级且降级可见**：`GET /health` 里分别有 `redis.connected`、
`vision.configured`、`rerank.failed` 三栏。这不是凑巧 —— 是这个项目对
「静默失败」的一贯处理：**可以退化，但不能让你不知道它退化了**。

### 情况 C：Docker 一键起（⚠️ 未经实测）

```bash
# 项目根目录放一个 .env（compose 只读根目录的 .env，不读 ai-service/.env）
echo "LLM_API_KEY=sk-你的key" > .env

docker compose up -d --build
# 打开 http://127.0.0.1:8080
```

起来的是 **MySQL + Redis + 双服务** 四个容器（MySQL 建的是**空表**，
要试搜书还得灌数据 + 建索引，见 [docker-compose.yml](docker-compose.yml) 头部）。

> ⚠️ **这份配置没有在真机上跑过** —— 写它的时候本机没装 Docker Desktop。
> 配置是照着两边 Dockerfile 的实际内容推的（Python 版本、torch CPU 源、
> 模型挂载路径都对齐了），但**没有实跑验证**。第一次请用
> `docker compose up --build`（前台）起，报错直接看得见。
> 已知最容易出问题的两处已经写在 compose 的头部注释里了。

### 数据是怎么造出来的（三步，都能重跑）

```
scripts/fetch_book_catalog.py   抓豆瓣公开列表页 → data/book_catalog.json（2651 本真实出版物）
        ↓                       一次性动作，产物已经跟着仓库走，平时不用重跑
scripts/gen_mock_posts.py       按书目让 DeepSeek 写卖书帖 → MySQL（book_post）
        ↓                       「不放回抽牌」保证每本书被抽到的次数均匀
scripts/backfill_extract.py     批量抽取（一次请求 20 条）→ 回填结构化字段
```

**数据性质（面试会被问，别含糊）：书目是真的，帖子是编的。**
书目来自豆瓣公开列表页的真实出版物元数据；帖文全部由 DeepSeek 生成，
不含任何真实交易、真实用户或个人信息。

**为什么不用「30 本手写种子随机挑几本」**：那样生成 1 万条会退化成
「每本几百份」的复读机 —— 任何查询都能命中，检索评测直接失去意义
（top-3 命中率飙到 100%，测不出任何东西）。现在 1 万条帖子摊到 2651 本书上，
约每本 4 份，接近真实二手书市场的密度。

### 四个必踩的坑

**1. 新克隆后一定要跑 `/index/build`。**
向量库落在 `ai-service/data/`，而这个目录在 `.gitignore` 里 ——
**新克隆的仓库索引是空的**。不建索引的话 `search_books` 永远返回 0 条，
表现成「agent 说找不到任何书」，很容易误判成代码坏了。
（写入用的是 upsert，重复调这个接口是幂等的。）

**2. 服务间调用写 `127.0.0.1`，不要写 `localhost`。**
Windows 上 `localhost` 会同时解析出 IPv4 和 IPv6，而 uvicorn 只绑了 IPv4，
客户端可能先试 `::1` 然后失败 —— 表现成「服务明明起着却连不上」。

**3. 改完 Java 代码要重启。** 项目没装 devtools。

改 `src/main/resources/static/` 下的前端文件**也要重启**：页面实际读的是
`target/classes/static/` 里那份**副本**，不重启的话浏览器拿到的还是旧页面。
（`spring-boot:run` 的 `addResources` 默认是 `false`，不会把源码目录直接挂上 classpath。）

**4. 加接口时，辅助函数别插在装饰器和 `def` 之间。**

```python
@app.post("/search", ...)     # ← 装饰器抓的是**紧跟其后**的那个 def
def _to_hits(...): ...        # ← 它被抓走了
def search(...): ...          # ← 这条没被注册
```

这么写，`/search` 会绑到 `_to_hits` 上，**每个请求都回 422**。
而它当时**完全静默**：服务正常启动、`/docs` 正常打开、78 条单测全绿
（没有一条从 HTTP 层打 `/search`），连之前的手工验证也「有效」——
因为那是改动**之前**启动的进程还在跑。

最后是补测试时才抓出来的。所以现在有一层零依赖的
[tests/test_routes.py](ai-service/tests/test_routes.py)：把「路径 → 处理函数名」
这张表钉死，谁插错位置谁就红。

---

## 接口一览

字段细节一律以 Swagger 为准（从代码注解自动生成，**代码改了文档跟着改**）：

| 服务 | Swagger UI | OpenAPI spec |
|---|---|---|
| AI 服务 | http://localhost:8000/docs | http://localhost:8000/openapi.json |
| 主服务 | http://localhost:8080/swagger-ui.html | http://localhost:8080/v3/api-docs |

两个 spec 都能直接导入 Apifox / Postman / YApi。

### 主服务（:8080）

| 方法 | 路径 | 干什么 |
|---|---|---|
| POST | `/api/chat` | **前端唯一要调的接口**。转发给 AI 服务，不解析 body |
| GET/POST | `/api/chat/{sessionId}/messages` | 会话历史读写。**调用方是 AI 服务，不是前端** |
| GET | `/api/posts` | 帖子列表（`?extractStatus=DONE` 给建索引用） |
| GET | `/api/posts/{id}` | 帖子详情（agent 的 `get_post_detail` 回调它） |
| POST | `/api/posts` | 发布帖子（只传 `userId` + `rawText`） |
| POST | `/api/posts/extract-pending` | 批量抽取待处理帖子，反复调到 `remaining` 为 0 |

### AI 服务（:8000）

| 方法 | 路径 | 干什么 | 花模型调用 |
|---|---|---|---|
| GET | `/health` | 探活。**顺带探主服务**（真打一次，1.5 秒超时），并暴露 rerank / vision / redis 各自的状态 | 否 |
| GET | `/usage` | 进程启动以来的 **token 用量 + 估算成本**。`calls_without_usage > 0` 说明统计不准了 | 否 |
| POST | `/chat` | **agent 对话**，带会话记忆。限流 20 次/分钟/IP，超了回 429 | 是（每轮 1~N 次） |
| POST | `/search` | 混合检索 + 精排。`rerank: false` 可关掉精排做 A/B | 否 |
| POST | `/search/by-image` | **以图搜书**（一张图可以有多本）：图片 → 视觉模型读出**一批**书名 → 逐本复用 `/search`。限流 10 次/分钟/IP | 是（1 次视觉 + N 次检索） |
| POST | `/extract` `/extract/batch` | 帖子 → 结构化字段。共用「60 次/分钟」一个桶 | 是 |
| POST | `/index/build` | 建向量索引 | 否 |

---

## 关键设计决策

完整论述在 [需求与接口](docs/需求与接口.md)，这里只列结论和理由：

| 决策 | 理由 |
|---|---|
| 检索文本不含价格和成色 | 向量只管「是哪本书」，条件过滤交给结构化字段。实测过：书名在检索文本里重复出现反而拉低分数（0.6258 → 0.6054 → 0.5953）；用「未知」占位填 `None` 也会毒化检索，全体掉 0.03~0.11 |
| 检索必须有相似度阈值 | 向量检索永远会返回 top_k 条，哪怕库里根本没有这本书。阈值不是可以随便调的参数，它决定了检索结果能不能信。**但它在 1 万条语料上已经到极限了**：正例下界 0.6187（「同济高数」）和能拦住的最高越界查询 0.6034（「核工程」→《Java核心技术》）只差 **0.0153** —— 再抬一点正例就死 |
| 上 cross-encoder 精排（粗排 20 条 → 逐对打分 → 取 top_k） | 双塔比夹角，分不出「这个词是不是真的指同一件事」；cross-encoder 把 query 和文档拼在一起，能看见字面的交叉。**但实测下来它改善的是排序不是过滤**：正例 top-1 15/16 → 16/16，越界查询一条没多拦 |
| 精排分只用来排序，**不做过滤**（`rerank_min_score = None`） | 精排分区间是正例 [0.6987, 0.7311] / 越界 [0.5324, 0.7151] —— 仍重叠。0.70 看着能拦 6/8 且零误杀，但离正例下界只有 0.0013：16 条样本在 0.03 宽的空隙里切一刀是过拟合。而且代价不对称 —— 漏一条越界用户看到一本沾边的书，误杀一条正例用户搜不到本来有的书 |
| 以图搜书走「图 → 书名 → 复用文本检索」，**不是**「图 → 向量比相似度」 | 语料是纯文本的，一条封面图都没有 —— 图像检索要求每本书有封面向量，这份数据里**建不出来**。而认出的书名直接进 `search()`，BM25、RRF、精排、阈值全部自动生效。拍的是书（封面/扉页/一排书脊），拍图这个动作传递的本来就是**文本信息**（哪天要做「找同款封面」才该换 CLIP） |
| 以图搜书抽的是一**批**书名，不是一个 | 二手书群里流传的不是干净封面照，是**一个书架上十几本书的书脊**。只回一个书名等于「随便挑了其中一本」，而且调用方看不出另外十几本被丢了 —— 静默降级。旧版 `_normalize_title` 里那句 `splitlines()[0]` 更狠：模型老老实实列了五行，被它切掉四行 |
| 识别的上限（30）和检索的上限（20）**分开设**，差额如实返回 | 识别是一次模型调用，认 1 本和认 30 本一样贵；检索是每本一次完整链路（BM25 + 向量 + RRF + 精排），全是真算力。所以「认得出多少」放宽、「搜多少」收紧。**`len(results) < len(recognized_books)` 就是那个差额** —— 默默丢掉一半装作只认出 20 本，比慢一点糟得多 |
| 两个上限都被**真实照片**往上顶过一轮（20/10 → 30/20） | 真图里有 17 本：识别上限 20 截掉 40%、检索上限 10 只搜六成。**上限设得比真实输入还紧，等于把闸门装在了水管中间** —— 而「静默丢掉 7 本」比「多等 1 秒」严重得多 |
| 代价如实说：整次请求**实测 10~12 秒** | 视觉调用约 6 秒 + 17 本**串行**检索约 5 秒（单次 `/search` 实测约 280ms，含精排）。**这很慢。** 真正的解法是这批检索并发或改后台任务轮询 —— **没做**（精排是 CPU 密集，线程池能不能真并行没验证过，不能说「改并发就好了」） |
| 过滤「不是书名」的行，规则是被**真图**逼出来的，不是想出来的 | 第一轮真图验收：20 条输出里 **6 条不是书名**（`吴传生主编`、`北京大学出版社`、`学生用书`、`（2021年版）`…）。当时已有的规则要求行首是「作者：」这种**带冒号**的标签，而模型抄的是书脊上的**裸正文**，一条都没拦住。**规则没错，错在它假设了输入里有标签。** 补的三条：署名/出版单位后缀、整行只有括号、通用标注停用词 —— 都只收「整行就是它」的情况，`C++ Primer（第5版）` 不受影响 |
| 视觉模型的输出只当**检索词**，不进任何 prompt | 图片是不可信输入 —— 一张图完全可以印着「忽略以上指令」。本实现里它不可能造成危害，因为识别结果只有一个出口：`search(query=book)`。**「不可信输入终止在检索边界」是设计出来的性质，不是碰巧**。返回一串书名没有削弱它：越权的判断标准是「有没有回到 prompt 里」，不是「返回了几个字符串」 |
| 限流用**滑动窗口**（Redis ZSET），不用固定窗口 | 固定窗口有个经典漏洞：0:59 打满 → 1:00 计数器归零 → 再打满，**两秒内实际放行两倍**。滑动窗口每次都先清掉窗口外的记录再数，任意时刻往前看 60 秒都不会超。清理+计数+判断+写入这四步封装在一个 Lua 脚本里 —— 分开发四条命令的话，并发下会同时读到 count=19 并同时放行 |
| 限流的身份取 **TCP 对端 IP**，故意不看 `X-Forwarded-For` | 那个头客户端可以随便写，`curl -H 'X-Forwarded-For: ...'` 换个值就换一个桶 —— 等于给限流开了个**一行命令就能绕过**的后门。只有在明确有可信代理时才该信它（且应由框架的 `--proxy-headers` 决定，不是手工读头） |
| Redis 挂了 **fail-open**（放行）+ warning 日志 | Redis 在这个项目里是可选依赖，为限流把 `/chat` 打死是本末倒置。fail-closed 属于计费/风控场景。但**放行不等于静默**：`/health` 里的 `redis.connected` 就是那条痕迹 |
| 会话缓存用 **cache-aside + 写时失效**，不用写穿 | 写穿要求缓存**复刻主服务的截断逻辑**（`limit=20` 意味着 append 后还得再截），哪天主服务改了规则，缓存里就留了一份「和接口不一样的真相」。失效则保证**缓存里要么是主服务真返回过的东西，要么不存在** —— 缓存永远不可能比主服务更聪明 |
| `/search` `/index/build` 不接限流，`/{chat,extract,extract/batch,search/by-image}` 接 | 限流限的是**成本**不是流量。检索走本地 BGE，只烧 CPU；给演示时狂点搜索框加闸门纯属自找麻烦。`/extract` 两个接口**共用**一个桶 —— 分开算的话「单条打满 + 批量打满」就是两倍上限 |
| `/search/by-image` 按成本归到**限流**那一类，不按 URL 前缀归到检索 | 名字长得像检索，实际第一步就是一次视觉调用（比文本模型慢一个数量级），后面还跟着最多 20 次检索。**它曾经漏了整整一个里程碑** —— 因为它被顺手归进「检索接口不限流」那类。限额 10 次/分钟，比 `/chat` 还紧 |
| 索引里只存 id 和列表页字段 | 向量库当索引用，不当副本。详情回主服务取，省掉双写一致性的麻烦 |
| 三态字段（有没有笔记、价格）不在出口压成两态 | 卖家「没提」和「说了没有」不是一回事。用 `bool(None)` 会把「没提」压成「没有笔记」，等于替卖家撒谎（1 万条语料上实测：**6533 条 = 65.3% 是「未提及」**，会被全部误判） |
| 每轮只存两条历史 | agent 一轮跑完返回的是整段会话，整个存回去会一轮比一轮重复。只留 `HumanMessage` 和最后那条 `AIMessage` —— 中间的工单和回执必须成对，只留一半下一轮直接报错 |
| `recursion_limit = 12` | 实测 N 轮工具调用 = 2N+1 个节点，上限至少得 2N+2。`MAX_ROUNDS=5` 反推出 12（默认 10007，等于没有护栏） |
| 前端单文件、不引框架 | 这一页只做「发请求 + 渲染 JSON」。上构建工具链对要演示的东西（agent 会不会自己调工具）没任何帮助 |
| token 统计挂在 **LLM 客户端的 callback** 上，不在调用处数 | 抽取用 `with_structured_output(...).invoke()` 直接返回 `BookInfo`，**usage 在解析那一步就被丢了**。要拿到它只有两条路：自己解析 tool_calls（重写整条链路）或者挂 `BaseCallbackHandler`（一个字的调用代码都不用改）。选后者 |
| `/usage` 里专门留了 `calls_without_usage` 一栏 | 「统计失效」最容易伪装成「成本为 0 元」—— provider 换个字段名，你会看到一个漂亮干净的 0。这一栏 > 0 就是在说「账目不全，别信这个数」 |
| LLM 客户端补上 `timeout=60` / `max_retries=2` | 这是全项目**唯一一个没有超时保护的出网调用**，之前对端一卡，请求就挂着占住 worker。重试只给 2 次 —— DeepSeek 的 429 是账号级的，重试多了只会把限流打得更死 |
| `/health` 真去探主服务（而不是只回一个配置值） | 之前那一栏是 `book_service: "http://127.0.0.1:8080"` —— 那是**配置**，不是**状态**。主服务挂了它照样显示得好好的。现在 1.5 秒超时真打一次；LLM 那条链路**故意不探**（每次探活烧一次 token 太贵，key 失效在真调用时自然暴露） |
| 请求耗时用 middleware 记，不在每个接口里手写 | 接口有 8 个，手写就是 8 处重复、加第 9 个必漏。middleware 还能覆盖 404 / 422 这些**没进到接口函数**的请求 —— 而「某个路径一直 404」恰恰最需要日志 |
| CI 只跑「不需要真 LLM」的那两层 | 单测 + Redis 集成。检索质量测试（`test_search_eval.py`）要 1 万条语料，而灌数据必须真调 LLM —— 把 key 塞进 CI secret、每次 push 烧一遍数据，是拿钱换一个没必要的绿勾。**CI 绿 ≠ 接口能调通**，这句话写进了 workflow 的头部注释 |

---

## 文档

| 文件 | 内容 |
|---|---|
| [项目全景.md](docs/项目全景.md) | 模块地图、设计决策、关键数字速查表 |
| [需求与接口.md](docs/需求与接口.md) | 每个里程碑的目标、接口契约、决策点、验收标准 |
| [演示流程.md](docs/演示流程.md) | 2 分钟演示脚本、演示前清单、翻车预案 |

---

## 目录结构

```
agent-book/
├── book-service/                  # Spring Boot 主服务（IDEA 打开）
│   └── src/main/
│       ├── java/com/book/
│       │   ├── client/            # 调 AI 服务的客户端（唯一知道 AI 服务存在的地方）
│       │   ├── controller/        # REST 接口
│       │   ├── dto/               # 请求 / 响应体
│       │   ├── entity/ mapper/ service/
│       │   └── exception/         # 统一异常出口
│       └── resources/
│           ├── db/schema.sql
│           └── static/index.html  # 前端页面
├── ai-service/                    # Python AI 服务（VS Code 打开）
│   ├── app/
│   │   ├── agent/                 # tools.py（工具定义）+ loop.py（agent 循环）
│   │   ├── clients/               # 调主服务的 HTTP 客户端
│   │   ├── services/              # extract / embedding / vector_store
│   │   │                          # + lexical（BM25）/ search（融合）/ rerank（精排）
│   │   │                          # + vision（以图搜书）/ usage（token 成本）
│   │   │                          # + redis_client / ratelimit / session_cache
│   │   ├── main.py                # FastAPI 入口
│   │   └── schemas.py             # 请求 / 响应模型
│   ├── tests/                     # 单测 150（无外部依赖）+ 集成 39（真连 MySQL/Redis）
│   ├── scripts/                   # 开发脚本（验收 / 实验 / 复现）
│   ├── Dockerfile
│   └── models/                    # 本地模型权重（.gitignore，1.1GB 不进仓库）
├── docker-compose.yml             # 一键起 MySQL + Redis + 双服务（⚠️ 未经实测）
└── docs/                          # 项目全景 / 需求与接口 / 演示流程
```

`ai-service/scripts/` 里是开发期脚本，服务跑起来用不到它们，大致分四类：

- `try_*.py`：单点验证（`try_search_books.py`、`try_tool_call.py` 等），
  调一次接口看返回，用来定位问题
- `try_continue_bug.py`、`try_post_ids_bug.py`、`try_framework_scan_bug.py`、
  `try_hallucinated_tool.py` 这几个是复现具体 bug 的最小用例，修完之后留着当回归脚本
- `eval_search.py`、`compare_agent.py`、`walk_agent_loop.py`：批量评测和循环跟踪
- **量化「某个改动到底有没有用」的四个**：
  `compare_bm25.py`（混合 vs 纯向量）、`analyze_threshold.py`（证明余弦阈值到顶了）、
  `tune_rerank.py`（精排分能不能分开 + 扫阈值）、`download_reranker.py`（下模型）

每个脚本的文件头都写了**花不花钱**（是否真的调模型）和跑法，可以先挑不花钱的跑。

---

## 开发日志

| | 做了什么 | 实测结果 |
|---|---|---|
| **M1** | 抽取落库 | 10000/10000 全部 DONE，零失败，3.3 分钟（批量抽取，一次请求 20 条） |
| **M2** | 混合检索（向量 + BM25/RRF） | 21 条用例（正例 16 / 负例 5）全过，阈值 0.61。另有 8 条「已知拦不住」单独记账，见下面「检索到顶了」 |
| **M2.5** | rerank 精排（cross-encoder） | top-1 命中率 **15/16 → 16/16**；但**过滤一条没多拦**。诚实结论见「检索到顶了」第 3 节：改善排序，不改善过滤 |
| **M2.6** | 多模态「以图搜书」 | 图 → 视觉模型读出**一批**书名 → 逐本复用 M2/M2.5 的检索链路。**58 条单测 + 3 条限流接线**。已过**真实书架照**验收：改 prompt 与过滤规则前，20 条输出里 **6 条根本不是书名** —— 模型抄的是书脊上的**裸署名**，而旧规则只防「作者：」这种**带冒号**的形式，一条都没拦住；改后 16 条全是书名、全部搜到。⚠️ 残留问题（版次粘连、两本书名被合成一个、遮挡漏认、**两次跑结果不一样**）如实记在 `docs/需求与接口.md` §8.5，**没修** |
| **M3** | Agent（手写循环 + 框架对照两版） | 验收 6/6。框架版反而多 5 行（38 vs 43），省下的主要是协议坑那部分 |
| **M4** | 打通前端 | 5/5。单文件页面 + `POST /api/chat` 转发 |
| **M4.5** | Redis：限流 + 会话缓存 | 滑动窗口（Lua 原子）+ fail-open 降级；`/chat` 20 次/分、`/extract` 60 次/分（共用一个桶）。缓存实测：命中不回源、写完即失效、不同 `limit` 不串味。**降级路径也测过** |
| **M5** | 打磨 | 边界/压测跑完（13 个畸形输入全 400；30 并发 `/search` 逐字节一致）；补齐文档 |
| **M6** | pytest + CI | **168 单测 + 40 集成**（单测 16 秒、零外部依赖）；GitHub Actions 三个 job。⚠️ 上 CI 之前先抓到一个**接口全挂却完全静默**的 bug，见「三个必踩的坑」第 4 条 |
| **M6.2** | token 成本 | `BaseCallbackHandler` 挂 LLM 客户端，`GET /usage` 出用量和估算成本。实测：单条抽取 1625 入 + 142 出 ≈ 0.0044 元；3 条并发批量正确记到 3 次调用 |
| **M6.3** | 可观测 | 请求耗时 middleware（响应头带 `X-Process-Time-Ms`）；`/health` **真探主服务**（原来那一栏只是配置值，主服务挂了照显正常）；LLM 客户端补 `timeout` / `max_retries` |
| **M6.4** | Docker | 两个 Dockerfile（多阶段 / CPU-only torch）+ `docker-compose.yml`。⚠️ **未经实测** —— 本机没装 Docker Desktop，只保证语法正确、配置互相自洽 |
| **M7** | 教材 PDF 切块问答 | 弹性目标，未开始 |

M1–M6 是计划内；M7 是我给自己留的弹性目标，没做也不影响前面几块。
每一块做完都留了可复现的验证结果（就是上面那一列），不是"写完了"就划掉。

---

## 检索到顶了 —— 1 万条语料上的实测上限

语料从 220 条扩到 1 万条之后，检索链路的**天花板**第一次露了出来。这一节记的是
实测数据，不是设想；复现方式写在每段下面。

### 1. 单一余弦阈值已经分不开正例和负例

阈值不是拍脑袋定的，是在评测集上**扫**出来的（`scripts/eval_search.py --min-score`）：

| 阈值 | 正例 top-3 | 负例 | 拦住的越界查询 |
|---|---|---|---|
| 0.58 | 16/16 | 4/5 | 0/8 |
| 0.60 | 16/16 | 5/5 | 0/8 |
| **0.61** | **16/16** | **5/5** | **1/8** |
| 0.62 | 15/16 | 5/5 | 3/8 |
| 0.63 | 14/16 | 5/5 | 4/8 |
| 0.65 | 12/16 | 5/5 | 4/8 |

0.61 是「正例全活」的上限，抬到 0.62 正例就开始死。**这不是找到了一个好阈值，
是刚好卡在刀刃上**：最低正例「同济高数」0.6187，而拦不住的那批
（教育学 0.6158 / 环境工程 0.6199 / 国际关系 0.6233 / 纺织 0.6959）
**全都落在正例区间里**。

### 2. 拦不住的是些什么（`scripts/eval_search.py` 的 `KNOWN_LEAKS`，每次评测重测）

```
纺织       0.6959 → 《这个词是怎么来的》
农业经济学   0.7301 → 《经济地理学》
法学概论     0.7004 → 《法学方法论》
建筑学      0.6707 → 《建筑物理》
国际关系     0.6233 → 《深度关系》
环境工程     0.6199 → 《Linux环境编程》
教育学      0.6158 → 《数学》
核工程      0.6034 → 《Java核心技术》   ← 0.61 能拦住它，其余 7 条拦不住
```

根因是**双塔（bi-encoder）的固有缺陷**：query 和文档各自压成一个向量再比夹角，
夹角只反映「整体语义方向」，分不出「这个词是不是真的指同一件事」。

### 3. 上了 rerank 之后：**改善的是排序，不是过滤**

这是全文最反直觉的一段。加了 cross-encoder（M2.5）之后：**上面 8 条越界查询仍然只拦下 1 条**。
指望 rerank「修好」这张表的人会失望。

但它确实有用，只是用处不在这儿：

| | 无 rerank | 开 rerank |
|---|---|---|
| 正例 top-3 命中率 | 16/16 | 16/16 |
| **正例 top-1 命中率** | **15/16** | **16/16** |
| 反例通过率 | 5/5 | 5/5 |
| 越界查询拦下 | 1/8 | 1/8 |

第 1 名变了 3 条，**全部不劣**：`有没有高数书`「数学」→「高等数学」；
`新视野大学英语`「…英语2」→「…英语」；`严蔚敏`「数据结构」→「算法分析」
（两本都是严蔚敏写的，查库确认过）。

复现：`eval_search.py --no-rerank --save 无rerank`，再 `eval_search.py --save rerank --diff 无rerank`。

> ⚠️ **top-3 命中率在这一栏是没有分辨力的** —— 它早就在 1 万条语料上饱和到 16/16 了，
> 任何改动都翻不动它。只盯它就会得出「rerank 没用」的假结论。
> 所以评测脚本里加了 top-1 那一行（`summarize()`），它才是用户实际看到的那一条。

**为什么拦不住的那些还是拦不住？** 拆开看其实是两类（详见 `tests/` 同级的
`scripts/eval_search.py` 里 `KNOWN_LEAKS` 的注释块）：

- **模型确实看穿了，只是没低到能安全切**：「核工程」精排 0.5324、「环境工程」0.5473、
  「教育学」0.5794 —— cross-encoder 读出了「工程 ≠ 核工程」。但正例那一侧
  rerank_score 只有 **0.6987~0.7311** 这么宽，切不出「拦掉它们」又不碰正例的位置。
  扫描表里 0.70 看着很香（拦 6/8、正例零误杀），可它离正例下界只有 **0.0013** ——
  16 条样本在 0.03 宽的空隙里切一刀，那叫过拟合。所以**默认不按精排分过滤**
  （`config.py: rerank_min_score = None`）。而且代价不对称：漏一条越界用户看到一本沾边的书，
  误杀一条正例用户**搜不到本来有的书**。
- **压根不是「越界」，是主题相邻的真实书**：「纺织」→《这个词是怎么来的》
  （那本就是讲纺织词源的）、「法学概论」→《法律的概念》、《建筑学》→《建筑物理》。
  精排给它们 0.67~0.72 **是正确判断**。把它们叫 leak 是余弦时代留下的框架 ——
  当时只看得到「分数高但库里没有对口书名」，就一律当成模型出错。
  这类该修的是**产品定义**（用户搜「纺织」时，一本讲纺织词汇书源的书算不算命中？），不是检索链路。

### 4. BM25 的收益在「清尾部」，不在「提命中率」

用 `scripts/compare_bm25.py` 只切通道、同一份索引逐条对比（比的是帖子 id 序列）：

- 16 条正例查询里，**top-3 只有 1 条变了**，top-10 变了 5 条
- 变化最大的一条：「计算机网络」把 6 条**同一本书的重复帖子**（《大学计算机基础》）
  挤出 top-10，换成《计算机网络：自顶向下方法》

所以诚实的说法是：**混合检索没有提升 top-3 命中率，它提升的是 10 条列表的可用性**。
这件事纯向量做不到 —— 重复帖子的向量几乎相同、余弦分一样高，向量排序天然分不开。

---

## ❓ 常见问题

**Q：为什么不直接用 `LIKE` 做关键词检索？**

因为「同济版高数」和「高等数学 同济大学出版社」字面上没有公共子串。
更直接的一条：**库里没有任何一条书名的名称里含「同济」**（「同济」是版本体系，
不是书名的一部分），`book_name LIKE '%同济%'` 命中 **0** 条，
而语义检索能把《高等数学》准确捞出来。

**Q：检索为什么一定要有相似度阈值？**

向量检索**永远**会返回 top_k 条，哪怕库里根本没有这本书。
搜「考古」（库里 0 条）在默认阈值下返回空；把阈值放到 0，就会返回一堆
无关的书 —— 相似度只有 0.5 上下。所以它不是一个可以随手调的参数。

⚠️ 但它**不是万能药**。1 万条语料上实测：8 条越界查询的分数**高过正例的下界**
（详见「检索到顶了」），任何单一阈值都拦不住 —— 这是双塔结构的固有缺陷，
不是调参能解决的。

那上了 cross-encoder 是不是就根治了？**没有**，实测只多拦下 0 条
（还是那 1/8）。它把「本来答对但排第 2」的那批收服了（top-1 命中率 15/16 → 16/16），
过滤这条线上帮不上大忙。原因拆开看是两类，见「检索到顶了」第 3 节。

**Q：索引里为什么不存价格和成色？**

向量只负责回答「是哪本书」，条件过滤交给结构化字段。实测过两件事：
书名在检索文本里重复出现反而**拉低**分数（0.6258 → 0.6054 → 0.5953）；
用「未知」给空字段占位会毒化检索，全体掉 0.03~0.11。

**Q：换一个 LLM 要改多少？**

三行。`ai-service/.env` 里的 `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`。
走的是 OpenAI 兼容接口，DeepSeek、通义、本地 vLLM 都能接。

**Q：agent 为什么手写循环，不用 LangChain 的 `create_agent`？**

两版都在 `ai-service/app/agent/loop.py` 里（`chat_once` 和 `chat_once_framework`），
可以直接对着看。实测手写 **38 行**、框架 **43 行**（都不含 docstring）——
框架省掉的是 `bind_tools`、工单↔回执配对这些协议细节，不是行数。

**Q：能不用 MySQL 吗？**

不能。会话历史存在主服务的 `chat_message` 表里——AI 服务自己确实不连数据库，
但它要用历史就得走主服务的接口。这也是 LangChain 自带的 InMemory / SQLite / Redis
三种会话存储一个都没用上的原因：它们都要求 AI 服务自己存。

**Q：不是用上 Redis 了吗？那 AI 服务不就有状态了？**

区分**业务数据**和**运维态**。会话历史仍然在 MySQL、仍然由 Spring Boot 拥有，
Python 从来没碰过它。Redis 里放的是限流计数和一份**随时可以丢的**历史缓存 ——
丢了只是慢一点（回主服务读一遍就对），不影响任何正确性。

反过来，限流计数**必须**放进程外：放进程内，扩两个副本就是各限各的，等于没限。

**Q：token 用量是怎么统计的？**

挂在 LLM 客户端的 `BaseCallbackHandler` 上，不在调用处数。原因：抽取走的是
`with_structured_output(...).invoke()`，它直接返回解析好的 `BookInfo`，
**usage 在解析那一步就丢了**。callback 是唯一「不动调用处返回结构」就能拿到它的路。

`GET /usage` 会返回累计用量和按配置单价估算的成本。注意里面有一栏
`calls_without_usage` —— 它 > 0 就说明有调用成功返回但没拿到 usage，
**账目不全**。设这一栏是因为「统计失效」最容易伪装成一个漂亮的 0 元。

**Q：CI 里跑真模型吗？**

不跑。`conftest.py` 会在 import 之前塞一个假 key，所有外部调用在用例里都被
monkeypatch 掉 —— **CI 不花一分钱**。

代价说清楚：**CI 绿 ≠ 接口能调通**。key 失效、额度用完、模型下线，
CI 一个都发现不了。检索质量测试（`test_search_eval.py`）也不进 CI ——
它要 1 万条语料，而灌数据必须真调 LLM。详见
[.github/workflows/ci.yml](.github/workflows/ci.yml) 头部。

---

## 作者

[kurukurumi-33](https://github.com/kurukurumi-33)

---

## License

[MIT](LICENSE)。随便用，包括商用。里面没有别人的私有代码，
`ai-service/scripts/gen_mock_posts.py` 生成的 1 万条帖子是**用模型编的假数据**，
不含任何真实交易或个人信息。书目（`book_catalog.json`）是豆瓣公开列表页上的真实出版物
元数据，只是书名/作者/出版社这类事实，不含用户评论或任何个人信息。
