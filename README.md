# 二手书智能匹配 Agent

基于 LangChain 的 RAG Agent 项目：把大学二手书交易里混乱的自然语言帖子
变成可语义检索的结构化数据，并用对话式 agent 帮学生找书。

## 架构

```
前端
  │
  ▼
Spring Boot 主服务 (book-service)   ← MySQL / Redis，对外 REST API
  │ HTTP
  ▼
Python AI 服务 (ai-service)          ← LangChain / FastAPI
  · LLM 信息抽取
  · Embedding 语义检索 (RAG)
  · Agent 对话 / tool 编排
  └─ agent 的 tool 回调主服务 API
```

**为什么要拆两个服务**：AI 服务不直连数据库，而是通过主服务的 API 取业务数据。
这样业务逻辑（权限、校验、事务）集中在主服务，AI 服务只负责「理解」。

## 目录

- `ai-service/` — Python + FastAPI + LangChain（用 VS Code 打开）
- `book-service/` — Spring Boot 主服务（用 IDEA 打开）
- `docs/` — [需求与接口](docs/需求与接口.md)（路线图 / 决策点 / 验收标准）、[面试素材](docs/面试素材.md)

## 接口文档

**不用手写，从代码注解自动生成**，代码改了文档跟着改：

| 服务 | Swagger UI（可在线调试） | OpenAPI spec |
|---|---|---|
| AI 服务 | http://localhost:8000/docs | http://localhost:8000/openapi.json |
| 主服务 | http://localhost:8080/swagger-ui.html | http://localhost:8080/v3/api-docs |

两个 spec 都可以直接导入 YApi / Apifox / Postman。

## 快速开始（AI 服务）

```bash
cd ai-service

# 1. 配置环境变量
cp .env.example .env    # 然后填入 LLM_API_KEY

# 2. 安装依赖（首次）
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 3. 冒烟测试：确认 LLM 能调通
.venv/Scripts/python.exe smoke_test.py

# 4. 启动服务
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

启动后访问 http://localhost:8000/health 应返回 `{"status":"ok",...}`，
API 文档在 http://localhost:8000/docs

## 快速开始（主服务）

```bash
cd book-service

# 1. 建库建表（首次）
mysql -uroot -p < src/main/resources/db/schema.sql

# 2. 配置数据库密码
#    复制 src/main/resources/application-local.yml.example 为 application-local.yml，
#    填入自己的密码（该文件已在 .gitignore 中排除）

# 3. 启动
mvn spring-boot:run
```

启动后接口在 http://localhost:8080

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/posts?limit=50` | 帖子列表 |
| GET | `/api/posts/{id}` | 帖子详情 |
| POST | `/api/posts` | 发布帖子（只传 userId + rawText） |
| GET | `/api/posts/count` | 总数 |
| POST | `/api/posts/extract-pending` | **批量抽取待处理帖子**（一批 50 条，反复调直到 `remaining` 为 0） |

## 技术栈

| 组件 | 版本 | 备注 |
|---|---|---|
| Java | 24 | JDK 23+ 需显式配置注解处理器，见 pom.xml |
| Spring Boot | 4.1.1 | |
| MyBatis-Plus | 3.5.17 | 需用 `spring-boot4-starter`；`IService` 已从 `extension` 移到 `spring` 包 |
| MySQL | 8.0 | |
| LangChain | 1.x | Python 侧 |

## 开发日志

- **D1** 环境搭建 + 冒烟测试
- **D2** MySQL 建库建表 + Spring Boot 骨架 + 模拟数据生成器
- **D3** 接口文档改成自动生成（FastAPI Swagger + springdoc）
- **D4** M1 抽取落库：`/extract/batch`（并发）+ `/api/posts/extract-pending`（编排），220 条全部抽取完成
