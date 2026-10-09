"""集中管理配置，全部从 .env 读取。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM（DeepSeek 提供 OpenAI 兼容接口）
    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # 超时与重试（M3.2 补的）。
    #
    # 这是这个项目里**唯一一个之前没有超时保护的出网调用** ——
    # /chat 最坏情况会连着打好几次 LLM，只要有一次对端卡住，
    # 请求就会一直挂着占住 worker，直到客户端自己超时。
    # 60 秒是「一次正常抽取 1~2 秒」的 30 倍余量，够宽松但不会挂死。
    llm_timeout: float = 60.0
    # 网络抖动重试 2 次。别再往上调：DeepSeek 的 429 是**账号级**的，
    # 重试太多次只会把限流打得更死（抽取那条路的 `max-concurrency: 8` 就是为这个留的余量）。
    llm_max_retries: int = 2

    # Spring Boot 主服务地址（agent 的 tool 会回调它）
    #
    # 用 127.0.0.1 而不是 localhost：localhost 同时解析到 IPv4 和 IPv6，
    # 客户端可能先试 IPv6 再失败，表现成「服务明明起着却连不上」。
    # Java 侧刚踩过这个坑（面试素材第 9 条那次 503 就是它）。
    book_service_url: str = "http://127.0.0.1:8080"

    # ---- M2 语义检索 ----

    # 本地 BGE 中文向量模型。512 维、约 100MB，CPU 上够快。
    # 选本地而不选 API：DeepSeek 根本没有 embedding 接口，
    # 换别家又要再申请一个 key；本地模型离线可跑，面试现场断网也能演示。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # 向量库落盘目录（相对 ai-service/）。
    # Chroma 是**嵌入式**库：跟 SQLite 一样是进程内的，没有独立服务要启动。
    chroma_dir: str = "./data/chroma"

    # collection 名。改个名 = 换一份独立索引，互不干扰。
    # Chroma 对名字有硬校验：3~512 字符、只能含 [a-zA-Z0-9._-]、首尾必须是字母或数字
    # —— 取成 "a" 这种短名会直接 Validation error（实测）。
    chroma_collection: str = "book_posts"

    # 启动时是否在后台线程里预热 embedding 模型。
    #
    # 开着：第一次 /search 是快的（模型已经加载好）。
    # 关掉：服务启动快 15 秒 —— **改 M2 代码时建议临时关掉**，
    #       uvicorn --reload 每次热重载都会重新跑一遍预热，很拖节奏。
    warmup_on_startup: bool = True

    # ---- M2.5 rerank 精排 ----

    # 精排开关。关掉 = 回到「向量+BM25 融合」的纯召回行为。
    # **留这个开关不是为了「灵活」，是为了能做 A/B** ——
    # eval_search.py 就是靠对比开/关来判断 rerank 到底有没有净增益的。
    rerank_enabled: bool = True

    # 精排模型。本地 `models/bge-reranker-base/` 存在时优先用它（原因见 rerank.py）。
    rerank_model: str = "BAAI/bge-reranker-base"

    # 精排**候选池**大小：粗排捞出多少条送去逐对打分。
    # 这是「效果 vs 延迟」的旋钮，不是越大越好：
    # cross-encoder 是 CPU 上的逐对前向，20 条约 1~2 秒，翻倍就直接拖垮演示。
    # 而且它只管重排，池子外的书再怎么精排也进不来（见 rerank.py 的边界说明）。
    rerank_pool: int = 20

    # 精排分的下限（sigmoid 之后的 0~1）。None = 不按精排分过滤，**只重排**。
    #
    # ⚠️ 默认 None 是**慎重决定**，不是没标定。`scripts/tune_rerank.py` 已经把表扫出来了：
    #
    #     阈值    越界被拦   正例存活
    #     0.55     1/8      16/16
    #     0.65     3/8      16/16
    #     0.70     6/8      16/16    ← 看着很香
    #     0.75     8/8       0/16    ← 一过 0.7311 正例全死
    #
    # 0.70 拦得最多、也不误杀，为什么不用？因为**正例那一侧只有 0.6987~0.7311 这么宽**
    # （16 条里最低的是「同济版高数」0.6987）。0.70 距离正例下界只有 **0.0013** ——
    # 样本再多几条，这个阈值立刻开始误杀。用 16 条样本在 0.03 宽的空隙里切一刀，
    # 那叫过拟合，不叫标定。
    #
    # 而且代价是不对称的：漏掉一条越界，用户看到一本沾边的书；
    # 误杀一条正例，用户**搜不到本来有的书**。后者更贵。
    #
    # 所以现在的定位很明确：**rerank 负责排序，不负责过滤**。
    # 过滤还是交给余弦阈值（它的召回率在 1 万条上已经够用）。
    # 真要开这个旋钮，先用 tune_rerank.py 在**更多**正例上重新扫一遍。
    rerank_min_score: float | None = None

    # ---- M2.6 多模态以图搜书 ----
    #
    # 走「图 → 书名 → 复用 search()」，不是「图 → 向量」，理由见 services/vision.py。
    #
    # 默认用智谱 GLM-4V-Flash：**有免费额度**，演示够用，注册就能拿 key。
    # 留空 = 关闭这个功能（接口返回 503 + 一句明确的「未配置」），
    # **不是**留空就崩 —— 没有 key 的人 clone 下来照样能把别的功能跑通。
    vision_api_key: str = ""
    vision_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    vision_model: str = "glm-4v-flash"
    # 视觉调用比文本慢（要传图、要走多模态前向）。30 秒是「用户愿意等」的上限，
    # 超过就该让用户重拍，而不是继续挂着。
    vision_timeout: float = 30.0

    # ---- M4.5 Redis：限流 + 会话历史缓存 ----
    #
    # 存的是**运维态**（限流计数、缓存），不是业务数据 —— 所以不违反
    # 「AI 服务不直连业务数据库」那条铁律。理由见 services/redis_client.py。
    #
    # ⚠️ URL 里带的密码**只写在 .env**（gitignored），不写在这里。
    #    本机 Redis 的密码是 123456，所以本机的 .env 里是：
    #        REDIS_URL=redis://:123456@127.0.0.1:6379/0
    #    默认值留不带密码的版本，别人 clone 下来不开 Redis 也能跑。
    redis_url: str = "redis://127.0.0.1:6379/0"

    # Redis 总开关。关掉 = 完全不连（限流放行、缓存直读），用于本地调试。
    redis_enabled: bool = True

    # 连接/读写超时（秒）。**必须设短** —— 默认的 None 意味着 Redis 挂掉时
    # 请求会一直挂着，比限流失效严重得多。0.5 秒足够本地通信。
    redis_timeout: float = 0.5

    # 限流：每个 IP 在 60 秒窗口内允许的次数。
    #
    # /chat 一次烧一次模型调用，卡得紧一点；/extract 系列是内部调用（建索引时会
    # 连着打几百次），给宽一些，但仍然要有天花板 —— 见 ratelimit.limit_for()。
    ratelimit_chat_per_min: int = 20
    ratelimit_extract_per_min: int = 60

    # 以图搜书单列一档，卡得最紧。
    #
    # ⚠️ 为什么不是复用 extract 的 60：视觉模型**比文本模型慢一个数量级**
    #    （一张图几秒），而且一次请求后面还跟着最多 10 次完整检索。
    #    60 次/分钟 = 每分钟 60 张图 + 600 次检索，本机 CPU 扛不住。
    #    10 次/分钟对「用户拍自己那摞书」来说绰绰有余。
    ratelimit_image_per_min: int = 10

    # 会话历史的缓存时长（秒）。
    # 它**不是**一致性保障 —— 一致性靠「写完就删」（见 session_cache.py）。
    # 这个 TTL 只管「没人访问的会话别一直占着内存」。
    session_cache_ttl_s: int = 300

    # ---- M3.2 token 成本 ----

    # 单价（元 / 百万 token），**只用来估算**，以 DeepSeek 官网当前价为准。
    #
    # 为什么要写成配置而不是硬编码在 usage.py：价格会变，而且你可以
    # 改成别家（Qwen / 智谱）的价再跑，估算逻辑一行不用动。
    # 这里的默认值对应 deepseek-chat 的「输入（缓存未命中）/ 输出」两档。
    llm_price_in_per_mtok: float = 2.0
    llm_price_out_per_mtok: float = 8.0


settings = Settings()
