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


settings = Settings()
