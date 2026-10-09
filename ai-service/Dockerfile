# 智觅 AI 服务镜像
#
# ## 为什么 pin 3.12 而不是本机的 3.14
#
# 本机跑的是 3.14，但 3.14 太新，torch / chromadb / sentence-transformers
# 这一串依赖里**总有一个还没出对应 wheel** —— 装的时候会退化成源码编译，
# 然后在没有编译器的 slim 镜像里失败。3.12 是这套依赖生态稳定支持的版本。
# （这也是「本机能跑」和「能打包」是两回事的典型例子。）
#
# ## 构建
#
#     docker build -t zhimi-ai ./ai-service
#
# ⚠️ 镜像里**不含** BGE 和 rerank 的模型权重（1.2GB）。
#    它们在容器**首次调用时**从 HuggingFace 下载，或者在 docker-compose 里
#    把宿主机的 `ai-service/models/` 挂进来（推荐，见 docker-compose.yml）。
#    原因见 .gitignore：1.1GB 的权重不可能进 Git，也不该烤进镜像层。

FROM python:3.12-slim

# 让 Python 不缓冲 stdout/stderr —— 否则 `docker compose logs` 里看不到实时日志，
# 出问题时只能干等容器退出。容器里**永远**该加这个。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace 的缓存目录写成显式路径，方便挂 volume 复用，不用每次重下 1.2GB
    HF_HOME=/models \
    # 容器里没有 GPU，关掉 torch 的 CUDA 探测，省掉启动时的告警
    CUDA_VISIBLE_DEVICES=""

# curl 只用来做健康检查（HEALTHCHECK 那行）。
# --no-install-recommends 是必须的：不加的话 apt 会顺带装一堆推荐包，
# 镜像能大出几十 MB。
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 依赖先装，代码后拷 ----
#
# 这个顺序是**故意的**，是 Docker 层缓存的用法：
# 只要 requirements.txt 没变，改了业务代码重新构建时，
# 「装依赖」这层直接命中缓存 —— 省掉重装 torch 的几分钟。
# 反过来（先 COPY 全部代码再装依赖）的话，改一行 Python 就要重装一遍 torch。
COPY requirements.txt .

# ⚠️ torch 必须显式从 CPU 源装：
#    默认源给的是 CUDA 版（2GB+），容器里没 GPU，纯属让镜像胖 2GB。
#    先装 CPU 版之后，requirements.txt 里那行 torch 就会判定「已满足」。
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

# 向量库落盘目录。声明成 volume 是为了让数据留在宿主机 ——
# 不然 `docker compose down` 一次，1 万条向量就没了，得重建 23 秒。
VOLUME ["/app/data", "/models"]

# 非 root 跑服务。容器内的进程不该有 root 权限 ——
# 万一服务被攻破，root 能做的事多得多。
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /models \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# 健康检查打 /health —— 它会真的去探 Redis 和主服务，
# 所以这一条同时能发现「本服务活着但下游全断了」。
#
# --start-period 给 20 秒：uvicorn 起来要几秒，且在预热模型。
# 没有它，容器会在启动阶段就被判成 unhealthy。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 不用 --reload：那是开发用的（文件一变就重启），生产镜像里开着会：
#   1. 额外起一个监视进程
#   2. 每次重启都重新加载模型
# 也不用 --proxy-headers：本项目是直连，没有反代；真上了反代再开，
# 并配上 --forwarded-allow-ips 白名单（见 main.py 的 _client_id）。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
