# 智觅主服务镜像（Java / Spring Boot）
#
# ## 多阶段构建：builder 里编译，runtime 里只跑 jar
#
# 单阶段的写法是「装个 JDK，把源码拷进去，`mvn spring-boot:run`」——
# 那样镜像里会同时有：JDK 编译器、Maven、整个 ~/.m2 仓库、源码。
# 光 ~/.m2 就是几百 MB，而运行一个 Spring Boot jar **只需要一个 JRE**。
#
# 分两阶段之后，最终镜像里没有源码、没有 Maven、没有编译器 ——
# 只剩一个 JRE 和一个 jar。体积通常能从 ~800MB 降到 ~300MB。
#
# ## 构建
#
#     docker build -t zhimi-book ./book-service
#
# ## 为什么用 temurin-24
#
# pom.xml 里 `<java.version>24</java.version>`，构建阶段和运行阶段的
# **Java 大版本必须一致** —— 用 21 编、24 跑（或者反过来）会遇到
# `UnsupportedClassVersionError`，而那个报错只说「版本不对」，
# 不告诉你是哪一步用的哪个版本，第一次见很难反应过来是两处没对齐。

# ============================ 阶段 1：编译 ============================
FROM maven:3.9-eclipse-temurin-24 AS builder

WORKDIR /build

# ---- 先只拷 pom.xml，把依赖层单独缓存起来 ----
#
# 和 Python 那边同一个套路，但这里收益更大：
# Spring Boot 的依赖树要下几百个 jar，第一次构建可能好几分钟。
# 只要 pom.xml 没改，重新构建就命中这一层，直接跳过下载。
COPY pom.xml .
RUN mvn -B dependency:go-offline

# 依赖下完了再拷源码 —— 改 Java 代码不会让上面那层失效
COPY src ./src

# -DskipTests：CI 里编译和测试是**分开的两件事**（见 .github/workflows/ci.yml）。
# 镜像构建只负责「打得出来」，测试失败了该由 CI 的红灯说话，
# 而不是让构建卡在这里、日志混在一起。
RUN mvn -B -DskipTests package

# ============================ 阶段 2：运行 ============================
FROM eclipse-temurin:24-jre

WORKDIR /app

# curl 给健康检查用
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 只从 builder 里拿产物 —— 源码、Maven、.m2 全都留在上一阶段了
COPY --from=builder /build/target/*.jar app.jar

# 非 root 用户跑服务
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Spring Boot Actuator 没引，所以打不了 /actuator/health。
# 退而求其次打 /api/posts —— 它会走一遍「解析请求 → 查 MySQL → 序列化」，
# 比单纯看端口通不通更能说明问题（MySQL 挂了这里就会失败）。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8080/api/posts?limit=1" || exit 1

# ⚠️ 这里**没有** -Dspring.profiles.active 之类的东西。
#    数据库地址、密码全部由 docker-compose.yml 通过环境变量注入
#    （SPRING_DATASOURCE_URL / DB_PASSWORD）—— 见 application.yml 里
#    那行 `password: ${DB_PASSWORD:CHANGE_ME}` 和 `spring.datasource.url`。
#
#    也就是说：**同一个镜像能连不同的库**，不需要为每个环境重新构建。
#    换个 MySQL 只改 compose 里的环境变量，镜像一个字不动。
#
#    UseContainerSupport（容器内存感知）现在默认开着，不用手动加
#    -XX:+UseContainerSupport，也**不要**去写 -Xmx ——
#    写死了反而会在小机器上被 OOM kill。
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
