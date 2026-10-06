package com.book.client;

import com.book.dto.ChatRequest;
import com.book.dto.ExtractBatchRequest;
import com.book.dto.ExtractBatchResponse;
import com.book.dto.ExtractRequest;
import com.book.dto.ExtractResponse;
import com.book.exception.AiServiceUnavailableException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import tools.jackson.databind.JsonNode;

import java.time.Duration;
import java.util.List;

/**
 * 调用 Python AI 服务的客户端。
 *
 * 主服务与 AI 服务的边界就在这里：主服务负责业务编排（查数据、写库），
 * AI 服务只提供无状态能力（文本进、结构出）。
 *
 * 只有这个类知道 AI 服务的存在 —— 换个模型服务商，改这一个文件就行。
 */
@Component
public class AiServiceClient {

    private static final Logger log = LoggerFactory.getLogger(AiServiceClient.class);

    private final RestClient restClient;
    private final String baseUrl;
    private final int maxConcurrency;

    /**
     * 用 Spring Boot 自动配置的 RestClient.Builder，但**显式指定 HTTP/1.1 的请求工厂**。
     *
     * ⚠️ 为什么不能用 Boot 4 的默认客户端：
     * Boot 4 默认用 JDK 的 java.net.http.HttpClient，它会先发一个
     * `Connection: Upgrade, HTTP2-Settings` + `Upgrade: h2c` 的请求，
     * 试图把连接升级到 HTTP/2。而 uvicorn 是纯 HTTP/1.1 服务，不认这套握手 ——
     * 结果是**请求发出了，body 却丢了**，FastAPI 收到空 body 直接 422：
     * `{"type":"missing","loc":["body"],"msg":"Field required"}`
     *
     * 这个 bug 的可怕之处：报错信息完全没提 HTTP/2，只说"字段缺失"，
     * 会让人一直往 DTO 序列化上查。真正的定位手段是**把原始请求抓下来看**。
     *
     * SimpleClientHttpRequestFactory 底层是 HttpURLConnection，纯 HTTP/1.1，没有这个协商过程。
     */
    public AiServiceClient(
            RestClient.Builder builder,
            @Value("${ai-service.base-url}") String baseUrl,
            @Value("${ai-service.max-concurrency}") int maxConcurrency) {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        // 一批 50 条要跑几十秒，不设读超时的话 AI 服务一卡死，请求线程就永远挂在那儿
        factory.setConnectTimeout(Duration.ofSeconds(10));
        factory.setReadTimeout(Duration.ofMinutes(2));

        this.restClient = builder
                .baseUrl(baseUrl)
                .requestFactory(factory)
                .build();
        this.baseUrl = baseUrl;
        this.maxConcurrency = maxConcurrency;
    }

    /**
     * 调 AI 服务抽取单条书籍信息。
     *
     * @param rawText 帖子原文
     * @return 抽取结果
     */
    public ExtractResponse extract(String rawText) {
        return restClient.post()
                .uri("/extract")
                .body(new ExtractRequest(rawText))
                .retrieve()
                .body(ExtractResponse.class);
    }

    /**
     * 批量抽取。一次 HTTP 请求发 N 条文本过去，AI 服务内部并发处理。
     *
     * @param texts 帖子原文列表，顺序即返回结果的顺序
     * @return 与 texts 等长、顺序一致的响应
     */
    public ExtractBatchResponse extractBatch(List<String> texts) {
        log.info("→ POST {}/extract/batch  共 {} 条文本, max_concurrency={}",
                baseUrl, texts.size(), maxConcurrency);
        try {
            ExtractBatchResponse response = restClient.post()
                    .uri("/extract/batch")
                    .body(new ExtractBatchRequest(texts, maxConcurrency))
                    .retrieve()
                    .body(ExtractBatchResponse.class);
            log.info("← /extract/batch 返回 {} 条结果",
                    response == null || response.getResults() == null ? 0 : response.getResults().size());
            return response;
        } catch (ResourceAccessException e) {
            // 连接被拒 / 读超时。这是**链路问题**：AI 服务根本没收到请求。
            // 常见原因：Python 服务没起、端口不对、localhost 解析成了 IPv6 而 uvicorn 只绑了 IPv4。
            throw new AiServiceUnavailableException(
                    "连不上 AI 服务 " + baseUrl + "（" + e.getMessage() + "）。"
                            + "确认 Python 服务已启动，且 base-url 用的是 127.0.0.1 而不是 localhost。",
                    e);
        } catch (RestClientResponseException e) {
            // 连上了，但对端返回了非 2xx。这是**对端处理失败**，不是链路问题 ——
            // 用 502 而不是 503：503 会让调用方以为该重试，但 4xx 重试多少次都是同样的错。
            // 把对端的响应体原样打出来 —— 校验类错误（FastAPI 的 422）的细节全在这里面。
            log.error("← /extract/batch 返回 {}，响应体：{}",
                    e.getStatusCode(), e.getResponseBodyAsString());
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "AI 服务返回 " + e.getStatusCode() + "：" + e.getResponseBodyAsString(), e);
        }
    }

    /**
     * 调 AI 服务的 agent 对话接口。
     *
     * <p>这是整条链路上最慢的一次调用：AI 服务内部要跑完 agent 循环
     * （模型 → 工具 → 模型 → ……），每一轮都是一次真实的 LLM 请求，
     * 而工具回调回来又要打一次主服务。2 分钟的读超时（见构造函数）就是给它留的。
     *
     * <h2>为什么返回 JsonNode，而不是一个 Java 响应 DTO</h2>
     *
     * 响应里有三处**没法用 Java 类型老实表达**：
     * <pre>
     *   tool_calls[].arguments   free-form dict —— 参数名由模型和各工具的 schema 决定
     *   books[].price            float | "未标价" —— 联合类型（见 schemas.py BookCard）
     *   books[].edition          可能是 null
     * </pre>
     *
     * 硬要映射就只能全写 {@code Object}，等于没类型；而写一套完整的 DTO 意味着
     * 同一个 schema 在 Python 和 Java 各维护一份 —— AI 服务加个字段，这边不报错、
     * 只是静默丢掉，正是最难查的那类 bug。
     *
     * <p>主服务在这条链路上的角色是**转发**，不是**理解**：它没有理由去认识
     * {@code books} 里有哪些字段。所以原样透传，让前端直接吃 AI 服务的输出格式 ——
     * 接口文档的唯一出处是 AI 服务的 Swagger（localhost:8000/docs）。
     *
     * <h2>⚠️ 这里的 JsonNode 必须是 tools.jackson 的，不是 com.fasterxml.jackson 的</h2>
     *
     * Spring Boot 4 / Spring Framework 7 换成了 <b>Jackson 3</b>，
     * databind 的包名从 {@code com.fasterxml.jackson.databind} 搬到了
     * {@code tools.jackson.databind}（annotations 那个包名没动）。
     *
     * <p>而 classpath 上**同时存在 Jackson 2**（springdoc 带进来的），所以
     * 写 import 时 IDEA 补全会两个都给、编译也能过 —— 只有在真正反序列化时才会炸：
     * <pre>
     *   InvalidDefinitionException: Cannot construct instance of
     *   `com.fasterxml.jackson.databind.JsonNode` (no Creators ...)
     * </pre>
     * 因为 Spring 的 message converter 用的是 Jackson 3 的 ObjectMapper，
     * 它不认识 Jackson 2 的那个 JsonNode 接口。**编译期查不出来，是运行期才炸的**。
     *
     * <h2>JsonNode 不吃 non_null 这条配置（实测过）</h2>
     *
     * application.yml 里配了 {@code spring.jackson.default-property-inclusion: non_null}，
     * 所以 {@code GET /api/posts/{id}} 直接返回实体时，值为 null 的字段会被**整个省掉**
     * （实测 209 号帖子的响应里干脆没有 edition 这个键）。
     *
     * <p>但这条配置**不作用于 JsonNode** —— 树上的 NullNode 是节点本身，不是「值为 null 的属性」。
     * 实测 {@code /chat} 的响应里 {@code "edition":null} 和 {@code "condition_desc":null}
     * 都原样保留了，跟 Python 那边一致，前端不用为两种风格分别写判断。
     *
     * @param request 已经过主服务校验的请求，字段名与 AI 服务侧对齐
     * @return AI 服务的响应原文
     */
    public JsonNode chat(ChatRequest request) {
        log.info("→ POST {}/chat  session_id={}  message={}",
                baseUrl, request.getSessionId(), brief(request.getMessage()));
        try {
            JsonNode response = restClient.post()
                    .uri("/chat")
                    .body(request)
                    .retrieve()
                    .body(JsonNode.class);
            log.info("← /chat 返回 ok，session_id={}，reply {} 字",
                    response == null ? null : response.path("session_id").asText(null),
                    response == null ? 0 : response.path("reply").asText("").length());
            return response;
        } catch (ResourceAccessException e) {
            // 链路断了：AI 服务没起、端口不对、或 agent 跑太久超过了读超时。
            // 抛 AiServiceUnavailableException → 全局异常处理翻译成 503，
            // 前端据此提示「AI 服务没启动」而不是白屏。
            throw new AiServiceUnavailableException(
                    "连不上 AI 服务 " + baseUrl + "（" + e.getMessage() + "）。"
                            + "确认 Python 服务已启动，且 base-url 用的是 127.0.0.1 而不是 localhost。",
                    e);
        } catch (RestClientResponseException e) {
            // 连上了但对端非 2xx：模型调用失败(500)、参数被 FastAPI 拒(422)、
            // 或者 AI 服务自己读写历史失败(503)。用 502 —— 错不在主服务的调用方。
            log.error("← /chat 返回 {}，响应体：{}",
                    e.getStatusCode(), e.getResponseBodyAsString());
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "AI 服务返回 " + e.getStatusCode() + "：" + e.getResponseBodyAsString(), e);
        }
    }

    /** 日志里不打印整句用户输入 —— 长了会把一行日志撑爆。 */
    private static String brief(String text) {
        if (text == null) {
            return "null";
        }
        String oneLine = text.replaceAll("\\s+", " ").trim();
        return oneLine.length() <= 40 ? oneLine : oneLine.substring(0, 40) + "…";
    }
}
