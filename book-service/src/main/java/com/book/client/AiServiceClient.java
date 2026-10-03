package com.book.client;

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
}
