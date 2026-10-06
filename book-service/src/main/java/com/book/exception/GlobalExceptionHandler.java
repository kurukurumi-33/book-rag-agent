package com.book.exception;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ResponseStatusException;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 统一异常出口。
 *
 * 为什么要自己写：Spring Boot 默认的错误响应体只有 {timestamp, status, error, path}，
 * **不含 message** —— 调用方看到 502 完全不知道发生了什么。
 * 这里把原因写进响应体，同时也打到日志里。
 *
 * 注意：暴露内部错误信息在开发期很有用，生产环境要评估信息泄露风险
 * （数据库报错原文可能泄露表结构），通常只回一个 traceId，细节进日志。
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /** AI 服务连不上 → 503。调用方看到这个应该去把 Python 服务拉起来。 */
    @ExceptionHandler(AiServiceUnavailableException.class)
    public ResponseEntity<Map<String, Object>> handleAiUnavailable(AiServiceUnavailableException e) {
        log.error("AI 服务不可用：{}", e.getMessage());
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(body("AI 服务不可用", e.getMessage()));
    }

    /** 其他带状态码的异常（比如 AI 服务返回了 4xx/5xx 时抛的 502）。 */
    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<Map<String, Object>> handleResponseStatus(ResponseStatusException e) {
        log.error("下游调用失败：{}", e.getReason());
        return ResponseEntity.status(e.getStatusCode())
                .body(body("下游服务返回错误", e.getReason()));
    }

    /**
     * {@code @Valid} 校验没过 → 400。
     *
     * <p>不加这个 handler 的话，Spring 走默认的错误路径，返回体是
     * {@code {timestamp, status, error, path}} —— **没有 message**，
     * 只有一句英文 {@code "Bad Request"}。实测过：
     * <pre>
     *   {"message": ""}  →  400 {"timestamp":...,"error":"Bad Request","path":"/api/chat"}
     * </pre>
     * 调用方（前端 / curl / 别人）完全不知道是 message 还是 session_id 出的问题。
     *
     * <p>为什么默认体里连 {@code message} 都没有：Spring 的
     * {@code server.error.include-message: always} 只在 BasicErrorController
     * 渲染时生效，而且 {@code DefaultErrorAttributes} 对于这类异常给的是 null ——
     * 再被 {@code spring.jackson.default-property-inclusion: non_null} 一过滤，
     * **键直接消失**（这条配置对 Map 的 value 也生效）。
     *
     * <p>这里把 {@code BindingResult} 里每条字段错误的消息拼成一句话。
     *
     * <p>⚠️ **只取 {@code getDefaultMessage()}，不取 {@code getField()}**，
     * 也不做「字段名：消息」的拼接。两个原因，都是实测踩出来的：
     *
     * <ol>
     *   <li>{@code getField()} 给的是 <b>Java 属性名</b>，不是 JSON 里的名字。
     *       这个 DTO 的字段是 {@code sessionId}，对外的键是 {@code session_id}
     *       （靠 {@code @JsonProperty} 映射）。拼进去会变成
     *       <pre>sessionId：session_id 只能包含字母、数字……</pre>
     *       调用方拿着这句话去查自己发的 JSON，**找不到 sessionId 这个键**。</li>
     *   <li>消息文案本身已经带了字段名，再拼一次就是
     *       {@code message：message 不能为空}。</li>
     * </ol>
     *
     * <p>所以定成一条约定：<b>DTO 上的校验消息必须自带字段名，而且用对外的那套名字</b>
     * （{@code @NotBlank(message = "message 不能为空")}）。写注解的地方就是写 JSON 名的地方，
     * 两者挨着，改的时候不会漏。
     *
     * <p>排序是为了输出稳定 —— {@code getFieldErrors()} 的顺序不保证，
     * 同一个请求可能给出不同的字符串，写断言会间歇性失败。
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, Object>> handleValidation(MethodArgumentNotValidException e) {
        // 多个字段同时不合法时全部列出，用「；」分隔 —— 只报第一条会让人改一次试一次
        String detail = e.getBindingResult().getFieldErrors().stream()
                .sorted(Comparator.comparing(FieldError::getField))
                .map(FieldError::getDefaultMessage)
                .collect(Collectors.joining("；"));
        log.warn("参数校验失败：{}", detail);
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(body("请求参数不合法", detail));
    }

    /**
     * 请求体连 JSON 都不是（半截 JSON、空 body、顶层是数组而不是对象）→ 400。
     *
     * <p>不回显 Jackson 的原始解析错误：那句话里带着字节偏移和字符片段，
     * 对调用方没用，还可能把请求体内容漏进日志和响应。
     * 原始信息只进日志，响应给一句人话。
     */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<Map<String, Object>> handleUnreadable(HttpMessageNotReadableException e) {
        log.warn("请求体读不出来：{}", e.getMessage());
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(body("请求体格式错误", "请求体必须是一个 JSON 对象，例如 {\"message\": \"有高数吗\"}"));
    }

    private Map<String, Object> body(String error, String detail) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("error", error);
        m.put("detail", detail);
        return m;
    }
}
