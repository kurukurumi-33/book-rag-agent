package com.book.exception;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ResponseStatusException;

import java.util.LinkedHashMap;
import java.util.Map;

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

    private Map<String, Object> body(String error, String detail) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("error", error);
        m.put("detail", detail);
        return m;
    }
}
