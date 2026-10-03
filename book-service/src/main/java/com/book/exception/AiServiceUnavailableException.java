package com.book.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

/**
 * AI 服务调不通：进程没起、端口不对、网络不通、读超时。
 *
 * 这跟「AI 服务返回了 500（模型调用失败）」不是一回事：
 * 前者是**链路断了**，后者是**链路通了但对端处理失败**。
 * 靠 @ResponseStatus 让 Spring 直接翻译成 503，controller 里不用写 try-catch。
 *
 * 为什么用 503 而不是 500：503 Service Unavailable 的语义是
 * 「服务暂时不可用，稍后重试可能成功」——正好是 AI 服务没启动的情形，
 * 调用方（脚本 / 前端）看到 503 就知道该去把 Python 服务拉起来。
 */
@ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
public class AiServiceUnavailableException extends RuntimeException {

    public AiServiceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
