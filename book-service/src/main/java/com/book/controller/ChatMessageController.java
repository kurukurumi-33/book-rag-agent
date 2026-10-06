package com.book.controller;

import com.book.entity.ChatMessage;
import com.book.service.ChatMessageService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 会话历史接口。
 *
 * **这一组接口没有前端调用方 —— 唯一的调用者是 Python 的 AI 服务。**
 * Agent 每轮开始前来这里读历史、结束后往这里写两条。
 *
 * 为什么不让 Python 直连 MySQL：铁律一（AI 服务不碰业务数据源）。
 * Python 侧重启不丢任何东西、可以随便扩多副本，代价就是每次读历史多一次 HTTP。
 */
@RestController
@RequestMapping("/api/chat/{sessionId}/messages")
@RequiredArgsConstructor
@Tag(name = "会话历史", description = "Agent 的记忆。读写方都是 AI 服务，不是前端")
public class ChatMessageController {

    private final ChatMessageService chatMessageService;

    @GetMapping
    @Operation(
            summary = "取会话最近 N 条消息",
            description = """
                    按时间**正序**返回（最老的在前），可以直接拼进模型的 messages 列表。

                    ## 为什么是「最近 N 条」不是「全部」

                    会话可以无限长。全量读回来会把模型的上下文窗口撑爆，
                    而且 token 是按量计费的 —— 第 100 轮时把前 99 轮全发一遍，
                    既慢又贵，且大部分内容跟当前问题无关。

                    截断的代价是很久以前说过的话会忘。要缓解就加大 `limit`，
                    或者在这里做摘要压缩（把 N 条之前的内容压成一段 SystemMessage）。
                    """
    )
    public List<ChatMessage> recent(
            @Parameter(description = "会话 ID") @PathVariable String sessionId,
            @Parameter(description = "取最近多少条，默认 20，上限 200")
            @RequestParam(defaultValue = "20") int limit) {
        return chatMessageService.recent(sessionId, limit);
    }

    @PostMapping
    @Operation(
            summary = "追加一轮消息（一次两条）",
            description = """
                    一次提交**一轮对话**产生的全部消息 —— 通常是
                    `[{role: human, ...}, {role: ai, ...}]` 两条。

                    ## 为什么不拆成两次调用

                    这两条必须一起落库。拆开发的话，两个请求之间进程一挂，
                    库里就只剩用户的问、没有模型的答。下一轮读回这段历史，
                    模型看到一句孤零零的问话，接不上话。

                    服务端这边用同一个事务写，所以要么两条都在，要么两条都不在。

                    ## 谁决定 id 和 sessionId

                    都不是客户端。`sessionId` 从 URL 取，`id` 由数据库自增 ——
                    客户端传了也会被忽略，跟 `POST /api/posts` 强制清空抽取字段同一个思路。
                    """
    )
    public List<ChatMessage> append(
            @Parameter(description = "会话 ID") @PathVariable String sessionId,
            @RequestBody List<ChatMessage> messages) {
        return chatMessageService.append(sessionId, messages);
    }
}
