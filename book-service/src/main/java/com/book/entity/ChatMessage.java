package com.book.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * Agent 会话历史的一条消息。
 *
 * 这张表**不是业务数据**，是 AI 服务的记忆。它放在主服务的库里，
 * 是因为「AI 服务不直连数据库」这条铁律 —— Python 那边通过 REST 读写，
 * 自己不持有连接、重启不丢东西。
 *
 * ## 为什么 role 用 human / ai，而不是 user / assistant
 *
 * 因为这份历史的**唯一消费者是 LangChain**。LangChain 的消息类型就是
 * HumanMessage(.type == "human") 和 AIMessage(.type == "ai")，
 * 用同一套叫法，Python 侧读回来直接查表构造对象，不用再做一次映射。
 *
 * （代价是这张表跟框架耦合了。可接受：换框架时改的是 Python 侧那个转换函数，
 *  不用动表结构 —— 因为表里存的是**纯文本**，不含任何 LangChain 的序列化格式。）
 */
@Data
@TableName("chat_message")
@Schema(description = "Agent 会话历史的一条消息")
public class ChatMessage {

    @TableId(type = IdType.AUTO)
    @Schema(description = "主键。自增，同时就是会话内的顺序 —— 不需要额外的 seq 字段")
    private Long id;

    @Schema(description = "会话 ID。前端不传时由 AI 服务生成 uuid4", example = "6f1c0e6a-...")
    private String sessionId;

    @Schema(
            description = "角色。human=用户说的，ai=模型的最终回答（工具往返不落库）",
            allowableValues = {"human", "ai"},
            example = "human"
    )
    private String role;

    @Schema(description = "消息正文", example = "有高数吗")
    private String content;

    @Schema(description = "写入时间")
    private LocalDateTime createdAt;
}
