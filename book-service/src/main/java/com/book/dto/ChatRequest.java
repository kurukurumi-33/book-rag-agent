package com.book.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 对话请求体。**同一个类跑两趟**：
 *
 * <pre>
 *   浏览器 ──POST /api/chat──▶ 主服务 ──POST /chat──▶ AI 服务
 *           （反序列化成这个）        （原样序列化回去）
 * </pre>
 *
 * 进出形状完全一样，所以不拆成两个类 —— 拆了就有两份定义，改一处漏一处。
 * 字段名跟 AI 服务的 {@code ChatRequest}（ai-service/app/schemas.py）逐一对齐，
 * 那边的 snake_case 靠 {@code @JsonProperty} 映射，Java 这边照样写驼峰。
 *
 * <p>为什么要跟对一个 Python 类的字段名：主服务在这条链路上只是**转发**，
 * 它多一个字段、少一个字段，AI 服务那边都不会报错 —— 只会静默丢数据。
 * 这种错编译期发现不了，只能靠两边对着看。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ChatRequest {

    @NotBlank(message = "message 不能为空")
    private String message;

    /**
     * 会话 ID。**不传 = 开新会话**，AI 服务生成后放在响应里返回。
     *
     * <p>校验规则跟 AI 服务那边一字不差，因为最终它会变成 Python 侧的路径片段：
     * 带斜杠的会被路由拆开，带空格的编码后会被 Tomcat 直接拒掉。
     * 主服务先挡一道，坏参数就不用白跑一趟 HTTP 才被拒。
     *
     * <p>⚠️ 这里**故意不加 {@code @NotBlank}**：
     * {@code null} 是合法值（= 新会话），空串不是。
     * {@code @Pattern} 对 {@code null} 直接放行、对空串判不匹配，
     * 正好就是要的行为 —— 换成 {@code @NotBlank} 会把「开新会话」也拒掉。
     */
    @Pattern(
            regexp = "^[A-Za-z0-9_-]{1,64}$",
            message = "session_id 只能包含字母、数字、下划线、连字符，长度 1~64")
    @JsonProperty("session_id")
    private String sessionId;
}
