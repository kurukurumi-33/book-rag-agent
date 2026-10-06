package com.book.controller;

import com.book.client.AiServiceClient;
import com.book.dto.ChatRequest;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import tools.jackson.databind.JsonNode;

/**
 * 对话接口 —— **前端唯一要调的接口**。
 *
 * <pre>
 *   浏览器 ──▶ 主服务 :8080 ──▶ AI 服务 :8000
 * </pre>
 *
 * <h2>为什么前端不直接调 AI 服务</h2>
 *
 * 前端只认一个地址（{@code localhost:8080}），Python 服务在哪个端口、部署在哪台机器
 * 都跟它无关。代价是多一跳 HTTP，换来的是：
 * <ul>
 *   <li>Python 换成别的实现（或换供应商），前端一行不用改</li>
 *   <li>跨域问题天然不存在 —— 页面和接口同源，不用配 CORS</li>
 *   <li>主服务能在这里补做校验、限流、鉴权，AI 服务侧保持「无脑算」</li>
 * </ul>
 *
 * <h2>路由为什么不是 /api/chat/{sessionId}</h2>
 *
 * 会话 ID 走**请求体**（{@code session_id}），不走路径。因为它常常还不存在 ——
 * 用户发的第一句就是用来开新会话的，那时候没有 ID 可以往路径里填。
 * 路径里带一个「可能没有」的段，就得再发明一个「新会话」的哨兵值。
 *
 * <p>隔壁 {@link ChatMessageController} 的 {@code /api/chat/{sessionId}/messages}
 * 不一样：那是 AI 服务在读写信史，调它的时候会话一定已经存在了。
 */
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@Tag(name = "对话", description = "前端跟 agent 说话的唯一入口。主服务只做转发，回答由 AI 服务产出")
public class ChatController {

    private final AiServiceClient aiServiceClient;

    @PostMapping
    @Operation(
            summary = "和 agent 对话",
            description = """
                    用户说一句，AI 服务那边的 agent 自己决定要不要查库、查几次，然后回答。
                    主服务这一层**只做转发**：不解析 body，不碰数据库。

                    ## 请求

                    | 字段 | 必填 | 说明 |
                    |---|---|---|
                    | `message` | 是 | 用户这一句原话 |
                    | `session_id` | 否 | **不传 = 开新会话**，服务端生成后放在响应里返回；前端存住它，下一句带上 |

                    ## 响应

                    结构由 AI 服务定义，**这里不重复描述** —— 唯一出处是它自己的接口文档
                    `http://localhost:8000/docs`。主服务原样透传，因为响应里有几处
                    用 Java 类型表达不了的东西（`arguments` 是自由 dict、
                    `books[].price` 是 `float | "未标价"` 的联合类型）。

                    给前端用的话，记住四个字段就够：

                    | 字段 | 用途 |
                    |---|---|
                    | `session_id` | 存下来，下一句原样带回 |
                    | `reply` | 回答正文 |
                    | `tool_calls` | 调了哪些工具、参数是什么 —— **演示时最该展示的一栏** |
                    | `books` | 要渲染成卡片的书（书名/版次/价格/成色/原帖） |

                    ## 关于 `tool_calls` 为什么值得单独展示

                    只有 `reply` 的话，看的人分不清「agent 自己决定查了库」和
                    「硬编码了一段回答」。把轨迹摊开，才能证明它真的在决策。

                    ## 慢

                    一次请求要跑完整的 agent 循环（模型 → 工具 → 模型 ……），
                    每轮都是一次真实 LLM 调用，工具还会回调主服务取数据。
                    **前端必须做好等待提示**，不能发完请求就不管了。
                    """,
            responses = {
                    @ApiResponse(responseCode = "200", description = "agent 的回答 + 工具轨迹 + 书卡"),
                    @ApiResponse(responseCode = "400",
                            description = "message 为空/缺失，或 session_id 形状不合法，或请求体不是合法 JSON。"
                                    + "响应体是 {error, detail}，detail 会点出具体是哪个字段"),
                    @ApiResponse(responseCode = "502", description = "AI 服务返回了非 2xx（模型调用失败等）"),
                    @ApiResponse(responseCode = "503", description = "连不上 AI 服务 —— Python 进程没启动"),
            }
    )
    public JsonNode chat(@Valid @RequestBody ChatRequest request) {
        return aiServiceClient.chat(request);
    }
}
