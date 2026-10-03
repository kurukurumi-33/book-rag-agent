package com.book.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * 一轮 POST /api/posts/extract-pending 的执行结果。
 *
 * 恒等式：picked == success + failed。
 */
@Data
@Schema(description = "一轮批量抽取的执行结果")
public class ExtractPendingResponse {

    @Schema(description = "本轮捞出的待抽取帖子数", example = "50")
    private int picked;

    @Schema(description = "成功抽取并写回的数量", example = "48")
    private int success;

    @Schema(description = "失败并被标记为 FAILED 的数量", example = "2")
    private int failed;

    @Schema(description = "跑完后仍处于 PENDING 的数量。> 0 说明还得再调一轮", example = "170")
    private long remaining;

    @Schema(description = "失败原因摘要，格式 `帖子 12：APITimeoutError: ...`")
    private List<String> errors;
}
