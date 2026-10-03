package com.book.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 批量抽取结果里的单条。
 *
 * bookInfo 和 error **恰好有一个非空**：
 * 成功时 bookInfo 有值、error 为 null；失败时反过来。
 */
@Data
@Schema(description = "批量抽取的单条结果：bookInfo 和 error 恰好有一个非空")
public class BatchItemResult {

    @JsonProperty("raw_text")
    @Schema(description = "原样回显的帖子原文，用于排查对齐问题")
    private String rawText;

    @JsonProperty("book_info")
    @Schema(description = "抽取结果；这一条失败时为 null")
    private BookInfo bookInfo;

    @Schema(description = "错误信息；这一条成功时为 null", example = "APITimeoutError: Request timed out.")
    private String error;
}
