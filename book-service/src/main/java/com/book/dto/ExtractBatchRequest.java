package com.book.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/** 调 AI 服务 POST /extract/batch 的请求体。 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "批量抽取请求（主服务 → AI 服务）")
public class ExtractBatchRequest {

    @Schema(description = "帖子原文列表，1~200 条")
    private List<String> texts;

    @JsonProperty("max_concurrency")
    @Schema(description = "最大并发数。由 application.yml 的 ai-service.max-concurrency 注入", example = "8")
    private Integer maxConcurrency;
}
