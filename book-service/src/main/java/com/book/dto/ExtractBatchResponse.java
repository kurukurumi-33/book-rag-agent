package com.book.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * AI 服务 POST /extract/batch 的响应体。
 *
 * results 与请求的 texts **等长且顺序严格一致**（不是完成顺序），
 * 所以写回时必须按**下标**对齐，不能按 rawText 文本匹配（原文可能重复）。
 */
@Data
@Schema(description = "批量抽取响应：results 与请求的 texts 等长且顺序严格一致")
public class ExtractBatchResponse {

    @Schema(description = "与请求等长、顺序一致的结果列表")
    private List<BatchItemResult> results;
}
