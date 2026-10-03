package com.book.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/** 调 AI 服务做抽取的请求体。 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ExtractRequest {

    @JsonProperty("raw_text")
    private String rawText;
}
