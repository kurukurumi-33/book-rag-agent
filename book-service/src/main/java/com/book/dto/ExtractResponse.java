package com.book.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/** AI 服务抽取接口的响应体。 */
@Data
public class ExtractResponse {

    @JsonProperty("raw_text")
    private String rawText;

    @JsonProperty("book_info")
    private BookInfo bookInfo;
}
