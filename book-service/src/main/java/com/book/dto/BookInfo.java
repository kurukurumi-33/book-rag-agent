package com.book.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.math.BigDecimal;

/**
 * AI 服务抽取出的结构化书籍信息。
 *
 * Python 侧用 snake_case，Java 侧用 camelCase，
 * 通过 @JsonProperty 做映射，两边各用各的命名习惯。
 */
@Data
public class BookInfo {

    @JsonProperty("book_name")
    private String bookName;

    private String edition;

    private String publisher;

    private String author;

    @JsonProperty("condition_desc")
    private String conditionDesc;

    private BigDecimal price;

    @JsonProperty("has_notes")
    private Boolean hasNotes;
}
