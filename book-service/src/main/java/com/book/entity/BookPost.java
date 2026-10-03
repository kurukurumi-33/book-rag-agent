package com.book.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 二手书帖子。
 *
 * 核心设计：rawText 是卖家发布的原始描述（混乱的自然语言），
 * 下面那批结构化字段是 AI 服务抽取出来的。抽取前它们都是 null。
 *
 * @Schema 注解会被 springdoc 读进 Swagger UI 的字段说明，改说明等于改文档。
 */
@Data
@TableName("book_post")
@Schema(description = "二手书帖子")
public class BookPost {

    @TableId(type = IdType.AUTO)
    @Schema(description = "主键", example = "1")
    private Long id;

    @Schema(description = "发布者 ID（简化处理，暂不做登录）", example = "42")
    private Long userId;

    // ---------- 输入 ----------

    @Schema(
            description = "卖家发布的原始描述，AI 抽取的输入。**这是唯一必填的业务字段**",
            example = "高数同济七版 上下册一起 40 有笔记 划重点了 可小刀"
    )
    private String rawText;

    // ---------- 以下由 AI 服务抽取填充 ----------

    @Schema(description = "书名（AI 抽取，归一化后的标准全名）", example = "高等数学")
    private String bookName;

    @Schema(description = "版次（AI 抽取）", example = "第七版")
    private String edition;

    @Schema(description = "出版社（AI 抽取，原文没写则为 null）")
    private String publisher;

    @Schema(description = "作者（AI 抽取）")
    private String author;

    @Schema(description = "成色描述（AI 抽取）", example = "有笔记，划过重点")
    private String conditionDesc;

    @Schema(description = "价格（元，AI 抽取）。免费为 0，面议为 null", example = "40.00")
    private BigDecimal price;

    @Schema(description = "是否带笔记（AI 抽取，未提及为 null）", example = "true")
    private Boolean hasNotes;

    // ---------- 状态 ----------

    @Schema(description = "售卖状态", allowableValues = {"ON_SALE", "SOLD"}, example = "ON_SALE")
    private String status;

    @Schema(
            description = "AI 抽取状态。PENDING=待抽取，DONE=已抽取，FAILED=抽取失败",
            allowableValues = {"PENDING", "DONE", "FAILED"},
            example = "PENDING"
    )
    private String extractStatus;

    @Schema(description = "创建时间")
    private LocalDateTime createdAt;

    @Schema(description = "更新时间")
    private LocalDateTime updatedAt;
}
