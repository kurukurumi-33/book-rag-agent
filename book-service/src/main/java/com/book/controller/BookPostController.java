package com.book.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.book.dto.ExtractPendingResponse;
import com.book.entity.BookPost;
import com.book.service.BookPostService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 帖子接口。
 *
 * 这里的接口不只给前端用 —— AI 服务里 agent 的 tool 也会回调它们来取数据。
 * 所有接口的文档由 springdoc 从下面的注解自动生成：/swagger-ui.html
 */
@RestController
@RequestMapping("/api/posts")
@RequiredArgsConstructor
@Tag(name = "帖子", description = "二手书帖子的增查接口。AI 服务里 agent 的 tool 会回调这些接口取业务数据")
public class BookPostController {

    private final BookPostService bookPostService;

    @GetMapping
    @Operation(
            summary = "帖子列表",
            description = """
                    按 id 倒序返回。`limit` 会被截断到 500，防止一次拉爆内存。

                    ## 为什么有 `extractStatus` 这个过滤参数

                    AI 服务建检索索引（M2）时，要先调**这个接口**拿到 `DONE` 的帖子 ——
                    按铁律一，AI 服务不直连数据库，业务数据一律从这里走。

                    所以这个参数不是给前端用的，是**给 AI 服务用的**。
                    """
    )
    public List<BookPost> list(
            @Parameter(description = "返回条数，默认 50，上限 500")
            @RequestParam(defaultValue = "50") int limit,
            @Parameter(description = "按抽取状态过滤：PENDING / DONE / FAILED；不传返回全部")
            @RequestParam(required = false) String extractStatus) {
        return bookPostService.list(
                new LambdaQueryWrapper<BookPost>()
                        .eq(extractStatus != null && !extractStatus.isBlank(),
                                BookPost::getExtractStatus, extractStatus)
                        .orderByDesc(BookPost::getId)
                        .last("LIMIT " + Math.min(limit, 500))
        );
    }

    @GetMapping("/{id}")
    @Operation(
            summary = "帖子详情",
            description = "AI 服务里 `get_post_detail` 工具就是回调这个接口。查不到返回 null。"
    )
    public BookPost getById(
            @Parameter(description = "帖子 id") @PathVariable Long id) {
        return bookPostService.getById(id);
    }

    @PostMapping
    @Operation(
            summary = "发布帖子",
            description = """
                    只需要传 `userId` 和 `rawText` 两个字段。

                    结构化字段（bookName / price 等）**不要传**，
                    它们由 AI 服务抽取后回填，这里会被强制置为 null。

                    `status` 固定为 ON_SALE，`extractStatus` 固定为 PENDING。
                    """
    )
    public BookPost create(@RequestBody BookPost post) {
        post.setId(null);
        post.setStatus("ON_SALE");
        post.setExtractStatus("PENDING");
        // 抽取字段由 AI 回填，不接受客户端传入
        post.setBookName(null);
        post.setEdition(null);
        post.setPublisher(null);
        post.setAuthor(null);
        post.setConditionDesc(null);
        post.setPrice(null);
        post.setHasNotes(null);
        bookPostService.save(post);
        return post;
    }

    @GetMapping("/count")
    @Operation(summary = "帖子总数")
    public long count() {
        return bookPostService.count();
    }

    @PostMapping("/extract-pending")
    @Operation(
            summary = "批量抽取待处理的帖子（M1 编排入口）",
            description = """
                    主服务作为**编排方**，把「查数据 → 调 AI → 写回」串起来。
                    AI 服务全程不知道数据库的存在。

                    ## 一次处理一批

                    1. 查 `extractStatus = PENDING` 的帖子，最多 `ai-service.batch-size` 条（默认 50）
                    2. 把它们的 `rawText` 打包，发给 AI 服务的 `POST /extract/batch`
                    3. 按**下标**把结果写回对应帖子 —— 不用文本匹配，因为 `rawText` 可能重复
                    4. 成功的置 `DONE`，失败的置 `FAILED`

                    条数固定成一批，是因为一批 50 条 ≈ 35 秒，已经接近 HTTP 超时的舒适区。
                    **调用方反复调这个接口，直到返回的 `remaining` 为 0。**

                    ## 单条失败不牵连其他条

                    某条抽取失败只把它自己标 `FAILED`，其余照常写回，失败的 id 和原因
                    在 `errors` 里返回。写回也不套事务，避免第 50 条失败把前 49 条一起回滚。

                    **不自动重试**：失败原因没解决就重试等于反复烧 token。
                    要重试就手动把那一行的 `extract_status` 改回 `PENDING`。

                    ## 恒等式

                    `picked == success + failed`，对不上说明代码有 bug。
                    """
    )
    @ApiResponse(responseCode = "200", description = "本轮结果统计")
    @ApiResponse(responseCode = "503", description = "AI 服务不可用（Python 服务没起 / 端口不对 / 超时）")
    @ApiResponse(responseCode = "500", description = "AI 返回条数与请求不一致，已拒绝写回以免错位")
    public ExtractPendingResponse extractPending() {
        return bookPostService.extractPendingBatch();
    }
}
