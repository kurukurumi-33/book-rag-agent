package com.book.service;

import com.baomidou.mybatisplus.spring.service.IService;
import com.book.dto.ExtractPendingResponse;
import com.book.entity.BookPost;

public interface BookPostService extends IService<BookPost> {

    /**
     * 抽一批 extract_status='PENDING' 的帖子，调 AI 服务抽取后写回。
     *
     * 一次只处理一批（条数由 ai-service.batch-size 决定），调用方反复调直到 remaining 为 0。
     *
     * @return 本轮的统计结果
     */
    ExtractPendingResponse extractPendingBatch();
}
