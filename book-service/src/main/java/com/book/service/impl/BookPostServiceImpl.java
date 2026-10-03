package com.book.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.spring.service.impl.ServiceImpl;
import com.book.client.AiServiceClient;
import com.book.dto.BatchItemResult;
import com.book.dto.BookInfo;
import com.book.dto.ExtractBatchResponse;
import com.book.dto.ExtractPendingResponse;
import com.book.entity.BookPost;
import com.book.mapper.BookPostMapper;
import com.book.service.BookPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class BookPostServiceImpl extends ServiceImpl<BookPostMapper, BookPost>
        implements BookPostService {

    private final AiServiceClient aiServiceClient;

    @Value("${ai-service.batch-size}")
    private int batchSize;

    @Override
    public ExtractPendingResponse extractPendingBatch() {
        ExtractPendingResponse result = new ExtractPendingResponse();
        result.setErrors(new ArrayList<>());

        // ---------- 1. 捞一批待抽取的帖子 ----------
        // 按 id 升序：先发的先处理，进度可预期；也用上了 idx_extract_status 索引
        List<BookPost> posts = list(new LambdaQueryWrapper<BookPost>()
                .eq(BookPost::getExtractStatus, "PENDING")
                .orderByAsc(BookPost::getId)
                .last("LIMIT " + batchSize));

        result.setPicked(posts.size());

        if (posts.isEmpty()) {
            result.setRemaining(0);
            return result;
        }

        // ---------- 2. 只把文本发出去 ----------
        // 业务实体（id / userId / 状态）一律不出主服务，AI 服务只看到纯文本。
        List<String> texts = posts.stream().map(BookPost::getRawText).toList();
        ExtractBatchResponse aiResponse = aiServiceClient.extractBatch(texts);
        List<BatchItemResult> items = aiResponse.getResults();

        // ---------- 3. 对齐校验 ----------
        // 写回靠的是「下标 i」，一旦长度对不上，继续写就会把 A 帖的抽取结果写到 B 帖上 ——
        // 这是最难查的一类脏数据（数据看着正常，只是属于别人）。
        // 所以宁可整批放弃抛错，也不能错位写。
        if (items == null || items.size() != posts.size()) {
            throw new IllegalStateException(
                    "AI 返回条数与请求不一致：请求 " + posts.size() + " 条，返回 "
                            + (items == null ? "null" : items.size()) + " 条，拒绝写回以免错位");
        }

        // ---------- 4. 逐条写回 ----------
        // 这里刻意**不套 @Transactional**：每行独立提交。
        // 如果整批一个事务，第 50 条写库失败会把前 49 条已经抽好的结果一起回滚 ——
        // 而 token 已经花掉了，钱白花。
        // 这个「单条失败不牵连其他条」的取向，和 AI 侧 return_exceptions=True 是对称的。
        int success = 0;
        int failed = 0;
        for (int i = 0; i < posts.size(); i++) {
            BookPost post = posts.get(i);
            BatchItemResult item = items.get(i);

            if (item.getBookInfo() != null) {
                applyBookInfo(post, item.getBookInfo());
                post.setExtractStatus("DONE");
                updateById(post);
                success++;
            } else {
                // 抽取失败：只改状态，7 个业务字段保持 null。
                // 留个 FAILED 标记在那儿可见可查，**不自动重试** —— 失败原因没解决时
                // 自动重试就是反复烧 token。要重试就把这行改回 PENDING。
                post.setExtractStatus("FAILED");
                updateById(post);
                result.getErrors().add("帖子 " + post.getId() + "：" + item.getError());
                failed++;
            }
        }

        result.setSuccess(success);
        result.setFailed(failed);
        result.setRemaining(count(new LambdaQueryWrapper<BookPost>()
                .eq(BookPost::getExtractStatus, "PENDING")));
        return result;
    }

    /** 把 AI 抽出来的字段搬到实体上。抽不到的字段是 null，照搬，不做任何补全。 */
    private void applyBookInfo(BookPost post, BookInfo info) {
        post.setBookName(info.getBookName());
        post.setEdition(info.getEdition());
        post.setPublisher(info.getPublisher());
        post.setAuthor(info.getAuthor());
        post.setConditionDesc(info.getConditionDesc());
        post.setPrice(info.getPrice());
        post.setHasNotes(info.getHasNotes());
    }
}
