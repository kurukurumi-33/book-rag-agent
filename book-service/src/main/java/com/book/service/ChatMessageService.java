package com.book.service;

import com.baomidou.mybatisplus.spring.service.IService;
import com.book.entity.ChatMessage;

import java.util.List;

public interface ChatMessageService extends IService<ChatMessage> {

    /**
     * 取某个会话最近的 limit 条消息，**按时间正序**返回（最老的在前）。
     *
     * 为什么是「最近的 N 条」而不是「全部」：会话可以无限长，
     * 全量读回来会撑爆模型的上下文窗口。只取最近 N 条是有意的截断，
     * 代价是很久以前说过的话会忘 —— 生产环境要么加大 N，要么做摘要压缩。
     *
     * @param limit 上限，会被截断到 200
     */
    List<ChatMessage> recent(String sessionId, int limit);

    /**
     * 把**一轮**产生的消息一次写进去。要么全部成功，要么全部失败。
     *
     * 一轮 = 用户这一句 + 模型这一答，两条。必须一起进去，
     * 否则历史里会出现「有问无答」，下一轮模型就接不上话。
     * 方法上的 @Transactional 就是为这个加的 —— 跟 BookPostServiceImpl 里
     * 反向的选择（批量写回故意不套事务，避免 50 条被 1 条拖累）对照着看。
     *
     * @return 写进去的行，带上了数据库生成的自增 id
     */
    List<ChatMessage> append(String sessionId, List<ChatMessage> messages);
}
