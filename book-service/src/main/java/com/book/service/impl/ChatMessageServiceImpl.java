package com.book.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.spring.service.impl.ServiceImpl;
import com.book.entity.ChatMessage;
import com.book.mapper.ChatMessageMapper;
import com.book.service.ChatMessageService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

@Service
@RequiredArgsConstructor
public class ChatMessageServiceImpl extends ServiceImpl<ChatMessageMapper, ChatMessage>
        implements ChatMessageService {

    /** 一次最多读回多少条。防止调用方传个 100000 把内存和模型上下文一起撑爆。 */
    private static final int MAX_LIMIT = 200;

    /** 允许的角色。跟 LangChain 的 Message.type 对齐，见 ChatMessage 的类注释。 */
    private static final Set<String> ROLES = Set.of("human", "ai");

    /**
     * 允许的 sessionId 形状。跟 Python 侧 `book_service.py` 里的 `_SESSION_ID_RE`
     * **必须保持一致** —— 两边不一致的话，Python 放过来、这边拒绝，报错会很难查。
     *
     * 为什么不放开成「随便什么字符串」：sessionId 出现在 URL 的**路径段**里，
     * 而路径段装不下任意输入 —— 带斜杠的会被路由拆开，带空格的编码成 %20 之后，
     * 带 %2F 的更是被 Tomcat 在容器层直接回 400（默认拒绝路径里的 %2F，防目录穿越），
     * Spring 连方法都进不去。与其在传输层跟编码搏斗，不如把 ID 的形状约束死。
     * 最长的 64 跟建表时的 VARCHAR(64) 对齐。
     */
    private static final Pattern SESSION_ID_PATTERN = Pattern.compile("[A-Za-z0-9_-]{1,64}");

    @Override
    public List<ChatMessage> recent(String sessionId, int limit) {
        requireValidSessionId(sessionId);
        int n = Math.clamp(limit, 1, MAX_LIMIT);

        List<ChatMessage> rows = list(new LambdaQueryWrapper<ChatMessage>()
                .eq(ChatMessage::getSessionId, sessionId)
                // 先倒序取「最近 N 条」，再翻回正序 ——
                // 直接 orderByAsc + LIMIT 会取到**最老的** N 条，正好取反。
                .orderByDesc(ChatMessage::getId)
                .last("LIMIT " + n));

        Collections.reverse(rows);
        return rows;
    }

    @Override
    @Transactional
    public List<ChatMessage> append(String sessionId, List<ChatMessage> messages) {
        requireValidSessionId(sessionId);

        if (messages == null || messages.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "messages 不能为空");
        }

        List<ChatMessage> toSave = new ArrayList<>(messages.size());
        for (ChatMessage m : messages) {
            if (m.getRole() == null || !ROLES.contains(m.getRole())) {
                throw new ResponseStatusException(
                        HttpStatus.BAD_REQUEST,
                        "role 只能是 human / ai，收到的是：" + m.getRole());
            }
            if (m.getContent() == null || m.getContent().isEmpty()) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "content 不能为空");
            }

            // 客户端传什么都不算数：id 和 sessionId 由服务端定，
            // 跟 BookPostController.create 里把抽取字段强制置 null 是同一个思路 ——
            // 不让调用方决定「这一行属于谁、排第几」。
            ChatMessage row = new ChatMessage();
            row.setSessionId(sessionId);
            row.setRole(m.getRole());
            row.setContent(m.getContent());
            toSave.add(row);
        }

        // saveBatch 让两条 INSERT 走同一个事务和同一个批处理会话。
        // 任一条失败 → @Transactional 整体回滚，不会留下「有问无答」的半截历史。
        saveBatch(toSave);
        return toSave;
    }

    private void requireValidSessionId(String sessionId) {
        if (sessionId == null || !SESSION_ID_PATTERN.matcher(sessionId).matches()) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "session_id 只能是字母/数字/下划线/连字符，最长 64 位");
        }
    }
}
