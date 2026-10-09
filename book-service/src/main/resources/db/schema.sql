-- 智觅（二手书智能匹配 Agent） - 数据库结构
-- 执行：mysql -uroot -p < schema.sql

CREATE DATABASE IF NOT EXISTS book_agent
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE book_agent;

DROP TABLE IF EXISTS book_post;

CREATE TABLE book_post (
    id              BIGINT          NOT NULL AUTO_INCREMENT COMMENT '主键',
    user_id         BIGINT          NOT NULL                COMMENT '发布者ID（简化，暂不做登录）',

    -- ===== 输入：卖家发布的原始描述（混乱的自然语言）=====
    raw_text        TEXT            NOT NULL                COMMENT '原始帖子文本，AI 抽取的输入',

    -- ===== 输出：AI 服务抽取出的结构化字段（抽取前为 NULL）=====
    book_name       VARCHAR(255)    DEFAULT NULL            COMMENT '书名',
    edition         VARCHAR(64)     DEFAULT NULL            COMMENT '版次，如 第七版',
    publisher       VARCHAR(128)    DEFAULT NULL            COMMENT '出版社',
    author          VARCHAR(128)    DEFAULT NULL            COMMENT '作者',
    condition_desc  VARCHAR(255)    DEFAULT NULL            COMMENT '成色描述',
    price           DECIMAL(10, 2)  DEFAULT NULL            COMMENT '价格',
    has_notes       TINYINT(1)      DEFAULT NULL            COMMENT '是否带笔记',

    -- ===== 状态 =====
    status          VARCHAR(32)     NOT NULL DEFAULT 'ON_SALE' COMMENT 'ON_SALE / SOLD',
    extract_status  VARCHAR(32)     NOT NULL DEFAULT 'PENDING' COMMENT 'AI 抽取状态 PENDING / DONE / FAILED',

    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    KEY idx_status (status),
    KEY idx_extract_status (extract_status),
    KEY idx_book_name (book_name)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COMMENT = '二手书帖子';


-- ============================================================================
-- M3 第 4 步：Agent 的会话历史
-- ============================================================================

DROP TABLE IF EXISTS chat_message;

CREATE TABLE chat_message (
    id          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键',
    session_id  VARCHAR(64)  NOT NULL                COMMENT '会话 ID，由 AI 服务生成',
    role        VARCHAR(16)  NOT NULL                COMMENT 'human=用户说的 / ai=模型的最终回答',
    content     TEXT         NOT NULL                COMMENT '消息正文',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    -- 查询永远是「某会话最近 N 条，按顺序」。id 自增即顺序，
    -- 所以把两者放一个联合索引里：先按 session_id 定位，再靠 id 直接有序取。
    KEY idx_session_id_id (session_id, id)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COMMENT = 'Agent 会话历史';

-- 为什么用自增 id 当顺序，而不另加一个 seq：
-- 一轮的两条消息是一次事务写进去的，自增 id 天然保证它们相邻且有序。
-- 另加 seq 就得自己维护「这个会话下一条是几」，还要处理并发写同一个会话 —— 白给自己找事。
