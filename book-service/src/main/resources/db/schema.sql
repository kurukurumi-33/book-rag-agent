-- 二手书智能匹配 - 数据库结构
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
