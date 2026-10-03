package com.book;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 二手书智能匹配 - 主服务。
 *
 * 接口文档由 springdoc 从注解自动生成，不用手写：
 *   Swagger UI  → http://localhost:8080/swagger-ui.html
 *   OpenAPI spec → http://localhost:8080/v3/api-docs  （可导入 YApi / Apifox / Postman）
 */
@SpringBootApplication
@MapperScan("com.book.mapper")
@OpenAPIDefinition(
        info = @Info(
                title = "二手书主服务",
                version = "0.1.0",
                description = """
                        业务的唯一数据源：MySQL 读写、状态流转、事务都在这里。

                        ## 和 AI 服务的关系

                        ```
                        前端
                          │
                          ▼
                        本服务 (:8080)          ← 业务逻辑 / 数据库
                          │  HTTP
                          ▼
                        AI 服务 (:8000)         ← 只负责「理解」
                          └─ agent 的 tool 回调回本服务取数据
                        ```

                        **AI 服务不直连数据库**，所有业务数据都通过本服务的 REST API 获取。
                        这样业务规则（校验、权限、事务）只需要维护一处。

                        ## 文档

                        - AI 服务文档：http://localhost:8000/docs
                        - 需求与接口说明：`docs/需求与接口.md`
                        """
        )
)
public class BookServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(BookServiceApplication.class, args);
    }
}
