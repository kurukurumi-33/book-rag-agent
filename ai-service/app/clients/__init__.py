"""对外部服务的 HTTP 客户端。

跟 app/services/ 的区别：
- clients/  —— 「怎么把数据拿过来」，只关心 HTTP、超时、错误码
- services/ —— 「拿到数据之后做什么」，只关心业务逻辑

分开是为了能换实现：哪天主服务换成 gRPC，只动 clients/ 这一个文件。
"""
