"""M3 spike：看一眼 tool calling 的原始协议。

这个脚本**不执行任何工具**。它只做一件事：
让模型「想调一个工具」，然后把返回的东西整个打印出来。

为什么用假工具 get_weather：
    这次要看的是**协议**，不是检索。假工具让脚本不依赖索引 / 数据库 /
    Spring Boot —— 十几行就能跑完，排除所有干扰。

用法（在 D:\\agent-book\\ai-service 下）：

    .venv\\Scripts\\python.exe scripts\\try_tool_call.py
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import get_llm  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

llm = get_llm( temperature=0.0)
# 手写 schema 而不用 @tool —— 因为要看见「即将发给 API 的原始 JSON」。
# 注意 function.description 这个字段：它就是模型判断「要不要调这个工具」的依据。

TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"query_api",
            "description":"查看某只股票的具体价格",
            "parameters":{
                "type":"object",
                "properties":{
                    "input":{"type":"string","descrpition":"查询参数"}
                },
                "required":["city"]
            },
        },
    }
]
bound = llm.bind_tools(TOOLS)
resp = bound.invoke([HumanMessage("北京天气怎么样")])
print("=" * 70)
print("返回的对象类型 :", type(resp).__name__)
print("=" * 70)
print("resp.content    :", repr(resp.content))
print("resp.tool_calls :", resp.tool_calls)
print()
print("---------------------------------------------------------------")
print("整条消息：")
print("---------------------------------------------------------------")
print(resp)
