"""手写 agent loop（M3 第 3 步）。

框架替你做的事，剥开就这么几行：

    把 [用户消息] + [工具定义] 发给模型
    → 模型不说人话，它下工单：「我要调 search_books，参数 query="高数"」
    → 你**替它执行**，把结果作为一条新消息塞回列表
    → 再发一次
    → 这次模型说人话（content 非空、tool_calls 为空）→ 循环结束

这一个**全部手写**，不用 LangChain 的 AgentExecutor。
框架版作为**对照**另写在一个函数里（见 docs/需求与接口.md §6.2 决策 2）——
是并存不是替换：手写的这个才是讲「框架替我做了什么、代价是什么」的参照物。

协议形态在第 1 步实测过（scripts/try_tool_call.py）：

    content=''  +  finish_reason='tool_calls'  +  tool_calls=[{name, args, id}]

**模型只下工单，不动手。动手的是你。** 这个循环就是"你动手"的那部分。

两个工具（search_books / get_post_detail）现在都在用：
search_books 按语义搜一批，get_post_detail 按 post_id 回主服务取单条全量字段。
后者是「双服务架构」在 Agent 上的体现 —— 它不是本地函数，是 HTTP 回调。
"""
import json
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent.tools import get_post_detail, search_books
from app.services.llm import get_llm

SYSTEM_PROMPT = """你是二手书交易平台的导购助手，帮买家从平台库存里找书。
只根据工具返回的结果回答。库里没有的书、没标的价格，就直接说没有/未知，
不要凭常识补充书名、版本或价格。
用户问与找书无关的事，礼貌说明你只负责找书。

你有两个工具，配合使用：
- search_books     按语义搜一批书，返回摘要（书名 / 价格 / 有没有笔记）
- get_post_detail  按 post_id 取某一条的完整信息（成色、卖家原话、还在不在售）

找书用 search_books。用户追问某本的具体情况——成色怎么样、书里划得多不多、
还在卖吗——先从 search_books 的结果里拿到那本的 post_id，再调 get_post_detail。
用户没指明是哪一本时，先问清楚再看详情，不要对列表里每条都调一遍 get_post_detail。
摘要里没有的字段不要凭印象编。用户只是要一份列表时，直接回答就行，不用逐条查详情。
"""
MAX_ROUNDS = 5  # 最多几轮工具调用，防死循环（见 docs/需求与接口.md §6.2 决策 4）
FAILED_ANSWER = "抱歉，我查了几轮还是没能给你答案。换个说法，或者把条件放宽一点再问一次？"
RECURSION_LIMIT=12


def chat_once( 
    user_message: str,
    history: list | None = None,
) -> tuple[str, list[dict], list]:
    """跑一轮完整的 agent 循环：用户说一句，agent 回一句。

    :param user_message: 用户这一句原话
    :param history: 这个会话**之前**的消息。调用方存着、这一轮传回来；不传 = 新会话
    :return: (给用户看的回答, 这一轮的工具调用轨迹, 这个会话**之后**的消息)
             轨迹每项形如 {"name": "search_books", "arguments": {...}, "result_count": 3}
             —— §6.3 要求 `POST /chat` 必须暴露 tool_calls，这个返回值就是它的数据源
             第三个返回值交给调用方去存 —— chat_once 不关心历史存在哪
             （内存 / MySQL / Redis 都能换，见下面的「历史规则」）

    ==========================================================================
    实现要点（每条都对应下面代码里的一处，也都真踩过坑）
    ==========================================================================
    1. 消息从一条 SystemMessage + 一条 HumanMessage 开始，history 拼在中间。

    2. bind_tools(list(TOOLS.values())) 绑工具定义。
       ⚠️ 必须传**列表** —— 传单个工具对象会 ValueError: Unsupported function。

    3. ToolMessage 的 tool_call_id 是「结果」对「工单」的回执。
       一次可能来好几张工单（模型并行调用），少回一条 → 下一轮 400：
       "An assistant message with 'tool_calls' must be followed by tool messages
        responding to each 'tool_call_id'."
       **工单和结果双向配对**，所以 append(ToolMessage(...)) 必须落在所有分支的
       汇合点上 —— 任何 continue / return 都不能出现在它前面（面试素材第 17 条）。

    4. 工具抛异常 → **不让它冒出去**，包成一条 ToolMessage 喂回模型。
       冒出去整个 /chat 变 500，用户只看到「服务出错」，模型也没机会自己补救。

    5. 工具名是编的（幻觉）→ 也**照样回一条结果**说明可用工具，不是跳过。
       用 TOOLS.get(name) 判空，不是 TOOLS[name] —— 后者抛 KeyError。
       实测模型拿到这句话会自己换正确的工具重开一张工单。

    6. 跑满 MAX_ROUNDS 还在要工具 → 返回 FAILED_ANSWER。
       这条路径正常问句走不到，要验证得临时把 MAX_ROUNDS 改成 1 才撬得出来。

    7. 工单参数是 dict，靠 @tool 生成的 schema 转成一次真实调用
       （tool.invoke(call["args"])）。不手拼参数字典 —— 否则模型看到的参数名
       和实际执行的那套会错位。

    8. SystemMessage 决定角色和边界，写什么模型就只做什么。
       （实测：拿 M1 抽取器那句喂同一段帖子原文，它会回一段 JSON。）

    ==========================================================================
    历史规则（两个 return 都要用）
    ==========================================================================
    history 是「上一轮到这一轮之间」的消息。但**不能把整个 messages 列表存下来**，
    它里面混着三种东西：

        HumanMessage               用户说的话
        AIMessage(tool_calls 非空) 模型下的工单
        ToolMessage                工具的执行结果

    后两种必须**成对出现** —— 只留工单不留结果，下一轮 invoke 直接报错。
    而且每一轮的工具返回都会重新发给模型，全留着 token 会随轮数成倍涨。

    所以要**筛**：只留 HumanMessage 和最后那条 AIMessage
    (tool_calls 为空的那条）。中间的工具往返全丢 ——
    查到的信息已经写在最终回答里了。
    """
    llm = get_llm(temperature=0.0)

    TOOLS = {t.name : t for t in [search_books, get_post_detail]}
    bound = llm.bind_tools(list(TOOLS.values()))

    text = (user_message or "").strip()
    messages = [SystemMessage(SYSTEM_PROMPT)] + list(history or []) + [HumanMessage(text)]
    trace = []

    for _ in range(MAX_ROUNDS):
     resp = bound.invoke(messages)

     if not resp.tool_calls:
        return resp.content,trace,(history or []) + [HumanMessage(text),resp]

     messages.append(resp)

     for call in resp.tool_calls:
        post_ids = []
        tool = TOOLS.get(call["name"])
        if tool is None :
          content = f'没有叫{call["name"]}的工具,可用的工具只有：{'、'.join(TOOLS)}'
          count = 0
        else:
           try:
              results = tool.invoke(call["args"])
              content = str(results)
              count = len(results)
             # 判键在不在 —— 缺键不能让它抛，那会把上面两行一起抹掉
              post_ids =[r["post_id"]for r in results if "post_id" in r]
           except Exception as e:
               content = f"工具执行出错:{type(e).__name__}:{e}"
               count = 0
               post_ids = []

        messages.append(ToolMessage(
            content = content,
            tool_call_id = call["id"],
         ))
        trace.append({
           "name":call["name"],
           "arguments":call["args"],
           "result_count":count,
           "post_ids":post_ids,
        })
    return FAILED_ANSWER,trace,(history or []) + [HumanMessage(text),AIMessage(FAILED_ANSWER)]

def chat_once_framework(user_message: str, history: list | None = None) -> tuple[str, list[dict], list]:
    """同样的活，用 langchain.agents.create_agent 干。跟 chat_once 并存，做对照。"""
    agent = create_agent(model = get_llm(temperature=0.0),
                         tools = [search_books, get_post_detail],
                         system_prompt=SYSTEM_PROMPT)
    text = (user_message or "").strip()
    messages = list(history or []) + [HumanMessage(text)]
    trace = []
    #invoke的参数必须是dict
    try:
     resp = agent.invoke(
        {"messages": messages},
        config={"recursion_limit":RECURSION_LIMIT},
     )
    except GraphRecursionError: 
     return FAILED_ANSWER, trace, (history or []) + [HumanMessage(text), AIMessage(FAILED_ANSWER)]
    out_msgs = resp["messages"]
    for i,m in enumerate(out_msgs):
       if not isinstance(m,AIMessage):
          continue
       
       if not m.tool_calls:
         continue
       
       for call in m.tool_calls:
             post_ids = []
             count = 0
             for later in out_msgs[i+1:]:
                if not isinstance(later,ToolMessage):
                   continue
                if later.tool_call_id != call["id"]:
                   continue
                try:
                   rows = json.loads(later.content)
                   count = len(rows)
                   post_ids = [
                      r["post_id"]for r in rows 
                      if isinstance(r,dict) and "post_id" in r
                   ]
                except Exception:
                   count = 0
                   post_ids = []
                break
             trace.append({
                "name":call["name"],
                "arguments":call["args"],
                "result_count":count,
                "post_ids":post_ids,
             })
    reply = out_msgs[-1].content
    return(
          reply,
          trace,
          (history or [])+[HumanMessage(text),out_msgs[-1]],
       )
                      
                   
                   
                
          


