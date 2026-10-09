"""Agent 的工具集（M3）。

这个文件和 app/services/ 的分工：

    services/   是「能力本身」—— 抽取、检索。它不知道 agent 的存在。
    agent/      是「给模型用的接口」—— 只负责把能力包成模型能调的形状。

M3 要两个工具（见 docs/需求与接口.md 第 370~371 行）：

    search_books      语义检索找书      → 走本地（M2 的 search()）
    get_post_detail   查单条帖子详情    → 回调 Spring Boot（还没写，第 4 步）

⚠️ 写完先单测这个文件，**别接 agent、别接 LLM**。
   接上 agent 之后出了错，你会分不清是「检索错了」还是「模型选错了工具」，
   而这两种问题的修法完全不同。
"""
from langchain_core.tools import tool

from app.clients.book_service import get_post
from app.services.search import search

_SEARCH_POOL = 50
_PRICE_UNKNOWN = -1.0 
_STATUS_CN = {"ON_SALE": "在售", "SOLD": "已卖出"}

@tool(parse_docstring=True)
def search_books(
    query: str,
    has_notes: bool | None = None,
    price_max: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """按语义找二手书。

    把 M2 的 search() 包成 agent 能调的一个动作。检索的分工不变：
        「这是哪本书」  →  向量库（容忍口语、简称、别名、错别字）
        「什么条件」    →  元数据过滤（有没有笔记、多少钱 —— 这个必须精确）
    一个走向量、一个走过滤。

    Args:
            query:     自然语言查询
            has_notes: True = 只要有笔记的;False = 没笔记的和没有说明的都要;None = 不筛。
            price_max: 价格上限(元)。None = 不筛。                           
            top_k:     最多返回几条。


    """
    # ⚠️ 这里要 50 条（_SEARCH_POOL），比搜索引擎自己的召回池（20）大 ——
    # 因为后面要按 has_notes / price 过滤，过滤完可能剩不下几条，多要一点才够挑。
    # 精排（rerank）只会重排前 20 条（config 的 rerank_pool），第 21~50 名保持 RRF 顺序。
    # 这是有意的：精排是 CPU 上的逐对前向，50 条要 3~5 秒，对话里等不起。
    old_list = search(query,_SEARCH_POOL)
    results = []
    for hits in old_list:
        meta = hits["metadata"]

        #用户没给这一栏就不查
        if has_notes is not None and meta["has_notes"] != has_notes:
            continue

        if price_max is not None:
            price = meta["price"]
            if price != _PRICE_UNKNOWN and price > price_max:
                continue

        results.append({
            "post_id": meta["post_id"],
            "book_name": meta["book_name"],
            "price": meta["price"] if meta["price"] != _PRICE_UNKNOWN else "未知",
        })
    return results[:top_k]

@tool(parse_docstring=True)
def get_post_detail(post_id: int) -> list[dict]:
    """
    查某一条帖子的完整信息

    search_books 返回的是「一批书的摘要」，每条只有书名、价格、有没有笔记三栏。
    这个工具按 id 回主服务取那一条的全量字段 —— 成色描述、卖家原话、还在不在售，
    这些摘要里都没有。

    用户追问某本书的具体情况（成色怎么样 / 书里划得多不多 / 还在卖吗）时用它。
    用户只是要一份列表时不用逐条查 —— 那是 N+1 次 HTTP，慢且没必要。

    Args:
        post_id: 来自search_book的metadata里的字段
    """
    post = get_post(post_id)

    # 主服务对不存在的 id 返回的是 **200 + 空响应体**（不是 JSON 的 null，
    # 见 clients/book_service.py 里的判空），到这儿就是 None。
    # 必须返回**空列表**、不能返回 None —— loop.py 那句 count = len(results)
    # 会在 None 上抛 TypeError，然后被 try 兜成一条假的「工具执行出错」，
    # 你就完全看不到「其实是没查到」这个真相了。
    if post is None:
        return []

    price = post.get("price")
    notes = post.get("hasNotes")

    # 价格 null **不能**翻成「面议」：实体注释说「面议为 null」，抽取 schema 说
    # 「抽不到填 null」—— 两种含义挤在同一个值里。实测 id=209 的 rawText 是「出一本高数」，
    # 压根没提价格，真实含义是「没标价」。取更弱的断言，别替卖家承诺可以还价。
    # （和 has_notes 的三态丢失是同一类：源头分不清的事，出口不能替它下结论。）
    price_out = price if price is not None else "未标价"
    notes_out = notes if notes is not None else "未提及"
    status_out = _STATUS_CN.get(post.get("status") , "状态未知")

    return [
        {
            "post_id": post_id,
            "book_name": post.get("bookName"),
            "edition": post.get("edition"),
            "publisher": post.get("publisher"),
            "author": post.get("author"),
            "condition_desc": post.get("conditionDesc"),
            "price": price_out,
            "has_notes": notes_out,
            "status": status_out,
            # 卖家原话。信息量最大的一栏，也是「详情」区别于「列表页」的地方 ——
            # 用户问「书角卷了吗」这种开放问题，模型只能从这里答。
            "raw_text": post.get("rawText"),
        }
    ]

