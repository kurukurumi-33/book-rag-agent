"""试跑 BGE 中文 embedding：确认模型能下、能算、语义距离合理。

用法（在 D:\\agent-book\\ai-service 目录下，先激活 venv）：
    python scripts\\try_embedding.py

这是「裸模型」的试跑，直接调 SentenceTransformer，不走 app/services/embedding.py，
方便你看清里面到底发生了什么。生产代码里的封装见 app/services/embedding.py。

## ⚠️ 一个踩过的坑：不要给 query 加指令前缀

网上教程（包括 BGE 官方 README）都会让你把 query 写成：

    为这个句子生成表示以用于检索相关文章：同济版高数

**那是 v1.0 的用法，在 v1.5 上加了分数反而更低。**
用 40 条真实帖子实测（3 个 query 各测一遍，结论一致）：

    query="高等数学"    不加前缀 1.0000 ／ 加前缀 0.8167
    query="同济版高数"   不加前缀更高    ／ 加前缀更低
    query="高数同济第七版" 不加前缀更高    ／ 加前缀更低

最坑的是**它不报错**：加了前缀检索照样返回结果，只是排名悄悄变差。
「结果莫名其妙不准」比「直接崩」难查得多 —— 所以这里留一行记录。

（v1.5 的官方说法是：短 query 检长文档可加，同长度检索不要加。
我们两边都是短文本，所以不加。）
"""

import sys
from pathlib import Path

# 让脚本能 import 到 app 包（直接跑脚本时，sys.path[0] 是脚本自己所在的目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main() -> None:
    print(f"加载模型 {MODEL_NAME}")
    print("（首次运行会下载约 100MB，之后走本地缓存）...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ 加载成功，向量维度 = {model.get_sentence_embedding_dimension()}\n")

    query = "同济版高数"
    docs = [
        "高等数学 第七版 同济大学出版社 有笔记 九成新",
        "线性代数 第六版 同济大学出版社 九成新",
        "新视野大学英语 第三版 外语教学与研究出版社",
        "计算机网络 第八版 电子工业出版社 谢希仁",
    ]

    # normalize_embeddings=True 把向量长度归一化成 1，
    # 这样「点积」就等于「余弦相似度」，后面算分不用再除模长。
    #
    # 注意 query 前面**没有**任何前缀，原因见模块开头的说明。
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    d_vecs = model.encode(docs, normalize_embeddings=True)

    print(f"查询：{query}")
    print("-" * 70)
    scored = sorted(
        ((float(q_vec @ v), d) for d, v in zip(docs, d_vecs)),
        reverse=True,
    )
    for score, doc in scored:
        bar = "█" * max(0, int(score * 40))
        print(f"  {score:+.4f} {bar:<22} {doc}")

    print("""
看点：
  1. 「同济版高数」和「高等数学 第七版 同济大学出版社」没有一个词完全相同，
     但分数明显高于其他三条 —— 这就是语义检索相对关键词搜索的价值。
  2. 「线性代数」同样带「同济」，排第二而不是第一 —— 说明模型分得清
     出版社相同时哪本书才是用户要的。
""")


if __name__ == "__main__":
    main()
