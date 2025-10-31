import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# --- 配置 ---
# 1. 你的查询（你想问的问题）
#YOUR_QUERY = "嘉靖是否沉迷丹药"
# YOUR_QUERY = "土木堡之变"
#YOUR_QUERY = "张居正的改革"
YOUR_QUERY = "浙江"

# 2. 你想返回几个最相关的结果？
TOP_K = 5

# 3. 指定模型和文件（必须和 1_build_index.py 中的一致）
MODEL_NAME = 'shibing624/text2vec-base-chinese'
INDEX_FILE = 'mingshi.index'
DATA_FILE = 'mingshi_data.pkl'

# --- 准备工作：加载所有工具 ---
print("正在加载模型、索引和数据... (会比构建时快很多)")
model = SentenceTransformer(MODEL_NAME, device='cpu')
index = faiss.read_index(INDEX_FILE)

with open(DATA_FILE, 'rb') as f:
    corpus_chunks = pickle.load(f)

print("--- 系统准备就绪，开始搜索 ---")

# ---------------------------------------------------------------
# 核心步骤 1: 将你的“问题”也向量化
# ---------------------------------------------------------------
# 注意：encode函数需要一个列表，所以用 [YOUR_QUERY]
query_vector = model.encode(
    [YOUR_QUERY],
    normalize_embeddings=True # 同样需要标准化
)

# ---------------------------------------------------------------
# 核心步骤 2: 在 FAISS 索引中执行搜索
# ---------------------------------------------------------------
# index.search 会返回两个东西:
# D = 相似度得分 (Distances/Scores)
# I = 对应文本块的索引ID (Indices)
D, I = index.search(query_vector, TOP_K)

# ---------------------------------------------------------------
# 核心步骤 3: 打印结果
# ---------------------------------------------------------------
print(f"\n查询: “{YOUR_QUERY}”")
print(f"找到 Top {TOP_K} 个最相关的史料原文：\n")

for i, score in zip(I[0], D[0]):
    # I[0] 是一个包含k个索引ID的列表 [id1, id2, ...]
    # D[0] 是一个包含k个分数的列表 [score1, score2, ...]
    # i 就是 corpus_chunks 列表中的位置
    
    print(f"【相关度得分: {score:.4f}】")
    print("【史料原文】:")
    print(corpus_chunks[i].strip()) # strip() 用于去除首尾多余的空白
    print("-" * 50 + "\n")