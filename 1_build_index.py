import faiss  # 导入faiss
import numpy as np
import pickle  # 用于保存我们的数据
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# 【已修正】导入正确的“文本切割器”
# ---------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ---------------------------------------------------------------

# --- 配置 ---
# 1. 指定你的史料文件
CORPUS_FILE = 'corpus_ming_history.txt'

# 2. 指定模型
MODEL_NAME = 'shibing624/text2vec-base-chinese'

# 3. 指定保存的索引和数据文件名
INDEX_FILE = 'mingshi.index'
DATA_FILE = 'mingshi_data.pkl'

print(f"正在加载模型 '{MODEL_NAME}'... (第一次运行需要下载，请耐心等待)")
# ---------------------------------------------------------------
# 第一步：加载AI模型
# ---------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME, device='cpu')

# ---------------------------------------------------------------
# 第二步：读取和切割文本
# ---------------------------------------------------------------
print(f"正在读取史料文件 '{CORPUS_FILE}'...")
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# 使用我们刚刚正确导入的 RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)

corpus_chunks = text_splitter.split_text(text)

print(f"史料被成功切分为 {len(corpus_chunks)} 个小段落。")

# ---------------------------------------------------------------
# 第三步：文本向量化 (将文字 -> 向量)
# ---------------------------------------------------------------
print("正在为所有段落生成“语义向量”... (这可能需要几分钟，请稍候)")
corpus_embeddings = model.encode(
    corpus_chunks,
    show_progress_bar=True,  
    normalize_embeddings=True 
)

print(f"向量生成完毕，形状为: {corpus_embeddings.shape}")

# ---------------------------------------------------------------
# 第四步：构建并保存 FAISS 索引
# ---------------------------------------------------------------
print("正在构建 FAISS 索引库...")
d = corpus_embeddings.shape[1] 
index = faiss.IndexFlatIP(d)
index.add(corpus_embeddings)

print(f"正在保存索引到 '{INDEX_FILE}'...")
faiss.write_index(index, INDEX_FILE)

print(f"正在保存原始文本块到 '{DATA_FILE}'...")
with open(DATA_FILE, 'wb') as f:
    pickle.dump(corpus_chunks, f)

print("--- 索引构建完成！---")