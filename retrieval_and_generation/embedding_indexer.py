import faiss
import numpy as np
import os
import torch
from tqdm import tqdm  # 导入tqdm

class EmbeddingIndexer:
    def __init__(self, index_dir="index"):
        self.index_dir = index_dir
        self.index_path = os.path.join(self.index_dir, "faiss_index")
        self.metadata_path = os.path.join(self.index_dir, "metadata.pkl")

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        self.dimension = 384  # Sentence-BERT 维度
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def add_embeddings(self, embeddings, texts, sources):
        """ 添加文章的嵌入到索引中 """
        vectors = np.vstack([e.cpu().numpy() for e in embeddings])
        self.index.add(vectors)
        self.metadata.extend(zip(texts, sources))
        self.save_index()

    def save_index(self):
        """ 保存FAISS索引和元数据 """
        faiss.write_index(self.index, self.index_path)
        torch.save(self.metadata, self.metadata_path)

    def load_index(self):
        """ 加载FAISS索引和元数据 """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            self.metadata = torch.load(self.metadata_path)
            return True
        return False

    def create_index(self, articles, model):
        """ 创建索引并显示进度条 """
        texts, sources = zip(*articles)
        embeddings = [model.encode(text, convert_to_tensor=True) for text in texts]

        # 使用 tqdm 显示进度条
        for i in tqdm(range(len(embeddings)), desc="Indexing articles", unit="article"):
            self.add_embeddings([embeddings[i]], [texts[i]], [sources[i]])
