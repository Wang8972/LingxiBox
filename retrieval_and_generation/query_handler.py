import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

class QueryHandler:
    def __init__(self, indexer, model_name="all-MiniLM-L6-v2"):
        self.indexer = indexer
        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=1):
        """ 在索引中搜索包含关键字的最佳匹配文章 """
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().numpy()
        _, indices = self.indexer.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        for i in indices[0]:
            if i != -1:
                text, source = self.indexer.metadata[i]
                results.append((text, source))
        return results
