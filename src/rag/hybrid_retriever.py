# rag/hybrid_retriever.py

import numpy as np
import faiss


class HybridRetriever:

    def __init__(self, embedding_model, faiss_index_path, metadata_path):
        self.embedding_model = embedding_model
        self.index = faiss.read_index(faiss_index_path)

        import json
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def retrieve(self, weighted_query, top_k=5):
        query_vector = self.embedding_model.encode([weighted_query])
        query_vector = np.array(query_vector).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results