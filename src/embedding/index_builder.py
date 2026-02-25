import faiss
import numpy as np
import json
import os


class VectorIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity (normalized)

        self.metadata = []  # stores mapping of vector → text

    def add_embeddings(self, embeddings, metadata_list):
        """
        Add embeddings and corresponding metadata.
        """
        self.index.add(np.array(embeddings))
        self.metadata.extend(metadata_list)

    def save(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_vector, k=5):
        scores, indices = self.index.search(query_vector, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "score": float(score),
                "text": self.metadata[idx]["text"],
                "source": self.metadata[idx]["source"]
            })

        return results