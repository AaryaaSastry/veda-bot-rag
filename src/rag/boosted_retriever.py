import numpy as np
import faiss


class BoostedRetriever:

    def __init__(self, embedding_model, faiss_path, metadata_path, booster):
        self.embedding_model = embedding_model
        self.index = faiss.read_index(faiss_path)
        self.booster = booster

        import json
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def retrieve(self, user_input, top_k=5):

        # Step 1: Build boosted vector
        query_vector = self.booster.build_weighted_vector(user_input)
        query_vector = np.array([query_vector]).astype("float32")

        # Step 2: FAISS search (broader search first)
        distances, indices = self.index.search(query_vector, top_k * 5)

        candidates = []
        query_terms = set(user_input.lower().split())

        for idx in indices[0]:
            chunk = self.metadata[idx]
            chunk_terms = set(chunk["text"].lower().split())

            # Intersection score
            overlap = len(query_terms.intersection(chunk_terms))

            candidates.append((overlap, chunk))

        # Step 3: Sort by intersection overlap
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Return top_k
        return [c[1] for c in candidates[:top_k]]