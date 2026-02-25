# rag/metadata_filtered_retriever.py

import numpy as np
import faiss
import json


class MetadataFilteredRetriever:
    """
    Retriever that filters results by body system metadata.
    Useful when chunks have been enriched with primary_system field.
    """

    def __init__(self, embedding_model, faiss_path, metadata_path):
        self.embedding_model = embedding_model
        self.index = faiss.read_index(faiss_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query_text, query_system=None, top_k=5):
        """
        Retrieve chunks, optionally filtered by body system.
        
        Args:
            query_text: The search query
            query_system: Optional body system to filter by
            top_k: Number of results to return
        
        Returns:
            List of metadata dicts for matching chunks
        """
        # Step 1: Filter metadata indices by system if specified
        if query_system and query_system != "other":
            filtered_indices = [
                i for i, m in enumerate(self.metadata)
                if m.get("primary_system") == query_system
            ]
        else:
            filtered_indices = list(range(len(self.metadata)))

        if not filtered_indices:
            print("No metadata match, falling back to full search.")
            filtered_indices = list(range(len(self.metadata)))

        # Step 2: Encode query
        query_vector = self.embedding_model.encode([query_text])
        query_vector = np.array(query_vector).astype("float32")

        # Step 3: Search FULL index first (get more results for filtering)
        search_k = min(top_k * 10, len(self.metadata))
        distances, indices = self.index.search(query_vector, search_k)

        # Step 4: Filter retrieved results by metadata
        results = []
        for idx in indices[0]:
            if idx in filtered_indices:
                results.append(self.metadata[idx])

            if len(results) >= top_k:
                break

        return results

    def retrieve_with_fallback(self, query_text, query_system=None, top_k=5):
        """
        Retrieve with fallback: if filtered results are insufficient,
        fall back to unfiltered search.
        
        Args:
            query_text: The search query
            query_system: Optional body system to filter by
            top_k: Number of results to return
        
        Returns:
            List of metadata dicts for matching chunks
        """
        # Try filtered search first
        if query_system and query_system != "other":
            results = self.retrieve(query_text, query_system, top_k)
            if len(results) >= top_k:
                return results
        
        # Fallback to unfiltered search
        return self.retrieve(query_text, None, top_k)
