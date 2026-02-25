import os
import config
from embedding.embed import TextEmbedder
from embedding.index_builder import VectorIndex
#embedding.index_builder

class Retriever:

    def __init__(self, vector_db_dir):
        self.embedder = TextEmbedder("BAAI/bge-small-en")
        self.store = VectorIndex(dimension=384)
        
        # Load from the directory provided
        self.store.load(
            os.path.join(vector_db_dir, "faiss.index"),
            os.path.join(vector_db_dir, "metadata.json")
        )

    def retrieve(self, query, k=config.K_DEFAULT, source_filter=None):
        # BGE models require a prefix for queries to get better results
        instruction = "Represent this sentence for searching relevant passages: "
        prefixed_query = instruction + query
        query_vector = self.embedder.embed_query(prefixed_query)
        # VectorIndex.search takes the query vector and k
        # We might need to over-retrieve if we filter afterwards to maintain 'k' results
        # but for simplicity, let's retrieve more and then filter
        search_k = k * 10 if source_filter else k
        results = self.store.search(query_vector, k=search_k)

        if source_filter:
            results = [r for r in results if r['source'] == source_filter]
            return results[:k]
            
        return results[:k]