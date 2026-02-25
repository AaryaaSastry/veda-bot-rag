from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-small-en"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Convert list of text chunks into normalized embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return np.array(embeddings)

    def embed_query(self, query):
        """
        Convert a single query into embedding.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return np.array(embedding)