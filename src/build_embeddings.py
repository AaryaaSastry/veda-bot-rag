import os
import json
from embedding.embed import TextEmbedder
from embedding.index_builder import VectorIndex

CHUNKS_PATH = "data/chunks"
EMBEDDINGS_PATH = "data/embeddings"

def load_chunks():
    all_texts = []
    metadata = []

    for file in os.listdir(CHUNKS_PATH):
        if file.endswith(".json"):
            with open(os.path.join(CHUNKS_PATH, file), "r", encoding="utf-8") as f:
                chunks = json.load(f)

                for chunk in chunks:
                    # chunk is a dict with fields like 'text', 'source', 'chapter', etc.
                    all_texts.append(chunk["text"])
                    metadata.append(chunk)

    return all_texts, metadata


if __name__ == "__main__":

    texts, metadata = load_chunks()

    embedder = TextEmbedder("BAAI/bge-small-en")

    embeddings = embedder.embed_texts(texts)

    dimension = embeddings.shape[1]

    index = VectorIndex(dimension)
    index.add_embeddings(embeddings, metadata)

    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

    index.save(
        os.path.join(EMBEDDINGS_PATH, "faiss.index"),
        os.path.join(EMBEDDINGS_PATH, "metadata.json")
    )

    print("Embeddings and FAISS index saved successfully.")