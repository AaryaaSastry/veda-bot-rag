import os
import json
from embedding.embed import TextEmbedder
from embedding.index_builder import VectorIndex

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")

def load_chunks():
    all_texts = []
    metadata = []

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks directory not found: {CHUNKS_PATH}")

    for file in os.listdir(CHUNKS_PATH):
        if file.endswith(".json"):
            with open(os.path.join(CHUNKS_PATH, file), "r", encoding="utf-8") as f:
                chunks = json.load(f)

                for chunk in chunks:
                    # chunk is a dict with fields like 'text', 'source', 'chapter', etc.
                    all_texts.append(chunk["text"])
                    metadata.append(chunk)

    return all_texts, metadata


def build_embeddings():
    texts, metadata = load_chunks()

    if not texts:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}. Run main.py first to generate chunk files.")

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


if __name__ == "__main__":
    build_embeddings()
