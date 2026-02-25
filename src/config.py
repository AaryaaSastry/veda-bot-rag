RAW_DATA_PATH = "data/raw_pdfs"
CLEANED_PATH = "data/cleaned_text"
CHUNKS_PATH = "data/chunks"

CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50

# --- RAG Settings ---
K_RETRIEVAL = 20
K_RERANK = 8
K_DEFAULT = 12
TEMPERATURE = 0.1
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HEADER_FOOTER_THRESHOLD = 0.7
