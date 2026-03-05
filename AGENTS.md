# AGENTS.md — Veda Bot RAG

## Build & Run
- **Install**: `pip install pymupdf tiktoken sentence-transformers faiss-cpu google-genai numpy` (Python 3.8+, venv in `.venv/`)
- **Process PDFs**: `python src/main.py` (or `python src/process_new_pdfs.py` for end-to-end)
- **Build embeddings**: `python src/build_embeddings.py`
- **Run chatbot**: `python src/run_rag.py`
- **Evaluate**: `python src/benchmarks/evaluate_rag.py`
- No test framework is configured; there are no unit tests.

## Architecture
- **Pipeline**: PDF extraction → cleaning → chapter parsing → chunking → embedding (FAISS) → RAG chatbot
- **Entrypoints**: `src/main.py` (PDF processing), `src/build_embeddings.py` (indexing), `src/run_rag.py` (chatbot)
- **Key modules**: `src/extraction/`, `src/cleaning/`, `src/structure/`, `src/chunking/`, `src/embedding/`, `src/rag/`
- **RAG core**: `rag/rag_pipeline.py` (orchestrator), `rag/retriever.py` + `rag/hybrid_fusion_retriever.py` (FAISS+BM25), `rag/generator.py` (Gemini LLM), `rag/memory.py` (conversation state), `rag/validator.py`
- **Config**: `src/config.py` — all tunable params (chunk size, retrieval K, hybrid weights, diagnosis thresholds)
- **Data**: `data/raw_pdfs/` → `data/cleaned_text/` → `data/chunks/` (JSON) → `data/embeddings/` (FAISS index + metadata.json)
- **LLM**: Google Gemini (`gemma-3-27b-it`); API key set via `GEMINI_API_KEY` env var or hardcoded in `run_rag.py`

## Code Style
- Pure Python, no frameworks; imports use relative package paths from `src/` (e.g., `from rag.retriever import Retriever`)
- Run scripts from `src/` directory (`cd src && python main.py`)
- Functions use snake_case, classes use PascalCase; docstrings in `"""triple quotes"""`
- Config constants are UPPER_SNAKE_CASE in `src/config.py`; access via `import config` then `config.PARAM`
- Error handling: broad `try/except` with print-based logging; `src/utils/logger.py` exists but logging is mostly print statements
- No type annotations used; no linter or formatter configured
