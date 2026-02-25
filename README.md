# Veda Bot RAG

An AI-powered Retrieval-Augmented Generation (RAG) system for Ayurvedic medical knowledge, built to process PDF texts and provide intelligent diagnostic conversations.

## Overview

Veda Bot RAG is a modular Python pipeline that transforms raw Ayurvedic PDF books into structured, searchable knowledge. It enables semantic search and multi-turn diagnostic conversations through an AI chatbot interface.

### Key Features

- **PDF Processing Pipeline**: Extract, clean, and structure text from medical PDFs
- **Hybrid Retrieval**: Dense FAISS retrieval + lexical BM25 fusion, then cross-encoder reranking
- **Metadata Extraction**: Automatic detection of Ayurvedic concepts (doshas, srotas, treatment types)
- **Multi-turn Conversation**: Context-aware diagnostic dialogue system
- **Modular Architecture**: Each processing stage is independent and configurable

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  EXTRACTION  │───▶│   CLEANING   │───▶│   STRUCTURE  │───▶│   CHUNKING   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│   RAG CHAT   │◀───│   RETRIEVAL  │◀───│  VECTOR DB   │◀──────┘
└──────────────┘    └──────────────┘    └──────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AaryaaSastry/veda-bot-rag.git
   cd veda-bot-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install pymupdf tiktoken sentence-transformers faiss-cpu google-genai numpy
   ```

## Usage

### Stage 1: Process PDFs

Extract text from PDFs, clean, and create structured chunks:

```bash
cd src
python main.py
```

**Output:**
- `data/cleaned_text/` - Cleaned text files
- `data/chunks/` - Structured JSON chunks with metadata

### Stage 2: Build Embeddings

Generate vector embeddings and create FAISS index:

```bash
cd src
python build_embeddings.py
```

**Output:**
- `data/embeddings/faiss.index` - Vector index
- `data/embeddings/metadata.json` - Chunk metadata

Alternative single command:
```bash
python process_new_pdfs.py
```
This runs PDF processing + embedding/index build end-to-end.

### Stage 3: Run Chatbot

Start the interactive RAG chatbot:

```bash
cd src
python run_rag.py
```

**Example Interaction:**
```
You: I have a cough

AI: To provide an accurate Ayurvedic assessment, could you share your age and gender?

You: 35, male

AI: What kind of cough is it (dry or productive), and when is it worst?

You: Dry cough, worse at night

AI: [After minimum configured gathering questions, diagnosis is generated and verified before final output]
```

## Project Structure

```
veda-bot-rag/
├── data/
│   ├── raw_pdfs/              # Input PDF files
│   ├── cleaned_text/          # Processed text output
│   ├── chunks/                # JSON chunks with metadata
│   └── embeddings/            # FAISS index and metadata
│
├── src/
│   ├── main.py                # Main pipeline entry point
│   ├── build_embeddings.py    # Embedding generation
│   ├── run_rag.py             # RAG chatbot interface
│   ├── config.py              # Configuration parameters
│   │
│   ├── extraction/            # PDF extraction modules
│   ├── cleaning/              # Text cleaning modules
│   ├── structure/             # Chapter parsing & metadata
│   ├── chunking/              # Token-based chunking
│   ├── embedding/             # Vector embedding & indexing
│   └── rag/                   # RAG pipeline components
│       ├── rag_pipeline.py    # Main RAG orchestrator
│       ├── retriever.py       # Vector search
│       ├── generator.py       # LLM generation
│       ├── memory.py          # Conversation memory
│       └── validator.py       # Response validation
│
└── documentations/
    ├── SRS.md                 # Software Requirements Spec
    ├── CODEBASE_DOCUMENTATION.md
    └── HOW_TO_RUN.md
```

## Configuration

Key parameters in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `K_RETRIEVAL` | 20 | Initial retrieval count |
| `K_RERANK` | 8 | Final chunks after reranking |
| `TEMPERATURE` | 0.1 | LLM generation temperature |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `USE_HYBRID_RETRIEVAL` | `True` | Enables dense + BM25 fusion retrieval |
| `MIN_GATHERING_QUESTIONS` | `15` | Minimum user turns before first diagnosis attempt |
| `EXTRA_GATHERING_QUESTIONS_IF_UNCERTAIN` | `5` | Extra turns if diagnosis verification fails |

## API Key Setup

The chatbot requires a Google Gemini API key:

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set it in your environment (recommended):
   - PowerShell: `setx GEMINI_API_KEY "your_key_here"` (new terminal required)
3. Update code to read from env if needed in [`src/run_rag.py`](src/run_rag.py)

## Evaluation

Run:
```bash
python src/benchmarks/evaluate_rag.py
```

Evaluation supports:
- Gold retrieval metrics (`Recall@k`, `MRR`, `nDCG@k`) when `data/evaluation/retrieval_gold.json` exists
- Fallback keyword-based retrieval scoring otherwise

## Supported Ayurvedic Concepts

The system automatically extracts metadata for:

- **Doshas**: Vata, Pitta, Kapha
- **Srotas** (Body Channels): Pranavaha, Annavaha, Rasavaha, etc.
- **Treatment Types**: Shodhana (purification), Shamana (palliative)
- **Formulations**: Churna (powder), Vati (tablet), Ghrita (ghee), Asava (fermented)
- **Care Levels**: PHC (Primary Health Center), CHC (Community Health Center)

## Tech Stack

- **PDF Processing**: PyMuPDF
- **Tokenization**: Tiktoken (GPT-4 tokenizer)
- **Embeddings**: SentenceTransformers (BAAI/bge-small-en)
- **Vector Search**: FAISS
- **Reranking**: Cross-Encoder (MS MARCO MiniLM)
- **LLM**: Google Gemini (gemma-3-27b-it)

## Documentation

- [How to Run](documentations/HOW_TO_RUN.md) - Detailed setup and usage instructions
- [Codebase Documentation](documentations/CODEBASE_DOCUMENTATION.md) - Technical module documentation
- [Software Requirements Specification](documentations/SRS.md) - Project requirements

## License

This project is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
