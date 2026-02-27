# How to Run the Book Corpus Pipeline

This document provides step-by-step instructions for running the Book Corpus Pipeline.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Running the Pipeline](#3-running-the-pipeline)
4. [Running Individual Modules](#4-running-individual-modules)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Prerequisites

### System Requirements
- **Operating System**: Windows 11 (or Windows 10)
- **Python**: Version 3.8 or higher
- **Disk Space**: ~500MB for dependencies + space for PDFs and generated data

### Required Software
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

---

## 2. Installation

### Step 2.1: Open Terminal in Project Directory

Open PowerShell or Command Prompt in the project folder:
```
C:\Users\aarri\Desktop\book_corpus_pipeline
```

### Step 2.2: Activate Virtual Environment

The project already has a virtual environment (`.venv`). Activate it:

**PowerShell:**
```powershell
.venv\Scripts\Activate
```

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` appear before your prompt, indicating the virtual environment is active.

### Step 2.3: Install Dependencies

Install all required packages:

```bash
pip install pymupdf tiktoken sentence-transformers faiss-cpu google-genai numpy
```

**Package Details:**

| Package | Version | Purpose |
|---------|---------|---------|
| `pymupdf` | latest | PDF text extraction |
| `tiktoken` | latest | Token counting for chunking |
| `sentence-transformers` | latest | Embedding generation |
| `faiss-cpu` | latest | Vector similarity search |
| `google-genai` | latest | Gemini LLM API |
| `numpy` | latest | Array operations |

### Step 2.4: Verify Installation

Test that all packages are installed correctly:

```bash
python -c "import pymupdf; import tiktoken; import sentence_transformers; import faiss; import google.genai; print('All packages installed successfully!')"
```

---

## 3. Running the Pipeline

The pipeline has **three main stages** that must be run in order.

### Stage 1: Process PDFs (Extract → Clean → Chunk)

**What it does:**
- Extracts text from PDFs in `data/raw_pdfs/`
- Cleans text (removes headers, footers, page numbers, TOC)
- Parses chapters
- Creates structured chunks with metadata
- Saves output to `data/chunks/`

**Command:**
```bash
cd src
python main.py
```

**Expected Output:**
```
--- Processing: ayurvedic_treatment_file1.pdf ---
Step 1: Extracting pages...
Step 2: Cleaning headers, footers and page numbers...
Step 3: Removing front matter...
Step 4: Parsing chapters...
   Found 39 chapters.
Step 5: Chunking and adding metadata...
Step 6: Saving 1500 structured chunks to JSON...
 WORKFLOW COMPLETE: ayurvedic_treatment_file1_chunks.json

--- Processing: Ayurvedic-Home-Remedies-English.pdf ---
...
```

**Output Files Created:**
- `data/cleaned_text/{filename}_final_clean.txt` - Cleaned text
- `data/chunks/{filename}_chunks.json` - Structured chunks

---

### Stage 2: Build Embeddings

**What it does:**
- Loads all chunk JSON files from `data/chunks/`
- Generates vector embeddings using `BAAI/bge-small-en` model
- Creates FAISS index for similarity search
- Saves index and metadata

**Command:**
```bash
cd src
python build_embeddings.py
```

**Expected Output:**
```
Loading embedding model: BAAI/bge-small-en
Batches: 100%|██████████| 47/47 [00:15<00:00,  3.12it/s]
Embeddings and FAISS index saved successfully.
```

**Output Files Created:**
- `data/embeddings/faiss.index` - FAISS vector index
- `data/embeddings/metadata.json` - Chunk metadata mapping

**Note:** This step may take 2-5 minutes depending on your CPU. The embedding model will be downloaded automatically on first run (~50MB).

**Alternative (single command):**
```bash
cd src
python process_new_pdfs.py
```
This runs Stage 1 and Stage 2 together.

---

### Stage 3: Run RAG Chatbot

**What it does:**
- Loads FAISS index and metadata
- Uses hybrid retrieval (dense + BM25 fusion) when enabled in `src/config.py`
- Starts interactive chat session
- Uses a multi-turn diagnosis flow:
  - asks at least `MIN_GATHERING_QUESTIONS` user turns (default 15)
  - generates diagnosis and runs verification (`YES`/`NO` check)
  - if verification fails, asks `EXTRA_GATHERING_QUESTIONS_IF_UNCERTAIN` more (default 5)
<<<<<<< HEAD
- **Safety Flow**: When high-risk symptoms are detected, asks for more details instead of terminating
- **Remedies Flow**: After diagnosis, asks if user wants remedies:
  - If YES: Provides home remedies, do's/don'ts, lifestyle recommendations
  - If NO: Ends with "Thank you for consulting with us. Take care and hope I helped! Goodbye."
=======
>>>>>>> f0d38fdb403f82dc7552cdf2560d1b0e92fb5ae3

**Command:**
```bash
cd src
python run_rag.py
```

**Or from project root:**
```bash
python src/run_rag.py
```

**Expected Output:**
```
Loading embedding model: BAAI/bge-small-en
Retriever mode: hybrid (dense + BM25 fusion)
--- Ayurvedic AI Chatbot (type 'exit' to quit) ---

You: I have a cough

AI: To provide an accurate Ayurvedic assessment, could you share your age and gender?

You: 35, male

AI: What kind of cough is it (dry or productive), and when is it worst?

You: Dry cough, worse at night

AI: ...
```

**To Exit:** Type `exit` and press Enter.

---

## 4. Running Individual Modules

You can also run individual modules for testing or debugging.

### 4.1 Analyze Corpus Quality

**Purpose:** Analyze the chunked corpus for quality metrics.

**Command:**
```bash
cd src
python analyze_corpus.py
```

**Expected Output:**
```
Total words: 125000
Total chunks: 1500
Avg chunk length: 83.33 words
Weird char ratio: 0.0012
Duplicate ratio: 0.0000
```

---

### 4.2 Test PDF Extraction Only

Create a test script:

```python
# test_extraction.py
import sys
sys.path.append('src')

from extraction.pdf_extractor import extract_pages_from_pdf

pages = extract_pages_from_pdf("data/raw_pdfs/ayurvedic_treatment_file1.pdf")
print(f"Extracted {len(pages)} pages")
print(f"First page preview:\n{pages[0][:500]}...")
```

Run:
```bash
python test_extraction.py
```

---

### 4.3 Test Cleaning Functions

Create a test script:

```python
# test_cleaning.py
import sys
sys.path.append('src')

from extraction.pdf_extractor import extract_pages_from_pdf
from cleaning.header_footer import remove_headers_footers
from cleaning.page_numbers import remove_page_numbers

# Extract
pages = extract_pages_from_pdf("data/raw_pdfs/ayurvedic_treatment_file1.pdf")

# Clean
pages = remove_headers_footers(pages)
pages = remove_page_numbers(pages)

print(f"Cleaned {len(pages)} pages")
```

Run:
```bash
python test_cleaning.py
```

---

### 4.4 Test Embedding Generation

Create a test script:

```python
# test_embedding.py
import sys
sys.path.append('src')

from embedding.embed import TextEmbedder

embedder = TextEmbedder("BAAI/bge-small-en")
texts = ["This is a test sentence.", "Another test sentence."]
embeddings = embedder.embed_texts(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

Run:
```bash
python test_embedding.py
```

---

### 4.5 Test Vector Search

Create a test script:

```python
# test_search.py
import sys
sys.path.append('src')

from rag.retriever import Retriever

retriever = Retriever("data/embeddings")
results = retriever.retrieve("What is the treatment for cough?", k=5)

for i, result in enumerate(results):
    print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
    print(f"Source: {result['source']}")
    print(f"Text: {result['text'][:200]}...")
```

Run:
```bash
python test_search.py
```

---

### 4.6 Run Evaluation

**Purpose:** Evaluate retrieval/safety and sample response quality.

**Command:**
```bash
python src/benchmarks/evaluate_rag.py
```

**Notes:**
- If `data/evaluation/retrieval_gold.json` exists, retrieval uses gold metrics (`Recall@k`, `MRR`, `nDCG@k`).
- If no gold file exists, evaluation falls back to keyword-based retrieval scoring.

---

## 5. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pymupdf'`

**Cause:** Dependencies not installed.

**Solution:**
```bash
pip install pymupdf tiktoken sentence-transformers faiss-cpu google-genai numpy
```

---

### Error: `FileNotFoundError: PDF file not found`

**Cause:** No PDF files in `data/raw_pdfs/` directory.

**Solution:** Add PDF files to `data/raw_pdfs/` folder.

---

### Error: `RuntimeError: could not open data/embeddings\faiss.index`

**Cause:** Embeddings not built yet.

**Solution:** Run `python build_embeddings.py` first.

---

### Error: `can't open file 'src/src/run_rag.py'`

**Cause:** Running from wrong directory or using wrong path.

**Solution:**
- If you're in `src/` directory: `python run_rag.py`
- If you're in project root: `python src/run_rag.py`

---

### Error: `ImportError: cannot import name 'config'`

**Cause:** Running script from wrong directory.

**Solution:** Make sure you're in the `src/` directory when running, or add `src` to Python path:
```bash
cd src
python main.py
```

---

### Warning: `You are sending unauthenticated requests to the HF Hub`

**Cause:** No Hugging Face token set.

**Solution (Optional):** This is just a warning. To remove it:
1. Create account at https://huggingface.co/
2. Get token from Settings → Access Tokens
3. Set environment variable:
   ```bash
   set HF_TOKEN=your_token_here
   ```

---

### Error: `google.genai.errors.APIError: API key not valid`

**Cause:** Invalid or expired Gemini API key.

**Solution:**
1. Get new API key from https://aistudio.google.com/apikey
2. Update `API_KEY` in `src/run_rag.py` (line 8)

---

### Error: `429 RESOURCE_EXHAUSTED` (Gemini quota exceeded)

**Cause:** Input token quota exceeded for the selected model.

**Solution:**
1. Wait for retry window and try again.
2. Use a lower-cost model in `src/rag/generator.py` (for example Gemini Flash).
3. Keep `K_RERANK` moderate and avoid very long sessions without restarting.

---

### Slow Embedding Generation

**Cause:** Running on CPU without optimizations.

**Solutions:**
1. Use GPU: Install `faiss-gpu` instead of `faiss-cpu`
2. Reduce batch size in `embed.py`
3. Use smaller embedding model

---

## Quick Reference Card

```bash
# === SETUP ===
.venv\Scripts\Activate                                    # Activate venv
pip install pymupdf tiktoken sentence-transformers faiss-cpu google-genai numpy

# === RUN PIPELINE ===
cd src && python main.py                                  # Stage 1-4: Process PDFs
cd src && python build_embeddings.py                      # Stage 5: Build embeddings
cd src && python process_new_pdfs.py                      # Stage 1-5 in one command
cd src && python run_rag.py                               # Stage 6-7: Run chatbot

# === UTILITIES ===
cd src && python analyze_corpus.py                        # Analyze corpus quality
python src/benchmarks/evaluate_rag.py                     # Run evaluation suite

# === FROM PROJECT ROOT ===
python src/main.py                                        # Process PDFs
python src/build_embeddings.py                            # Build embeddings
python src/run_rag.py                                     # Run chatbot
python src/analyze_corpus.py                              # Analyze corpus
```

---

## Directory Structure After Running

```
book_corpus_pipeline/
├── data/
│   ├── raw_pdfs/                    # INPUT: Your PDF files
│   │   ├── ayurvedic_treatment_file1.pdf
│   │   └── Ayurvedic-Home-Remedies-English.pdf
│   │
│   ├── cleaned_text/                # OUTPUT: Cleaned text files
│   │   ├── ayurvedic_treatment_file1_final_clean.txt
│   │   └── Ayurvedic-Home-Remedies-English_final_clean.txt
│   │
│   ├── chunks/                      # OUTPUT: JSON chunks
│   │   ├── ayurvedic_treatment_file1_chunks.json
│   │   └── Ayurvedic-Home-Remedies-English_chunks.json
│   │
│   └── embeddings/                  # OUTPUT: Vector index
│       ├── faiss.index              # FAISS binary index
│       └── metadata.json            # Chunk metadata
│
├── src/                             # SOURCE CODE
│   ├── main.py                      # Main pipeline entry
│   ├── build_embeddings.py          # Embedding builder
│   ├── run_rag.py                   # RAG chatbot
│   └── analyze_corpus.py            # Corpus analyzer
│
└── documentations/
    ├── SRS.md
    ├── CODEBASE_DOCUMENTATION.md
    └── HOW_TO_RUN.md                # This file
```

---

*Last updated: February 2026*
