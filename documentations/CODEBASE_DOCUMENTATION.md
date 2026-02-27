# Book Corpus Pipeline - Comprehensive Technical Documentation

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Module-by-Module Documentation](#3-module-by-module-documentation)
4. [How to Run Each Module](#4-how-to-run-each-module)
5. [Data Flow](#5-data-flow)
6. [Configuration](#6-configuration)
7. [Dependencies](#7-dependencies)

---

## 1. System Overview

The **Book Corpus Pipeline** is a modular Python system designed to process raw PDF books into structured, searchable knowledge for Retrieval-Augmented Generation (RAG). The pipeline is specifically tailored for Ayurvedic medical texts but can be adapted for other domains.

### Purpose
- Extract text from PDF documents
- Clean and normalize extracted text
- Parse document structure (chapters, sections)
- Chunk text into semantically meaningful segments
- Generate vector embeddings for semantic search
- Provide an interactive RAG-based chatbot for querying the knowledge base

### Key Features
- **Modular Architecture**: Each processing stage is independent and can be run separately
- **Metadata Extraction**: Automatic detection of Ayurvedic concepts (doshas, srotas, treatment types)
- **Semantic Search**: FAISS-based vector similarity search with cross-encoder reranking
- **Multi-turn Conversation**: Context-aware diagnostic dialogue system

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BOOK CORPUS PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   STAGE 1    │───▶│   STAGE 2    │───▶│   STAGE 3    │───▶│   STAGE 4    │  │
│  │  EXTRACTION  │    │   CLEANING   │    │  STRUCTURE   │    │   CHUNKING   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │          │
│         ▼                   ▼                   ▼                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Raw Pages   │    │ Cleaned Text │    │  Chapters    │    │   Chunks     │  │
│  │  (List[str]) │    │    (str)     │    │ (List[Dict]) │    │(List[Dict])  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                     │          │
│                                                                     ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   STAGE 7    │◀───│   STAGE 6    │◀───│   STAGE 5    │◀───│  EMBEDDING   │  │
│  │  RAG CHAT    │    │   RETRIEVAL  │    │  VECTOR DB   │    │  GENERATION  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │          │
│         ▼                   ▼                   ▼                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Response    │    │ Ranked Docs  │    │ FAISS Index  │    │  Embeddings  │  │
│  │   (Stream)   │    │ (List[Dict]) │    │   + Metadata │    │  (np.array)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module-by-Module Documentation

### 3.1 Configuration Module

**File**: [`src/config.py`](src/config.py)

**Purpose**: Central configuration for all pipeline parameters.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `RAW_DATA_PATH` | `"data/raw_pdfs"` | Input directory for PDF files |
| `CLEANED_PATH` | `"data/cleaned_text"` | Output directory for cleaned text |
| `CHUNKS_PATH` | `"data/chunks"` | Output directory for JSON chunks |
| `CHUNK_SIZE` | `512` | Token count per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `K_RETRIEVAL` | `20` | Initial retrieval count |
| `K_RERANK` | `8` | Final chunks after reranking |
| `K_DEFAULT` | `12` | Default retrieval count |
| `TEMPERATURE` | `0.1` | LLM generation temperature |
| `RERANKER_MODEL` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model for reranking |
| `HEADER_FOOTER_THRESHOLD` | `0.7` | Frequency threshold for header/footer detection |

---

### 3.2 Extraction Module

**File**: [`src/extraction/pdf_extractor.py`](src/extraction/pdf_extractor.py)

**Function**: `extract_pages_from_pdf(pdf_path: str) -> List[str]`

**Purpose**: Extracts raw text from PDF files, preserving page boundaries.

**Process**:
1. Opens PDF using `pymupdf` library
2. Iterates through each page
3. Extracts text content from each page
4. Returns a list where each element is the text of one page

**Input**: Path to a PDF file

**Output**: List of strings, one per page

**Example**:
```python
pages = extract_pages_from_pdf("data/raw_pdfs/book.pdf")
# Returns: ["Page 1 content...", "Page 2 content...", ...]
```

**Dependencies**: `pymupdf`

---

### 3.3 Cleaning Modules

#### 3.3.1 Header/Footer Removal

**File**: [`src/cleaning/header_footer.py`](src/cleaning/header_footer.py)

**Function**: `remove_headers_footers(pages: List[str]) -> List[str]`

**Purpose**: Identifies and removes recurring headers and footers using frequency analysis.

**Algorithm**:
1. Collects first and last lines from all pages
2. Counts frequency of each line
3. Lines appearing in >30% of pages are flagged as potential headers/footers
4. Removes flagged lines from top 3 and bottom 3 positions of each page
5. Also removes a static header pattern: `"AYURVEDIC STANDARD TREATMENT GUIDELINES"`

**Input**: List of page strings

**Output**: List of cleaned page strings

---

#### 3.3.2 Page Number Removal

**File**: [`src/cleaning/page_numbers.py`](src/cleaning/page_numbers.py)

**Function**: `remove_page_numbers(pages: List[str]) -> List[str]`

**Purpose**: Removes standalone page numbers (Roman or Arabic numerals) from page extremities.

**Algorithm**:
1. Compiles regex pattern: `^(?:[ivxlc]+|\d+)$` (matches Roman and Arabic numerals)
2. For each page, checks first 3 and last 3 lines
3. Removes lines that match the pattern

**Input**: List of page strings

**Output**: List of cleaned page strings

---

#### 3.3.3 Table of Contents Removal

**File**: [`src/cleaning/toc_removal.py`](src/cleaning/toc_removal.py)

**Function**: `remove_front_matter(text: str) -> str`

**Purpose**: Removes front matter (TOC, Index, Foreword) by finding the start of actual content.

**Algorithm**:
1. Searches for specific content markers:
   - `"Pranavaha Srotas Roga"`
   - `"Brief Introduction of the disease: Kasa"`
   - `"Kasa (Cough)"`
2. Looks for marker appearing after `"INTRODUCTION"` section
3. Returns text from the marker position onwards

**Input**: Full text as a single string

**Output**: Text with front matter removed

---

#### 3.3.4 Hyphenation Fix

**File**: [`src/cleaning/hyphenation.py`](src/cleaning/hyphenation.py)

**Function**: `fix_hyphenation(text: str) -> str`

**Purpose**: Fixes words broken across lines with hyphens.

**Algorithm**: Uses regex `r'-\n(\w+)'` to find hyphen followed by newline and joins the word.

**Example**:
```
"comput-\ner" → "computer"
```

---

#### 3.3.5 Line Merger

**File**: [`src/cleaning/line_merger.py`](src/cleaning/line_merger.py)

**Function**: `merge_lines(text: str) -> str`

**Purpose**: Merges single line breaks into spaces (preserves paragraph breaks).

**Algorithm**: Replaces single newlines (not followed or preceded by another newline) with spaces.

---

#### 3.3.6 Normalization

**File**: [`src/cleaning/normalization.py`](src/cleaning/normalization.py)

**Function**: `normalize(text: str) -> str`

**Purpose**: Normalizes Unicode and whitespace.

**Algorithm**:
1. Applies NFKC Unicode normalization
2. Collapses multiple spaces/tabs to single space
3. Collapses 3+ newlines to 2 newlines
4. Strips leading/trailing whitespace

---

### 3.4 Structure Modules

#### 3.4.1 Chapter Parser

**File**: [`src/structure/chapter_parser.py`](src/structure/chapter_parser.py)

**Function**: `parse_chapters(text: str) -> List[Dict]`

**Purpose**: Parses full text into chapters based on known chapter titles.

**Algorithm**:
1. Defines a list of 39 known chapter markers (Ayurvedic conditions)
2. Creates regex pattern to find these markers at line starts
3. Splits text at each marker
4. Returns list of dictionaries with `title` and `content` keys

**Chapter Markers Include**:
- Kasa, Tamaka Swasa, Amlapitta, Jalodara, Amavata, Jwara, Pandu
- Ekakushtha, Kamala, Hypothyroidism, Madhumeha, Sthoulya, Arsha
- And 26 more Ayurvedic conditions

**Output Format**:
```python
[
    {"title": "Kasa", "content": "Chapter content here..."},
    {"title": "Tamaka Swasa", "content": "Chapter content here..."},
    ...
]
```

---

#### 3.4.2 Metadata Extractor

**File**: [`src/structure/metadata_extractor.py`](src/structure/metadata_extractor.py)

**Purpose**: Extracts Ayurvedic metadata from text using keyword matching.

**Functions**:

| Function | Detects | Keywords |
|----------|---------|----------|
| `detect_dosha(text)` | Vata, Pitta, Kapha | "vata", "pitta", "kapha" |
| `detect_category(text)` | Disease, Herb, Theory | "roga", "herb", "principle" |
| `detect_disease_type(text)` | Ano-rectal, Psychiatric | "fistula", "insomnia", etc. |
| `detect_srotas(text)` | Purishavaha, Manovaha | "stool", "mind", etc. |
| `detect_treatment_type(text)` | Shodhana, Shamana | "vamana", "powder", etc. |
| `detect_level_of_care(text)` | PHC, CHC | "phc", "chc" |
| `detect_formulation_type(text)` | Churna, Vati, Ghrita | "powder", "tablet", "ghee" |
| `extract_topic(text)` | First sentence | First 120 chars of first sentence |

---

#### 3.4.3 Schema Definition

**File**: [`src/structure/schema.py`](src/structure/schema.py)

**Class**: `KnowledgeChunk`

**Purpose**: Data class representing a chunk with metadata.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The chunk text content |
| `source` | `str` | Source PDF filename |
| `chapter` | `Optional[str]` | Chapter title |
| `topic` | `Optional[str]` | Extracted topic |
| `dosha` | `Optional[str]` | Detected dosha |
| `category` | `Optional[str]` | Content category |
| `disease_type` | `Optional[str]` | Type of disease |
| `srotas` | `Optional[str]` | Affected body channel |
| `treatment_type` | `Optional[str]` | Type of treatment |
| `level_of_care` | `Optional[str]` | PHC or CHC level |
| `formulation_type` | `Optional[str]` | Type of formulation |

**Methods**:
- `to_dict()`: Converts to dictionary for JSON serialization

---

### 3.5 Chunking Module

**File**: [`src/chunking/chunker.py`](src/chunking/chunker.py)

**Class**: `TokenChunker`

**Purpose**: Splits text into overlapping token-based chunks.

**Initialization**:
```python
chunker = TokenChunker(model_name="gpt-4", chunk_size=700, overlap=100)
```

**Algorithm**:
1. Encodes text using `tiktoken` (GPT-4 tokenizer)
2. Slides window of `chunk_size` tokens with `overlap` tokens overlap
3. Decodes each token window back to text

**Method**: `chunk_text(text: str) -> List[str]`

**Function**: `create_structured_chunk(text, source, chapter) -> KnowledgeChunk`

Wraps a text chunk with all extracted metadata.

---

### 3.6 Embedding Modules

#### 3.6.1 Text Embedder

**File**: [`src/embedding/embed.py`](src/embedding/embed.py)

**Class**: `TextEmbedder`

**Purpose**: Converts text to vector embeddings using SentenceTransformers.

**Model**: `BAAI/bge-small-en` (384 dimensions)

**Methods**:

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `embed_texts(texts)` | `List[str]` | `np.array` | Batch embedding |
| `embed_query(query)` | `str` | `np.array` | Single query embedding |

**Note**: Embeddings are L2-normalized for cosine similarity search.

---

#### 3.6.2 Vector Index

**File**: [`src/embedding/index_builder.py`](src/embedding/index_builder.py)

**Class**: `VectorIndex`

**Purpose**: FAISS-based vector storage and retrieval.

**Initialization**:
```python
index = VectorIndex(dimension=384)
```

**Methods**:

| Method | Description |
|--------|-------------|
| `add_embeddings(embeddings, metadata_list)` | Adds vectors with associated metadata |
| `save(index_path, metadata_path)` | Saves index and metadata to disk |
| `load(index_path, metadata_path)` | Loads index and metadata from disk |
| `search(query_vector, k=5)` | Returns top-k results with scores |

**Index Type**: `faiss.IndexFlatIP` (Inner Product for cosine similarity with normalized vectors)

---

### 3.7 RAG Modules

#### 3.7.1 RAG Pipeline

**File**: [`src/rag/rag_pipeline.py`](src/rag/rag_pipeline.py)

**Class**: `RAGPipeline`

**Purpose**: Orchestrates the complete RAG workflow.

**Components**:
- `Retriever`: Vector search
- `Generator`: LLM for response generation
- `CrossEncoder`: Reranking model

**Workflow**:

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Rewriting │  ← Expands shorthand, adds context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  ← K_RETRIEVAL (20) chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reranking     │  ← Cross-encoder scoring
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │  ← LLM with context
└─────────────────┘
```

**Conversation Flow**:

| Turn | Mode | Behavior |
|------|------|----------|
| 1-4 | `gathering` | Asks clarifying questions |
| 5 | `diagnosis` | Generates and verifies diagnosis |
| 6-9 | `gathering` | Extra gathering if verification fails |
| 10+ | `diagnosis` | Final diagnosis |
| Any | `escalation` | Medical escalation required, with remedies consent |
| Post-diagnosis | `remedies` | If user says YES to remedies |
| Post-diagnosis | bye message | If user says NO to remedies |

---

#### Safety Engine

**Files**: 
- [`src/rag/dynamic_safety_engine.py`](src/rag/dynamic_safety_engine.py)
- [`src/rag/risk_embeddings.py`](src/rag/risk_embeddings.py)

**Purpose**: Detects high-risk medical symptoms and prevents harmful advice.

**Risk Detection**: Uses embedding similarity to detect symptoms matching serious conditions:
- Vascular infection
- Deep vein thrombosis
- Heart failure exacerbation
- Pulmonary embolism
- Stroke symptoms
- And 20+ other critical conditions

**Safety Flow**:
1. User describes symptoms
2. Safety engine checks for high-risk patterns
3. If risk detected (similarity > 0.65): Asks user for more details instead of terminating
4. Session continues with more symptom gathering

---

#### 3.7.2 Retriever

**File**: [`src/rag/retriever.py`](src/rag/retriever.py)

**Class**: `Retriever`

**Purpose**: Handles vector search with query prefixing.

**Query Prefix**: `"Represent this sentence for searching relevant passages: "`

This prefix is required by BGE models for optimal retrieval performance.

**Method**: `retrieve(query, k=12, source_filter=None)`

**Source Filtering**: If `source_filter` is provided, retrieves 10x results and filters by source.

---

#### 3.7.3 Generator

**File**: [`src/rag/generator.py`](src/rag/generator.py)

**Class**: `Generator`

**Purpose**: LLM-based text generation using Google Gemini.

**Model**: `gemma-3-27b-it` (can be changed to Gemini models for production)

**Methods**:

| Method | Purpose |
|--------|---------|
| `generate_text(prompt)` | Simple text generation |
| `generate_diagnosis(history, chunks)` | Generates diagnosis report |
| `verify_diagnosis(report, history)` | Validates diagnosis |
| `generate(question, chunks, history, mode)` | Streaming generation with mode-specific prompts |

**Generation Modes**:

| Mode | Behavior |
|------|----------|
| `gathering` | Asks ONE clarifying question, requests age/gender first |
| `diagnosis` | Provides diagnosis summary, asks about remedies |
| `remedies` | Lists treatments, do's and don'ts |
| `final` | Professional concluding response |
| `escalation` | Handles cases requiring medical escalation, asks about remedies |
| `risk_gate_question` | Asks about current medicines and health conditions before remedies |

---

#### 3.7.4 Conversation Memory

**File**: [`src/rag/memory.py`](src/rag/memory.py)

**Class**: `ConversationMemory`

**Purpose**: Tracks conversation state and history.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `history` | `List[Dict]` | List of conversation turns |
| `max_turns` | `int` | Maximum turns to keep (default: 50) |
| `user_turn_count` | `int` | Number of user messages |
| `diagnosis_complete` | `bool` | Whether diagnosis is done |
| `waiting_remedies_consent` | `bool` | Waiting for user to request remedies |
| `last_diagnosis` | `str` | Last generated diagnosis report |

**Methods**:
- `add_turn(role, content)`: Adds a conversation turn
- `get_formatted_history()`: Returns formatted history string
- `mark_complete()`: Marks diagnosis as complete
- `clear()`: Clears history

---

#### 3.7.5 Query Rewriter

**File**: [`src/rag/query_rewriter.py`](src/rag/query_rewriter.py)

**Function**: `rewrite_query(generator, conversation_history, current_question) -> str`

**Purpose**: Rewrites follow-up questions into standalone search queries.

**Examples**:
- `"?"` → Expanded based on context
- `"tell me more"` → Expanded to specific topic
- `"what about diet?"` → `"Ayurvedic diet recommendations for [condition]"`

---

#### 3.7.6 Validator

**File**: [`src/rag/validator.py`](src/rag/validator.py)

**Purpose**: Validates JSON output from LLM.

**Required Keys**: `dosha`, `mechanism`, `symptoms`, `management`, `citations`

**Functions**:
- `looks_like_json(text)`: Checks if text appears to be JSON
- `validate_json(output)`: Validates structure and returns parsed data

---

#### 3.7.7 Structured Generator

**File**: [`src/rag/structured_generator.py`](src/rag/structured_generator.py)

**Class**: `StructuredGenerator`

**Purpose**: Wraps base generator with retry logic for structured output.

**Algorithm**:
1. Attempts generation up to `max_retries` times
2. Validates JSON structure on each attempt
3. Returns parsed JSON or error dict

---

#### 3.7.8 Prompts

**File**: [`src/rag/prompts.py`](src/rag/prompts.py)

**Purpose**: Contains system prompts for structured output mode.

**System Prompt Requirements**:
- Use only provided context
- Return valid JSON only
- Follow specific schema
- No markdown or commentary

---

### 3.8 Utility Modules

#### 3.8.1 File Utilities

**File**: [`src/utils/file_utils.py`](src/utils/file_utils.py)

**Function**: `save_chunks(chunks, output_path)`

Saves chunks to JSON file with proper encoding.

---

#### 3.8.2 Corpus Analyzer

**File**: [`src/analyze_corpus.py`](src/analyze_corpus.py)

**Function**: `analyze_corpus(chunks_dir)`

**Purpose**: Analyzes chunked corpus for quality metrics.

**Metrics Reported**:
- Total word count
- Total chunk count
- Average chunk length
- Weird character ratio (non-ASCII)
- Duplicate ratio

---

## 4. How to Run Each Module

### 4.1 Prerequisites

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install tiktoken pymupdf faiss-cpu sentence-transformers google-genai
```

### 4.2 Running the Main Pipeline (Stages 1-4)

**File**: [`src/main.py`](src/main.py)

**Command**:
```bash
cd src
python main.py
```

**What Happens**:
1. Scans `data/raw_pdfs/` for PDF files
2. For each PDF:
   - **Step 1**: Extracts pages using `pdf_extractor.py`
   - **Step 2**: Removes headers, footers, page numbers
   - **Step 3**: Removes front matter (TOC, Index)
   - **Step 4**: Parses chapters
   - **Step 5**: Chunks text with metadata extraction
   - **Step 6**: Saves chunks to `data/chunks/`

**Output Files**:
- `data/cleaned_text/{filename}_final_clean.txt`
- `data/chunks/{filename}_chunks.json`

---

### 4.3 Building Embeddings (Stage 5)

**File**: [`src/build_embeddings.py`](src/build_embeddings.py)

**Command**:
```bash
cd src
python build_embeddings.py
```

**What Happens**:
1. Loads all JSON files from `data/chunks/`
2. Extracts text and metadata from each chunk
3. Generates embeddings using `BAAI/bge-small-en`
4. Creates FAISS index
5. Saves index and metadata

**Output Files**:
- `data/embeddings/faiss.index`
- `data/embeddings/metadata.json`

**Progress Display**: Shows progress bar during embedding generation

---

### 4.4 Running the RAG Chatbot (Stages 6-7)

**File**: [`src/run_rag.py`](src/run_rag.py)

**Command**:
```bash
# From project root
python src/run_rag.py

# Or from src directory
cd src
python run_rag.py
```

**What Happens**:
1. Loads FAISS index and metadata
2. Initializes RAG pipeline with Gemini API
3. Starts interactive chat loop
4. For each user input:
   - Rewrites query
   - Retrieves relevant chunks
   - Reranks results
   - Generates response
   - Updates conversation memory

**Conversation Flow**:
```
User: I have a cough
AI: [Asks about age, gender]

User: 35, male
AI: [Asks about symptoms]

User: Dry cough, worse at night
AI: [Asks more questions]

... (continues for 5 turns)

AI: [Provides diagnosis]
AI: Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?

User: Yes
AI: [Lists treatments, do's and don'ts]

---

**Alternative - User says NO:**
```
AI: Would you like home-based remedies, do's and don'ts, and lifestyle recommendations?

User: No
AI: Thank you for consulting with us. Take care and hope I helped! Goodbye.
```

---

**Safety Alert Flow:**
```
User: I have chest pain
AI: Could you please describe your symptoms in more detail? For example, when did it start, what makes it better or worse, and do you have any other symptoms?

User: [Provides more details]
AI: [Continues diagnosis flow]
```

**Exit Commands**: Type `exit` to quit

---

### 4.5 Analyzing the Corpus

**File**: [`src/analyze_corpus.py`](src/analyze_corpus.py)

**Command**:
```bash
cd src
python analyze_corpus.py
```

**What Happens**:
1. Scans `data/chunks/` directory
2. Loads all JSON chunk files
3. Computes quality metrics
4. Prints summary statistics

**Sample Output**:
```
Total words: 125000
Total chunks: 1500
Avg chunk length: 83.33 words
Weird char ratio: 0.0012
Duplicate ratio: 0.0000
```

---

## 5. Data Flow

### 5.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA TRANSFORMATIONS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PDF File                                                                   │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ extract_pages_from_pdf()                │                               │
│  │ Input: pdf_path (str)                   │                               │
│  │ Output: List[str] (one per page)        │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ remove_headers_footers()                │                               │
│  │ Input: List[str]                        │                               │
│  │ Output: List[str]                       │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ remove_page_numbers()                   │                               │
│  │ Input: List[str]                        │                               │
│  │ Output: List[str]                       │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ "\n".join(pages) → str                  │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ remove_front_matter()                   │                               │
│  │ Input: str                              │                               │
│  │ Output: str (front matter removed)      │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ parse_chapters()                        │                               │
│  │ Input: str                              │                               │
│  │ Output: List[Dict]                      │                               │
│  │   [{"title": str, "content": str}, ...] │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ TokenChunker.chunk_text()               │                               │
│  │ Input: str (chapter content)            │                               │
│  │ Output: List[str] (chunks)              │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ create_structured_chunk()               │                               │
│  │ Input: text, source, chapter            │                               │
│  │ Output: KnowledgeChunk                  │                               │
│  │   (with all metadata fields)            │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ JSON Serialization                      │                               │
│  │ KnowledgeChunk.to_dict() → dict         │                               │
│  │ json.dump() → .json file                │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ TextEmbedder.embed_texts()              │                               │
│  │ Input: List[str]                        │                               │
│  │ Output: np.array (embeddings)           │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ VectorIndex.add_embeddings()            │                               │
│  │ Input: embeddings, metadata             │                               │
│  │ Output: FAISS index + metadata.json     │                               │
│  └─────────────────────────────────────────┘                               │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────┐                               │
│  │ RAG Pipeline                            │                               │
│  │ Retriever → Reranker → Generator        │                               │
│  │ Input: user query (str)                 │                               │
│  │ Output: response (stream)               │                               │
│  └─────────────────────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 File Structure After Processing

```
book_corpus_pipeline/
├── data/
│   ├── raw_pdfs/
│   │   ├── ayurvedic_treatment_file1.pdf      # Input
│   │   └── Ayurvedic-Home-Remedies-English.pdf # Input
│   │
│   ├── cleaned_text/
│   │   ├── ayurvedic_treatment_file1_final_clean.txt  # Stage 3 output
│   │   └── Ayurvedic-Home-Remedies-English_final_clean.txt
│   │
│   ├── chunks/
│   │   ├── ayurvedic_treatment_file1_chunks.json      # Stage 4 output
│   │   └── Ayurvedic-Home-Remedies-English_chunks.json
│   │
│   └── embeddings/
│       ├── faiss.index     # Stage 5 output (binary)
│       └── metadata.json   # Stage 5 output (chunk metadata)
│
├── src/
│   ├── main.py              # Main pipeline entry point
│   ├── build_embeddings.py  # Embedding generation
│   ├── run_rag.py           # RAG chatbot
│   ├── analyze_corpus.py    # Corpus analysis
│   ├── config.py            # Configuration
│   │
│   ├── extraction/
│   │   └── pdf_extractor.py
│   │
│   ├── cleaning/
│   │   ├── header_footer.py
│   │   ├── page_numbers.py
│   │   ├── toc_removal.py
│   │   ├── hyphenation.py
│   │   ├── line_merger.py
│   │   └── normalization.py
│   │
│   ├── structure/
│   │   ├── chapter_parser.py
│   │   ├── metadata_extractor.py
│   │   └── schema.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── embedding/
│   │   ├── embed.py
│   │   └── index_builder.py
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   ├── memory.py
│   │   ├── query_rewriter.py
│   │   ├── prompts.py
│   │   ├── validator.py
│   │   └── structured_generator.py
│   │
│   └── utils/
│       ├── file_utils.py
│       └── logger.py
│
├── requirements.txt
└── documentations/
    ├── SRS.md
    └── detailed_srs_diagram.md
```

---

## 6. Configuration

### 6.1 Modifying Chunk Parameters

Edit [`src/config.py`](src/config.py):

```python
CHUNK_SIZE = 512    # Increase for more context per chunk
CHUNK_OVERLAP = 50  # Increase for better continuity
```

### 6.2 Modifying Retrieval Parameters

```python
K_RETRIEVAL = 20    # Initial retrieval count
K_RERANK = 8        # Final chunks after reranking
K_DEFAULT = 12      # Default when not specified
```

### 6.3 Changing Models

**Embedding Model** (in `src/embedding/embed.py` and `src/rag/retriever.py`):
```python
# Change from:
model_name = "BAAI/bge-small-en"  # 384 dimensions

# To:
model_name = "BAAI/bge-base-en"   # 768 dimensions
model_name = "BAAI/bge-large-en"  # 1024 dimensions
```

**Reranker Model** (in `src/config.py`):
```python
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Alternatives:
# "cross-encoder/ms-marco-MiniLM-L-4-v2" (faster)
# "cross-encoder/ms-marco-MiniLM-L-12-v2" (more accurate)
```

**LLM Model** (in `src/rag/generator.py`):
```python
self.model_id = "gemma-3-27b-it"
# Alternatives (require API access):
# "gemini-2.0-flash"
# "gemini-2.5-flash"
```

---

## 7. Dependencies

### 7.1 Required Packages

| Package | Purpose | Used In |
|---------|---------|---------|
| `pymupdf` | PDF text extraction | `pdf_extractor.py` |
| `tiktoken` | Token counting | `chunker.py` |
| `sentence-transformers` | Embeddings & Reranking | `embed.py`, `rag_pipeline.py` |
| `faiss-cpu` | Vector similarity search | `index_builder.py` |
| `google-genai` | Gemini LLM API | `generator.py`, `run_rag.py` |
| `numpy` | Array operations | `embed.py`, `index_builder.py` |

### 7.2 Installation

```bash
pip install pymupdf tiktoken sentence-transformers faiss-cpu google-genai numpy
```

### 7.3 API Keys

The Gemini API key is currently hardcoded in `src/run_rag.py`. For production:

1. Set environment variable:
   ```bash
   set GEMINI_API_KEY=your_api_key_here
   ```

2. Modify `run_rag.py`:
   ```python
   import os
   API_KEY = os.environ.get("GEMINI_API_KEY")
   ```

---

## 8. Error Handling

### 8.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: PDF file not found` | PDF path incorrect | Check `data/raw_pdfs/` directory |
| `RuntimeError: could not open faiss.index` | Embeddings not built | Run `build_embeddings.py` first |
| `File not found: src/src/run_rag.py` | Running from wrong directory | Use `python run_rag.py` from `src/` |
| `ImportError: No module named 'pymupdf'` | Missing dependency | `pip install pymupdf` |

### 8.2 Pipeline Resilience

The main pipeline (`main.py`) wraps each PDF processing in try-except:
- Continues to next PDF if one fails
- Prints error message to console
- Does not crash the entire pipeline

---

## 9. Extending the Pipeline

### 9.1 Adding New Cleaning Steps

1. Create new file in `src/cleaning/`:
   ```python
   # src/cleaning/my_cleaner.py
   def my_cleaning_function(text):
       # Your cleaning logic
       return cleaned_text
   ```

2. Import and use in `main.py`:
   ```python
   from cleaning.my_cleaner import my_cleaning_function
   # ...
   text = my_cleaning_function(text)
   ```

### 9.2 Adding New Metadata Fields

1. Add field to `KnowledgeChunk` in `src/structure/schema.py`:
   ```python
   @dataclass
   class KnowledgeChunk:
       # ... existing fields ...
       my_new_field: Optional[str] = None
   ```

2. Add detection function in `src/structure/metadata_extractor.py`:
   ```python
   def detect_my_field(text: str):
       # Detection logic
       return value
   ```

3. Update `create_structured_chunk()` in `src/chunking/chunker.py`

### 9.3 Adding New Chapter Markers

Edit `src/structure/chapter_parser.py`:
```python
chapter_markers = [
    # ... existing markers ...
    "New Chapter Name",
]
```

---

## 10. Performance Considerations

### 10.1 Memory Usage

| Stage | Memory Impact |
|-------|---------------|
| PDF Extraction | Low (one page at a time) |
| Embedding Generation | High (batch processing) |
| FAISS Search | Low (index is memory-mapped) |

### 10.2 Processing Time

| Stage | Time (per 200-page book) |
|-------|--------------------------|
| Extraction + Cleaning | ~10 seconds |
| Chunking | ~5 seconds |
| Embedding Generation | ~2-5 minutes (CPU) |
| RAG Query | ~3-10 seconds per query |

### 10.3 Optimization Tips

1. **Use GPU for embeddings**: Install `faiss-gpu` instead of `faiss-cpu`
2. **Batch processing**: Process multiple PDFs in parallel
3. **Caching**: Cache embeddings to avoid regeneration
4. **Smaller chunks**: Reduce `CHUNK_SIZE` for more granular retrieval

---

*Document generated for SRS creation - Book Corpus Pipeline v1.0*