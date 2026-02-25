# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for the Book Corpus Pipeline project, which processes raw book PDFs into cleaned, chunked, and embedded data for downstream NLP tasks such as retrieval-augmented generation (RAG).

### 1.2 Scope
The pipeline automates the extraction, cleaning, chunking, and embedding of book data, enabling efficient search and retrieval. It is intended for researchers and developers working with large text corpora.

### 1.3 Definitions, Acronyms, and Abbreviations
- **PDF**: Portable Document Format
- **RAG**: Retrieval-Augmented Generation
- **Embedding**: Vector representation of text
- **Chunking**: Splitting text into manageable pieces

## 2. Overall Description

### 2.1 Product Perspective
The pipeline is a standalone system that processes raw PDFs and outputs cleaned text, chunked data, and embeddings. It integrates with FAISS for vector search and supports modular processing steps.

### 2.2 Product Functions
- Extract text from PDFs
- Clean and normalize text (remove headers, footers, page numbers, etc.)
- Chunk text into segments
- Generate embeddings for each chunk
- Store embeddings and metadata for retrieval

### 2.3 User Classes and Characteristics
- **Researchers**: Use the pipeline to prepare corpora for NLP experiments
- **Developers**: Integrate the pipeline into larger systems

### 2.4 Operating Environment
- OS: Windows
- Python 3.x
- Required libraries: FAISS, PyPDF2, etc.

### 2.5 Design and Implementation Constraints
- Input files must be in PDF format
- Output directories must exist or be creatable

### 2.6 User Documentation
- README.md
- Inline code comments

### 2.7 Assumptions and Dependencies
- All dependencies are installed via requirements.txt
- Input PDFs are text-based (not scanned images)

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 PDF Extraction
- The system shall accept PDF files as input from the `data/raw_pdfs/` directory.
- The system shall extract text from each page of the PDF, preserving page boundaries.
- If a PDF file is missing or unreadable, the system shall log an error and skip the file.

#### 3.1.2 Text Cleaning
- The system shall remove recurring headers and footers using frequency analysis.
- The system shall remove standalone page numbers (Roman or Arabic numerals) from the top and bottom of pages.
- The system shall remove front matter (TOC, Index, Foreword) by detecting chapter start markers.
- Cleaned text shall be saved to `data/cleaned_text/` with a filename indicating the source and cleaning stage.

#### 3.1.3 Structural Parsing
- The system shall parse cleaned text into chapters using predefined chapter markers.
- Each chapter shall be represented as a dictionary with `title` and `content` fields.

#### 3.1.4 Chunking
- The system shall split chapter text into overlapping chunks based on a configurable token size and overlap.
- Each chunk shall be wrapped in a `KnowledgeChunk` object containing metadata (source, chapter, topic, dosha, category, disease type, srotas, treatment type, level of care, formulation type).
- Chunks shall be saved as JSON files in `data/chunks/`.

#### 3.1.5 Embedding Generation
- The system shall generate vector embeddings for each chunk using a SentenceTransformer model.
- Embeddings and associated metadata shall be stored in a FAISS index and a metadata JSON file in `data/embeddings/`.

#### 3.1.6 Retrieval-Augmented Generation (RAG)
- The system shall retrieve relevant chunks from the FAISS index given a user query, using a query rewriter and reranker.
- The system shall generate responses using a language model, grounded in the retrieved sources.
- The system shall support multi-turn conversations, tracking user history and context.

#### 3.1.7 Error Handling and Logging
- The system shall log all errors, warnings, and processing steps to the console and/or a log file.
- If a processing step fails for a file, the system shall continue with the next file.

### 3.2 Non-Functional Requirements
- The system shall process large PDF files efficiently (target: <5 minutes per 200-page book on a modern CPU).
- The system shall be modular, allowing individual pipeline stages to be run independently.
- The system shall be compatible with Windows OS and Python 3.x.
- The system shall use only open-source libraries or those listed in requirements.txt.

### 3.3 Interface Requirements
- Input: PDF files in `data/raw_pdfs/`.
- Output: Cleaned text files in `data/cleaned_text/`, chunked JSON files in `data/chunks/`, embeddings and metadata in `data/embeddings/`.
- Configuration: Parameters (chunk size, overlap, model names) set in `config.py`.

### 3.4 Data Requirements
- Each `KnowledgeChunk` shall include: text, source, chapter, topic, dosha, category, disease type, srotas, treatment type, level of care, formulation type.
- Embeddings shall be stored in FAISS index format, with metadata in JSON.

### 3.5 Security Requirements
- The system shall not expose sensitive data or API keys in logs or outputs.

### 3.6 Logging and Monitoring
- The system shall provide clear logging for each pipeline stage, including start/end, errors, and summary statistics.

### 3.7 Extensibility
- The system shall allow for new cleaning, chunking, or embedding modules to be added with minimal changes to the pipeline.

## 4. Appendices

### 4.1 Glossary
- **Chunk**: A segment of text, typically a few hundred tokens, used for embedding and retrieval.
- **Embedding**: A vector representation of text for similarity search.
- **FAISS**: Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors.
- **RAG**: Retrieval-Augmented Generation, a method combining retrieval and generation for question answering.

### 4.2 References
- Project source code and documentation
- [FAISS Documentation](https://faiss.ai/)
- [SentenceTransformers Documentation](https://www.sbert.net/)

### 4.3 Diagrams
See `diagrams.md` for activity and swimlane diagrams.
