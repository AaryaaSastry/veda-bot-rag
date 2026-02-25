# Detailed SRS Flowchart Diagram

```mermaid
flowchart TD
    subgraph User
        A[Provide PDF files]
    end
    subgraph Pipeline
        B[Extract PDF Text]
        C[Remove Headers/Footers]
        D[Remove Page Numbers]
        E[Remove Front Matter]
        F[Save Cleaned Text]
        G[Parse Chapters]
        H[Chunk Chapters]
        I[Extract Metadata]
        J[Save Chunks as JSON]
        K[Generate Embeddings]
        L[Build FAISS Index]
        M[Save Embeddings & Metadata]
        N[Retrieve Chunks (RAG)]
        O[Rewrite Query]
        P[Rerank Results]
        Q[Generate Response]
    end
    subgraph ErrorHandling[Error Handling]
        X1[Log Extraction Errors]
        X2[Log Cleaning Errors]
        X3[Log Chunking Errors]
        X4[Log Embedding Errors]
        X5[Log Retrieval Errors]
    end
    A --> B
    B -->|If error| X1
    B --> C
    C -->|If error| X2
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K -->|If error| X4
    K --> L
    L --> M
    M --> N
    N -->|If error| X5
    N --> O
    O --> P
    P --> Q
    H -->|If error| X3
```

*This diagram shows all major modules, data flows, and error handling in the Book Corpus Pipeline.*
