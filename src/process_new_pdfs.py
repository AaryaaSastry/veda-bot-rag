"""
Wrapper script for processing only new PDFs, then rebuilding embeddings.

This file intentionally delegates all work to:
- main.process_pdfs(...)
- build_embeddings.build_embeddings(...)
to avoid duplicated pipeline logic.
"""

import os
from main import process_pdfs, RAW_DIR
from build_embeddings import build_embeddings


def main():
    print("\n" + "=" * 60)
    print("PDF PROCESSING PIPELINE (NEW FILES ONLY)")
    print("=" * 60)

    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Raw PDF directory not found: {RAW_DIR}")

    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    print(f"\nFound {len(pdf_files)} PDFs in {RAW_DIR}")
    for f in pdf_files:
        print(f"  - {f}")

    process_pdfs(skip_existing=True)
    build_embeddings()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the RAG pipeline with: python src/run_rag.py")


if __name__ == "__main__":
    main()
