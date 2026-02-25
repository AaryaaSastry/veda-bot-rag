import os
import json
import config
from extraction.pdf_extractor import extract_pages_from_pdf
from cleaning.header_footer import remove_headers_footers
from cleaning.page_numbers import remove_page_numbers
from cleaning.toc_removal import remove_front_matter
from structure.chapter_parser import parse_chapters
from chunking.chunker import create_structured_chunk, TokenChunker

def save_chunks(chunks, output_path):
    """
    Saves a list of KnowledgeChunk objects to a JSON file.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw_pdfs")
    cleaned_dir = os.path.join(project_root, "data", "cleaned_text")
    chunks_dir = os.path.join(project_root, "data", "chunks")
    
    # Ensure directories exist
    for d in [cleaned_dir, chunks_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    for filename in os.listdir(raw_dir):
        if filename.endswith(".pdf"):
            print(f"\n--- Processing: {filename} ---")
            pdf_path = os.path.join(raw_dir, filename)
            
            try:
                # 1. Extraction (Page-level)
                print("Step 1: Extracting pages...")
                pages = extract_pages_from_pdf(pdf_path)
                
                # 2. Page-level Cleaning
                print("Step 2: Cleaning headers, footers and page numbers...")
                pages = remove_headers_footers(pages)
                pages = remove_page_numbers(pages)
                
                # 3. Global Cleaning
                print("Step 3: Removing front matter...")
                full_text = "\n".join(pages)
                full_text = remove_front_matter(full_text)
                
                # 4. Save intermediate cleaned text
                cleaned_filename = filename.replace(".pdf", "_final_clean.txt")
                with open(os.path.join(cleaned_dir, cleaned_filename), "w", encoding="utf-8") as f:
                    f.write(full_text)

                # 5. Structural Parsing
                print("Step 4: Parsing chapters...")
                chapters = parse_chapters(full_text)
                print(f"   Found {len(chapters)} chapters.")
                
                # 6. Chunking & Enrichment
                print("Step 5: Chunking and adding metadata...")
                all_structured_chunks = []
                chunker = TokenChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
                
                for chapter in chapters:
                    raw_chunks = chunker.chunk_text(chapter["content"])
                    for chunk in raw_chunks:
                        sc = create_structured_chunk(
                            text=chunk,
                            source=filename,
                            chapter=chapter["title"]
                        )
                        all_structured_chunks.append(sc)
                
                # 7. Final Output
                print(f"Step 6: Saving {len(all_structured_chunks)} structured chunks to JSON...")
                output_filename = filename.replace(".pdf", "_chunks.json")
                save_chunks(all_structured_chunks, os.path.join(chunks_dir, output_filename))
                
                print(f" WORKFLOW COMPLETE: {output_filename}")
                
            except Exception as e:
                print(f" Error: {e}")


if __name__ == "__main__":
    main()
