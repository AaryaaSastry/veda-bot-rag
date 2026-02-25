import os
import json
import re
from collections import Counter

def analyze_corpus(chunks_dir):
    total_words = 0
    total_chunks = 0
    all_chunks = []
    total_chars = 0
    weird_chars_count = 0
    
    # regex for "weird" characters (non-standard ASCII/punctuation)
    weird_char_pattern = re.compile(r'[^\x00-\x7F]+')

    if not os.path.exists(chunks_dir):
        print(f"Directory {chunks_dir} not found.")
        return

    for filename in os.listdir(chunks_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(chunks_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Assuming data is a list of chunks or a dict with a 'chunks' key
                    chunks = data if isinstance(data, list) else data.get('chunks', [])
                    
                    for chunk in chunks:
                        # Extract text if chunk is an object
                        text = chunk if isinstance(chunk, str) else chunk.get('text', '')
                        
                        total_chunks += 1
                        all_chunks.append(text)
                        
                        words = text.split()
                        total_words += len(words)
                        
                        total_chars += len(text)
                        weird_chars_count += len(weird_char_pattern.findall(text))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if total_chunks == 0:
        print("No chunks found to analyze.")
        return

    avg_chunk_length = total_words / total_chunks
    weird_char_ratio = weird_chars_count / total_chars if total_chars > 0 else 0
    
    # Duplicate ratio
    duplicates = total_chunks - len(set(all_chunks))
    duplicate_ratio = duplicates / total_chunks

    print(f"Total words: {total_words}")
    print(f"Total chunks: {total_chunks}")
    print(f"Avg chunk length: {avg_chunk_length:.2f} words")
    print(f"Weird char ratio: {weird_char_ratio:.4f}")
    print(f"Duplicate ratio: {duplicate_ratio:.4f}")

if __name__ == "__main__":
    # Path relative to project root
    CHUNKS_PATH = os.path.join("data", "chunks")
    analyze_corpus(CHUNKS_PATH)
