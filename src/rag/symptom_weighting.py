# rag/symptom_weighting.py

import glob
import json
import math
from collections import Counter


def build_frequency_index(chunks_path="data/chunks/*.json"):
    """
    Build a frequency index from all chunk files.
    
    Args:
        chunks_path: Glob pattern for chunk JSON files
    
    Returns:
        Tuple of (frequency Counter, total chunk count)
    """
    freq = Counter()
    total_chunks = 0

    files = glob.glob(chunks_path)
    
    if not files:
        print(f"[Warning: No chunk files found at {chunks_path}]")
        return freq, 1  # Return empty freq with 1 to avoid division by zero

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                total_chunks += 1
                # Handle both "text" and "content" keys
                text = chunk.get("text", chunk.get("content", ""))
                words = set(text.lower().split())

                for word in words:
                    freq[word] += 1
        except Exception as e:
            print(f"[Error loading {file}: {e}]")
            continue

    return freq, max(total_chunks, 1)  # Ensure at least 1 to avoid division by zero


def compute_weight(symptom: str, freq_index, total_chunks):
    """
    Compute IDF-style weight for a symptom phrase.
    
    Args:
        symptom: Symptom text
        freq_index: Counter of word frequencies
        total_chunks: Total number of chunks
    
    Returns:
        Weight score based on inverse document frequency
    """
    words = symptom.lower().split()
    score = 0

    for w in words:
        f = freq_index.get(w, 1)
        score += math.log(total_chunks / f)

    return score
