# rag/metadata_enricher.py

import json
import glob

SYSTEM_CLASSIFICATION_PROMPT = """
Analyze the following medical text and determine:

1. The primary body system involved 
   (circulatory, digestive, respiratory, musculoskeletal, nervous, urinary, reproductive, systemic, other)

2. Extract important symptom keywords (max 10).

Return ONLY valid JSON with no additional text:

{
  "primary_system": "",
  "symptom_keywords": []
}

Text:
{chunk_text}
"""

def enrich_chunks_with_metadata(generator, chunks_path="data/chunks/*.json"):
    """
    Enrich chunk metadata with body system classification and symptom keywords.
    
    Args:
        generator: Generator instance with generate_text method
        chunks_path: Glob pattern for chunk JSON files
    
    Returns:
        Number of chunks enriched
    """
    enriched_count = 0
    
    for file in glob.glob(chunks_path):
        try:
            with open(file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                # Skip if already enriched
                if chunk.get("primary_system"):
                    continue
                    
                prompt = SYSTEM_CLASSIFICATION_PROMPT.format(
                    chunk_text=chunk.get("text", chunk.get("content", ""))[:1500]
                )

                try:
                    response = generator.generate_text(prompt)
                    
                    # Handle potential markdown code blocks
                    response = response.strip()
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0]
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0]
                    
                    metadata = json.loads(response.strip())
                    
                    chunk["primary_system"] = metadata.get("primary_system", "other")
                    chunk["symptom_keywords"] = metadata.get("symptom_keywords", [])
                    enriched_count += 1
                    
                except json.JSONDecodeError as e:
                    chunk["primary_system"] = "other"
                    chunk["symptom_keywords"] = []
                except Exception as e:
                    chunk["primary_system"] = "other"
                    chunk["symptom_keywords"] = []

            # Save enriched chunks
            with open(file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[Error processing {file}: {e}]")
            continue

    print(f"Metadata enrichment complete. Enriched {enriched_count} chunks.")
    return enriched_count
