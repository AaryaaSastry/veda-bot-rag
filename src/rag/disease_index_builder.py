import sys
import os
import json

# Add project root to sys.path to allow imports from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.generator import Generator

DISEASE_INDEX_PROMPT = """
SYSTEM:
You are an Ayurvedic medical knowledge architect.
Your task is to synthesize balanced "Disease Profiles" from multiple knowledge chunks.

INPUT CHUNKS:
{chunks_text}

TASK:
Identify ONE specific Ayurvedic condition or disease mentioned in these chunks.
Create a structured index entry for it.

Return ONLY valid JSON with this exact schema:
{{
 "disease": "English name (Sanskrit name)",
 "symptoms": ["Symptom 1", "Symptom 2", "..."],
 "dosha": "Aggravated Dosha(s)",
 "srotas": "Channels involved",
 "treatments": ["Conservative treatment 1", "Herbs", "..."],
 "diet": ["Wholesome foods", "Foods to avoid"],
 "lifestyle": ["Recommended habits", "Precautions"],
 "source_references": ["book_name_1.pdf", "book_name_2.pdf"]
}}

Rules:
- If multiple diseases are found, choose the most prominent one.
- If no specific disease is found, return {{"error": "No disease found"}}.
- DO NOT invent information. Use ONLY the provided chunks.
"""

class DiseaseIndexBuilder:
    def __init__(self, api_key):
        self.generator = Generator(api_key)
        self.index_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "disease_index.json"
        )

    def build_index(self, chunks_metadata, batch_size=5):
        """
        Group chunks by content and generate disease profiles.
        """
        disease_index = []
        
        # Simplified: group by chapter/topic to find related chunks
        grouped_chunks = {}
        for chunk in chunks_metadata:
            key = f"{chunk.get('chapter', 'General')}_{chunk.get('category', 'Other')}"
            if key not in grouped_chunks:
                grouped_chunks[key] = []
            grouped_chunks[key].append(chunk)

        print(f"Generating profiles for {len(grouped_chunks)} groups...")
        
        for key, chunks in grouped_chunks.items():
            # Send small batches of text to the LLM to synthesize
            text_context = "\n\n---\n\n".join([c['text'][:1000] for c in chunks[:5]])
            
            prompt = DISEASE_INDEX_PROMPT.replace("{chunks_text}", text_context)
            try:
                response = self.generator.generate_text(prompt)
                # Cleanup markdown
                response = response.strip()
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                entry = json.loads(response.strip())
                if "disease" in entry and "error" not in entry:
                    disease_index.append(entry)
                    print(f"Added: {entry['disease']}")
            except Exception as e:
                print(f"Error processing group {key}: {e}")

        # Save the index
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(disease_index, f, indent=2, ensure_ascii=False)
        
        print(f"Disease index saved to {self.index_path}")

if __name__ == "__main__":
    import sys
    # Expect API Key as argument for standalone run
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable required")
        sys.exit(1)
        
    metadata_path = os.path.join("data", "embeddings", "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Metadata not found at {metadata_path}")
        sys.exit(1)
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    builder = DiseaseIndexBuilder(api_key)
    builder.build_index(meta)
