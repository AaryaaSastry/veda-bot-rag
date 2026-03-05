# src/rag/symptom_canonicalizer.py

import json
import re

CANONICALIZATION_PROMPT = """
You are a medical symptom normalizer. Your task is to map raw user descriptions of symptoms to the canonical symptom names from our Knowledge Base.

Knowledge Base Symptoms:
{knowledge_base_symptoms}

User Symptoms to map:
{user_symptoms}

Return ONLY valid JSON in this format:
{{
  "mappings": [
    {{"raw": "burning skin", "canonical": "burning sensation"}},
    {{"raw": "painless swelling", "canonical": "swelling"}}
  ]
}}

If a symptom doesn't match anything in the Knowledge Base, skip it or map to 'unknown'.
"""

def canonicalize_symptoms_llm(user_symptoms: list, kb_symptoms: list, generator) -> dict:
    """
    Maps raw symptom strings to canonical KB symptoms using LLM logic.
    """
    if not user_symptoms:
        return {}

    # Truncate KB list if too long to save tokens, or focus on relevant ones
    kb_str = "\n".join([f"- {s}" for s in kb_symptoms])
    user_str = "\n".join([f"- {s}" for s in user_symptoms])

    prompt = CANONICALIZATION_PROMPT.replace("{knowledge_base_symptoms}", kb_str).replace("{user_symptoms}", user_str)
    
    try:
        response = generator.generate_text(prompt)
        response = response.strip()
        
        # Handle markdown blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        mapping_data = json.loads(response.strip())
        
        # Convert to a direct lookup dict
        return {item["raw"]: item["canonical"] for item in mapping_data.get("mappings", []) if item.get("canonical") != "unknown"}
        
    except Exception as e:
        print(f"[Symptom canonicalization error: {e}]")
        return {}

def fuzzy_canonicalizer(raw_symptom: str, kb_symptoms: list) -> str:
    """
    Simple fallback fuzzy matcher for canonicalization.
    """
    raw = raw_symptom.lower().strip()
    # Direct match
    if raw in kb_symptoms:
        return raw
    
    # Substring match
    for kb in kb_symptoms:
        if raw in kb or kb in raw:
            return kb
            
    return None
