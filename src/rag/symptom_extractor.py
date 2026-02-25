# rag/symptom_extractor.py

import json

SYMPTOM_EXTRACTION_PROMPT = """
Extract all medically relevant symptoms from the user message.

Return ONLY valid JSON in this format with no additional text:

{
  "primary_symptoms": ["list of main symptoms"],
  "secondary_symptoms": ["list of secondary symptoms"],
  "systemic_symptoms": ["list of systemic symptoms like fever, fatigue"],
  "duration": "duration if mentioned",
  "severity": "severity if mentioned"
}

User message:
{user_input}
"""

def extract_symptoms_llm(user_input: str, generator):
    """
    Extract symptoms using the generator's generate_text method.
    
    Args:
        user_input: The user's symptom description
        generator: Generator instance with generate_text method
    
    Returns:
        Parsed JSON with symptom categories
    """
    prompt = SYMPTOM_EXTRACTION_PROMPT.format(user_input=user_input)
    
    try:
        response = generator.generate_text(prompt)
        
        # Try to parse JSON from response
        # Handle potential markdown code blocks
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        parsed = json.loads(response.strip())
        return parsed
        
    except json.JSONDecodeError as e:
        # Return empty structure if parsing fails
        print(f"[JSON parsing error in symptom extraction: {e}]")
        return {
            "primary_symptoms": [],
            "secondary_symptoms": [],
            "systemic_symptoms": [],
            "duration": "",
            "severity": ""
        }
    except Exception as e:
        print(f"[Symptom extraction error: {e}]")
        return {
            "primary_symptoms": [],
            "secondary_symptoms": [],
            "systemic_symptoms": [],
            "duration": "",
            "severity": ""
        }
