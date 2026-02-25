# rag/system_classifier.py

import json

QUERY_SYSTEM_PROMPT = """
Analyze the user symptoms and determine the primary affected body system.

Choose one:
circulatory, digestive, respiratory, musculoskeletal, nervous, urinary, reproductive, systemic, other

Return ONLY JSON with no additional text:

{
  "primary_system": ""
}

User input:
{user_input}
"""

def classify_query_system(user_input, generator):
    """
    Classify the body system affected by the user's symptoms.
    
    Args:
        user_input: User's symptom description
        generator: Generator instance with generate_text method
    
    Returns:
        String indicating the primary body system
    """
    prompt = QUERY_SYSTEM_PROMPT.format(user_input=user_input)
    
    try:
        response = generator.generate_text(prompt)
        
        # Handle potential markdown code blocks
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        parsed = json.loads(response.strip())
        return parsed.get("primary_system", "other")
        
    except json.JSONDecodeError as e:
        print(f"[JSON parsing error in system classification: {e}]")
        return "other"
    except Exception as e:
        print(f"[System classification error: {e}]")
        return "other"
