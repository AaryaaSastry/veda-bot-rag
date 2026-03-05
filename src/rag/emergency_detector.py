# src/rag/emergency_detector.py

import json

EMERGENCY_DETECTION_PROMPT = """
Analyze the latest user input and conversation history for any immediate medical emergencies or high-risk red flags.

Triggers for emergency:
- Severe chest pain or pressure
- Difficulty breathing or severe shortness of breath
- Uncontrolled or heavy bleeding
- Loss of consciousness, fainting, or sudden confusion
- Signs of stroke (facial drooping, arm weakness, speech difficulty)
- Severe allergic reaction (swelling of throat, lips, or tongue)
- Any mention of self-harm or suicidal ideation

Conversation History:
{history}

Latest User Input:
{user_input}

Return ONLY valid JSON:
{{
  "is_emergency": true/false,
  "emergency_type": "string describing the emergency or 'none'",
  "urgency_score": float between 0.0 and 1.0,
  "recommended_action": "e.g., 'Call local emergency services immediately'"
}}
"""

class EmergencyDetector:
    def __init__(self, generator):
        """
        Uses the provided generator (Gemini) to perform high-stakes emergency detection.
        """
        self.generator = generator

    def evaluate(self, user_input, history):
        prompt = EMERGENCY_DETECTION_PROMPT.format(
            history=history,
            user_input=user_input
        )
        
        try:
            # We use the raw text generator for the safety check to ensure zero hallucinations
            # and strictly structured JSON response.
            response = self.generator.generate_text(prompt)
            response = response.strip()
            
            # Handle markdown blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response.strip())
            return result
        except Exception as e:
            print(f"[Emergency Detector Error: {e}]")
            # Default to safe side if check fails
            return {
                "is_emergency": False, 
                "emergency_type": "none", 
                "urgency_score": 0.0, 
                "recommended_action": ""
            }
