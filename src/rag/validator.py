import json
import re

REQUIRED_KEYS = [
    "dosha",
    "mechanism",
    "symptoms",
    "management",
    "citations"
]

def looks_like_json(text: str) -> bool:
    return bool(re.search(r'^\s*\{.*\}\s*$', text, re.DOTALL))

def validate_json(output: str):
    try:
        data = json.loads(output)

        for key in REQUIRED_KEYS:
            if key not in data:
                return False, None

        if not isinstance(data["symptoms"], list):
            return False, None

        if not isinstance(data["management"], list):
            return False, None

        if not isinstance(data["citations"], list):
            return False, None

        return True, data

    except Exception:
        return False, None