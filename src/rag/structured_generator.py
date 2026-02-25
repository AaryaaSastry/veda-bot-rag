from .validator import validate_json, looks_like_json

class StructuredGenerator:

    def __init__(self, base_generator, max_retries=3):
        self.base_generator = base_generator
        self.max_retries = max_retries

    def generate(self, question, context):

        for attempt in range(self.max_retries):

            raw_output = self.base_generator.generate(question, context)

            if not looks_like_json(raw_output):
                continue

            valid, parsed = validate_json(raw_output)

            if valid:
                return parsed

        return {
            "error": "Model failed to produce valid structured output."
        }