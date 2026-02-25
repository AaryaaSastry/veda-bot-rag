from rag.rag_pipeline import RAGPipeline
from rag.memory import ConversationMemory
import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")
API_KEY = "AIzaSyANOCboTctudlub0sgx7DYyfh99rBRjK1E"

BOOK_1 = "ayurvedic_treatment_file1.pdf"
BOOK_2 = "Ayurvedic-Home-Remedies-English.pdf"

class GeminiGenerator:
    def __init__(self, api_key):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemma-3-27b-it"

    def generate_text(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return response.text

if __name__ == "__main__":

    pipeline = RAGPipeline(VECTOR_DB_PATH, API_KEY)
    memory = ConversationMemory()

    print("--- Ayurvedic AI Chatbot (type 'exit' to quit) ---")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        memory.add_turn("user", user_input)

        print("\nAI: ", end="", flush=True)
        full_response = ""
        # We still stream for better user experience
        for chunk in pipeline.run(user_input, memory):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()

        memory.add_turn("assistant", full_response)

        if memory.diagnosis_complete:
            print("\n--- Session Complete ---")
            break
