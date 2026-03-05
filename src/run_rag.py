"""
Ayurvedic AI Chatbot - Enhanced RAG Pipeline

Features:
- Medical safety detection for high-risk symptoms
- Symptom extraction and weighted retrieval
- Body system classification for metadata filtering
- Multi-stage retrieval with reranking
- Conversational memory with diagnosis flow
- Automatic evaluation report generation
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from rag.rag_pipeline import RAGPipeline
from rag.memory import ConversationMemory
# from rag.metadata_enricher import enrich_chunks_with_metadata

# Load .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluations")

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in .env file.")

BOOK_1 = "ayurvedic_treatment_file1.pdf"
BOOK_2 = "Ayurvedic-Home-Remedies-English.pdf"


class SessionEvaluator:
    """Generates evaluation reports for each assessment session."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session_data = {
            "start_time": None,
            "end_time": None,
            "turns": [],
            "safety_alerts": [],
            "diagnosis": None,
            "retrieval_stats": []
        }
    
    def start_session(self):
        """Mark session start time."""
        self.session_data["start_time"] = datetime.now().isoformat()
    
    def end_session(self):
        """Mark session end time and save report."""
        self.session_data["end_time"] = datetime.now().isoformat()
        return self.save_report()
    
    def log_turn(self, turn_number, user_input, assistant_response, 
                 safety_check=None, retrieval_query=None, chunks_retrieved=0):
        """Log a conversation turn."""
        turn_data = {
            "turn_number": turn_number,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": assistant_response,
            "chunks_retrieved": chunks_retrieved
        }
        
        if safety_check:
            turn_data["safety_check"] = {
                "is_safe": safety_check[0],
                "matched_risks": safety_check[1].get("matched_risks", []) if safety_check[1] else []
            }
            if not safety_check[0]:
                self.session_data["safety_alerts"].append({
                    "turn": turn_number,
                    "input": user_input,
                    "risks": safety_check[1].get("matched_risks", [])
                })
        
        if retrieval_query:
            turn_data["retrieval_query"] = retrieval_query
            
        self.session_data["turns"].append(turn_data)
    
    def log_diagnosis(self, diagnosis_report):
        """Log the final diagnosis."""
        self.session_data["diagnosis"] = diagnosis_report
    
    def log_retrieval_stats(self, query, num_chunks, top_scores=None):
        """Log retrieval statistics."""
        stats = {
            "query": query,
            "num_chunks": num_chunks,
            "timestamp": datetime.now().isoformat()
        }
        if top_scores:
            stats["top_scores"] = top_scores
        self.session_data["retrieval_stats"].append(stats)
    
    def save_report(self):
        """Save the evaluation report to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate summary statistics
        total_turns = len(self.session_data["turns"])
        total_safety_alerts = len(self.session_data["safety_alerts"])
        
        report = {
            "session_summary": {
                "start_time": self.session_data["start_time"],
                "end_time": self.session_data["end_time"],
                "total_turns": total_turns,
                "total_safety_alerts": total_safety_alerts,
                "diagnosis_completed": self.session_data["diagnosis"] is not None
            },
            "safety_alerts": self.session_data["safety_alerts"],
            "diagnosis": self.session_data["diagnosis"],
            "conversation_turns": self.session_data["turns"],
            "retrieval_statistics": self.session_data["retrieval_stats"]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath


if __name__ == "__main__":
    # Initialize the Bayesian-first RAG pipeline
    pipeline = RAGPipeline(
        vector_db_path=VECTOR_DB_PATH, 
        api_key=API_KEY
    )
    memory = ConversationMemory()
    evaluator = SessionEvaluator(EVALUATION_DIR)
    
    # Start session evaluation
    evaluator.start_session()
    
    print("--- Veda Bot: Bayesian Diagnostic Engine (Gemma-3-27b + Llama 3) ---")
    print("(Type 'exit' to quit)")

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Closing session. Stay well.")
                break

            if not user_input:
                continue

            # 1. Emergency Detection / Safety Check
            is_safe, safety_result = pipeline.check_safety(user_input)
            
            # Log turn for evaluation
            evaluator.log_turn(
                turn_number=len(memory.turns) + 1,
                user_input=user_input,
                assistant_response="[Processing...]",
                safety_check=(is_safe, safety_result)
            )

            # 2. Add to memory
            memory.add_turn("user", user_input)

            # 3. Run Pipeline (Generator yields chunks)
            print("\nAI: ", end="", flush=True)
            full_response = ""
            
            for chunk in pipeline.run(user_input, memory):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

            # 4. Update memory with response
            memory.add_turn("assistant", full_response)
            
            # 5. Check if session has concluded
            if getattr(memory, "diagnosis_complete", False):
                print("\n--- Diagnostic Session Concluded ---")
                if hasattr(memory, "last_diagnosis") and memory.last_diagnosis:
                    evaluator.log_diagnosis(memory.last_diagnosis)
                break
    
    finally:
        # Save evaluation report
        report_path = evaluator.end_session()
        print(f"\n[System] Evaluation report archived: {report_path}")
