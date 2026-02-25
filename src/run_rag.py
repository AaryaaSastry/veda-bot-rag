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
from rag.rag_pipeline import RAGPipeline
from rag.memory import ConversationMemory
from rag.metadata_enricher import enrich_chunks_with_metadata

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluations")
API_KEY = "AIzaSyCXAd1eoAyTYB80xnPmi0dqg1rAQhvhz0U"

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


def run_metadata_enrichment(api_key):
    """
    One-time metadata enrichment for chunks.
    Run this once to add primary_system and symptom_keywords to chunks.
    """
    from rag.generator import Generator
    generator = Generator(api_key)
    
    chunks_path = os.path.join(PROJECT_ROOT, "data", "chunks", "*.json")
    enrich_chunks_with_metadata(generator, chunks_path)


if __name__ == "__main__":
    # Initialize the enhanced pipeline
    pipeline = RAGPipeline(
        vector_db_path=VECTOR_DB_PATH, 
        api_key=API_KEY,
        use_enhanced_retrieval=True,  # Enable symptom extraction and system classification
        safety_threshold=0.65  # Threshold for medical safety detection
    )
    memory = ConversationMemory()
    evaluator = SessionEvaluator(EVALUATION_DIR)
    
    # Start session evaluation
    evaluator.start_session()
    
    print("--- Ayurvedic AI Chatbot (type 'exit' to quit) ---")
    print("--- Enhanced with symptom-based retrieval and safety detection ---")
    print("--- Evaluation reports saved to: data/evaluations/ ---\n")

    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Check for medical safety risks before processing
            is_safe, safety_result = pipeline.check_safety(user_input)
            
            if not is_safe:
                print("\n⚠️ MEDICAL SAFETY ALERT")
                print("The symptoms you described may indicate a serious condition.")
                print("Matched risks:", safety_result["matched_risks"])
                print("\nPlease seek immediate medical attention.")
                print("This AI is not a substitute for professional medical care.")
                
                # Log the safety alert
                evaluator.log_turn(
                    turn_number=memory.user_turn_count + 1,
                    user_input=user_input,
                    assistant_response="[SAFETY ALERT - Session terminated]",
                    safety_check=(is_safe, safety_result)
                )
                continue

            memory.add_turn("user", user_input)

            print("\nAI: ", end="", flush=True)
            full_response = ""
            
            # Get retrieval query for logging
            retrieval_query, _ = pipeline._classify_and_weight_query(user_input) if memory.user_turn_count == 0 else (user_input, None)
            
            # Stream response for better user experience
            for chunk in pipeline.run(user_input, memory):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

            memory.add_turn("assistant", full_response)
            
            # Log the turn
            evaluator.log_turn(
                turn_number=memory.user_turn_count,
                user_input=user_input,
                assistant_response=full_response,
                safety_check=(is_safe, safety_result),
                retrieval_query=retrieval_query
            )
            
            # Log diagnosis if completed
            if memory.last_diagnosis:
                evaluator.log_diagnosis(memory.last_diagnosis)

            if memory.diagnosis_complete:
                print("\n--- Session Complete ---")
                break
    
    finally:
        # Always save the evaluation report
        report_path = evaluator.end_session()
        print(f"\n--- Evaluation Report Saved: {report_path} ---")
