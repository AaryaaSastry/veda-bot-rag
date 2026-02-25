import config
import os
from rag.retriever import Retriever
from rag.hybrid_fusion_retriever import HybridFusionRetriever
from rag.generator import Generator
from sentence_transformers import CrossEncoder, SentenceTransformer
from rag.query_rewriter import rewrite_query
from rag.symptom_extractor import extract_symptoms_llm
from rag.symptom_weighting import build_frequency_index
from rag.weighted_query_builder import build_weighted_query
from rag.system_classifier import classify_query_system
from rag.risk_embeddings import RiskEmbeddingStore
from rag.dynamic_safety_engine import DynamicMedicalSafetyEngine


class RAGPipeline:
    """
    Enhanced RAG Pipeline with:
    - Medical safety detection for high-risk symptoms
    - Symptom extraction and weighted retrieval
    - Body system classification for metadata filtering
    - Multi-stage retrieval with reranking
    """

    def __init__(self, vector_db_path, api_key, use_enhanced_retrieval=True, safety_threshold=0.60):
        if getattr(config, "USE_HYBRID_RETRIEVAL", False):
            try:
                self.retriever = HybridFusionRetriever(vector_db_path)
                print("Retriever mode: hybrid (dense + BM25 fusion)")
            except Exception as e:
                print(f"[Hybrid retriever init failed, falling back to dense retriever: {e}]")
                self.retriever = Retriever(vector_db_path)
        else:
            self.retriever = Retriever(vector_db_path)
        self.generator = Generator(api_key)
        self.reranker = CrossEncoder(config.RERANKER_MODEL)
        
        # Enhanced retrieval components
        self.use_enhanced_retrieval = use_enhanced_retrieval
        if use_enhanced_retrieval:
            # Load embedding model for symptom weighting
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            # Build frequency index from chunks for IDF weighting
            chunks_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "data", "chunks", "*.json"
            )
            self.freq_index, self.total_chunks = build_frequency_index(chunks_path)
            
            # Initialize safety engine
            self.risk_store = RiskEmbeddingStore()
            self.safety_engine = DynamicMedicalSafetyEngine(
                self.embedding_model,
                self.risk_store,
                threshold=safety_threshold
            )

    def check_safety(self, user_input):
        """
        Check user input for high-risk medical conditions.
        
        Returns:
            Tuple of (is_safe, safety_result)
        """
        if not self.use_enhanced_retrieval:
            return True, None
            
        safety_result = self.safety_engine.evaluate(user_input)
        return not safety_result["risk_detected"], safety_result

    def _classify_and_weight_query(self, question):
        """
        Enhanced query processing:
        1. Extract symptoms from user input
        2. Classify the body system
        3. Build weighted query for better retrieval
        
        Returns:
            Tuple of (weighted_query, body_system)
        """
        try:
            # Step 1: Extract symptoms
            symptom_json = extract_symptoms_llm(question, self.generator)
            
            # Step 2: Classify body system
            body_system = classify_query_system(question, self.generator)
            
            # Step 3: Build weighted query
            weighted_query = build_weighted_query(
                symptom_json,
                self.freq_index,
                self.total_chunks
            )
            
            # Fallback to original question if weighted query is empty
            if not weighted_query:
                weighted_query = question
                
            return weighted_query, body_system
            
        except Exception as e:
            print(f"[Enhanced retrieval fallback: {e}]")
            return question, "other"

    def run(self, question, memory, source_filter=None):

        conversation_history = memory.get_formatted_history()
        user_turns = memory.user_turn_count
        min_gathering = getattr(config, "MIN_GATHERING_QUESTIONS", 15)
        extra_gathering = getattr(config, "EXTRA_GATHERING_QUESTIONS_IF_UNCERTAIN", 5)
        final_diagnosis_turn = min_gathering + extra_gathering

        # Choose retrieval strategy based on turn count
        if self.use_enhanced_retrieval and user_turns == 0:
            # First turn: Use enhanced symptom-based retrieval
            retrieval_query, body_system = self._classify_and_weight_query(question)
            # print(f"[Classified system: {body_system}]")
        else:
            # Subsequent turns: Use query rewriting
            retrieval_query = rewrite_query(
                generator=self.generator,
                conversation_history=conversation_history,
                current_question=question
            )
            body_system = None

        # print(f"Retrieving initial {config.K_RETRIEVAL} chunks...")
        retrieved_chunks = self.retriever.retrieve(
            retrieval_query, 
            k=config.K_RETRIEVAL, 
            source_filter=source_filter
        )

        if not retrieved_chunks:
            return ["No relevant information found."]

        # print(f"Reranking to top {config.K_RERANK} chunks...")
        # Prepare pairs for cross-encoder reranking
        pairs = [[retrieval_query, chunk['text']] for chunk in retrieved_chunks]
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for i, chunk in enumerate(retrieved_chunks):
            chunk['rerank_score'] = scores[i]
            
        reranked_chunks = sorted(
            retrieved_chunks, 
            key=lambda x: x['rerank_score'], 
            reverse=True
        )[:config.K_RERANK]

        # Optional: Filter by body system if classified (uncomment to enable)
        # if body_system and body_system != "other":
        #     reranked_chunks = [
        #         c for c in reranked_chunks 
        #         if c.get('primary_system') == body_system
        #     ][:config.K_RERANK]

        # LOGIC FLOW
        if memory.waiting_remedies_consent:
            if any(word in question.lower() for word in ["yes", "yeah", "ok", "sure", "please", "remedies"]):
                memory.waiting_remedies_consent = False
                memory.mark_complete()
                
                # Fetch remedies specifically for the diagnosis from the knowledge base
                diag_name = "the condition"
                if memory.last_diagnosis and "DIAGNOSIS:" in memory.last_diagnosis:
                    diag_name = memory.last_diagnosis.split("DIAGNOSIS:")[1].split("\n")[0].strip()

                remedy_query = f"Ayurvedic remedies, treatments, foods, habits (do's and don'ts) for {diag_name}"
                # print(f"Searching for remedies for {diag_name}...")
                remedy_chunks = self.retriever.retrieve(remedy_query, k=config.K_RERANK, source_filter=source_filter)
                
                return self.generator.generate(remedy_query, remedy_chunks, conversation_history, mode="remedies")
            else:
                memory.waiting_remedies_consent = False
                memory.mark_complete()
                def farewell_gen():
                    yield "Please let me know if you need anything else. Goodbye!"
                return farewell_gen()

        if memory.diagnosis_complete:
            # Diagnosis is already done, just chat normally
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="final")

        if user_turns < min_gathering:
            # Phase 1: Gathering
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")
        
        elif user_turns == min_gathering:
            # After minimum required questions, attempt diagnosis
            diagnosis_report = self.generator.generate_diagnosis(conversation_history, reranked_chunks)
            print(f"\n--- DIAGNOSIS REPORT ---\n{diagnosis_report}\n------------------------")
            
            print("Verifying diagnosis...")
            is_valid = self.generator.verify_diagnosis(diagnosis_report, conversation_history)
            
            if is_valid:
                print("Verification SUCCESS. Generating final answer...")
                memory.last_diagnosis = diagnosis_report
                memory.waiting_remedies_consent = True
                return self.generator.generate(question, reranked_chunks, conversation_history, mode="diagnosis")
            else:
                print(f"Verification FAILED. Need {extra_gathering} more questions.")
                return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")

        elif min_gathering < user_turns < final_diagnosis_turn:
            # Phase 2: Extra gathering
            print(f"Extra Gathering mode: Question {user_turns - min_gathering}/{extra_gathering}")
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")
        
        else:
            # Final diagnosis after extra gathering window
            print("Final diagnosis mode reached.")
            diagnosis_report = self.generator.generate_diagnosis(conversation_history, reranked_chunks)
            memory.last_diagnosis = diagnosis_report
            memory.waiting_remedies_consent = True
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="diagnosis")
