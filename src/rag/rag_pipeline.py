import config
import json
import os
import re

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
    - Differential diagnosis + uncertainty + safety gates
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

        self.use_enhanced_retrieval = use_enhanced_retrieval
        if use_enhanced_retrieval:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            chunks_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "chunks", "*.json"
            )
            self.freq_index, self.total_chunks = build_frequency_index(chunks_path)
            self.risk_store = RiskEmbeddingStore()
            self.safety_engine = DynamicMedicalSafetyEngine(
                self.embedding_model,
                self.risk_store,
                threshold=safety_threshold
            )

    def check_safety(self, user_input):
        if not self.use_enhanced_retrieval:
            return True, None
        safety_result = self.safety_engine.evaluate(user_input)
        return not safety_result["risk_detected"], safety_result

    def _classify_and_weight_query(self, question):
        try:
            symptom_json = extract_symptoms_llm(question, self.generator)
            body_system = classify_query_system(question, self.generator)
            weighted_query = build_weighted_query(
                symptom_json,
                self.freq_index,
                self.total_chunks
            )
            if not weighted_query:
                weighted_query = question
            return weighted_query, body_system
        except Exception as e:
            print(f"[Enhanced retrieval fallback: {e}]")
            return question, "other"

    def _is_affirmative(self, text):
        value = (text or "").lower()
        return any(word in value for word in ["yes", "yeah", "ok", "sure", "please", "remedies"])

    def _extract_age(self, conversation_history):
        age_match = re.search(r"\b(\d{1,3})\b", conversation_history or "")
        if not age_match:
            return None
        try:
            age = int(age_match.group(1))
            if 0 < age < 120:
                return age
        except Exception:
            pass
        return None

    def _detect_pre_diagnosis_red_flags(self, conversation_history):
        text = (conversation_history or "").lower()
        flags = []
        keyword_groups = {
            "severe_or_worst_pain": ["severe pain", "worst pain", "excruciating", "unbearable"],
            "sudden_onset": ["sudden", "abrupt", "all of a sudden"],
            "functional_loss": ["cannot walk", "can't walk", "weakness", "numbness", "fainting", "collapse"],
            "systemic_symptoms": ["high fever", "persistent fever", "weight loss", "night sweats", "vomiting blood", "blood in stool"],
        }
        for flag, keys in keyword_groups.items():
            if any(k in text for k in keys):
                flags.append(flag)

        age = self._extract_age(conversation_history)
        if age and age > 50 and ("new joint pain" in text or ("joint pain" in text and "new" in text)):
            flags.append("age_over_50_new_joint_pain")
        return flags

    def _build_reranked_chunks(self, retrieval_query, source_filter=None):
        retrieved_chunks = self.retriever.retrieve(
            retrieval_query,
            k=config.K_RETRIEVAL,
            source_filter=source_filter
        )
        if not retrieved_chunks:
            return []

        pairs = [[retrieval_query, chunk["text"]] for chunk in retrieved_chunks]
        scores = self.reranker.predict(pairs)
        for i, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = scores[i]
        reranked_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:config.K_RERANK]
        return reranked_chunks

    def _run_differential_decision(self, question, reranked_chunks, conversation_history, memory, final_attempt=False):
        red_flags = self._detect_pre_diagnosis_red_flags(conversation_history)
        if red_flags:
            memory.last_diagnosis = f"DIAGNOSIS: Escalation advised\nREASONING: Red flags present: {', '.join(red_flags)}"
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="escalation")

        differential = self.generator.generate_differential_diagnosis(conversation_history, reranked_chunks)
        self_check = self.generator.self_check_differential(differential, conversation_history)

        confidence_cap = self_check.get("adjusted_confidence_cap", 1.0)
        differential["most_likely_confidence"] = min(
            differential.get("most_likely_confidence", 0.0),
            confidence_cap,
        )
        if self_check.get("requires_medical_escalation"):
            differential["red_flags_present"] = list(set(differential.get("red_flags_present", []) + ["auditor_escalation"]))

        diagnosis_report = self.generator.format_differential_report(differential)
        memory.last_diagnosis = diagnosis_report

        confidence_threshold = getattr(config, "DIAGNOSIS_CONFIDENCE_THRESHOLD", 0.70)
        low_confidence = differential.get("most_likely_confidence", 0.0) < confidence_threshold
        high_uncertainty = differential.get("uncertainty_level") == "high"
        escalation_required = bool(differential.get("red_flags_present")) or self_check.get("requires_medical_escalation", False)

        report_payload = f"Differential report: {json.dumps(differential, ensure_ascii=False)}"

        if escalation_required:
            return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="escalation")

        if low_confidence or high_uncertainty or self_check.get("overconfident") or self_check.get("missing_differentials"):
            if final_attempt:
                memory.mark_complete()
                return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="uncertain_final")
            return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="uncertain")

        memory.waiting_remedies_consent = True
        return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="diagnosis")

    def run(self, question, memory, source_filter=None):
        conversation_history = memory.get_formatted_history()
        user_turns = memory.user_turn_count
        min_gathering = getattr(config, "MIN_GATHERING_QUESTIONS", 15)
        extra_gathering = getattr(config, "EXTRA_GATHERING_QUESTIONS_IF_UNCERTAIN", 5)
        final_diagnosis_turn = min_gathering + extra_gathering

        if self.use_enhanced_retrieval and user_turns == 1:
            retrieval_query, _ = self._classify_and_weight_query(question)
        else:
            retrieval_query = rewrite_query(
                generator=self.generator,
                conversation_history=conversation_history,
                current_question=question
            )

        reranked_chunks = self._build_reranked_chunks(retrieval_query, source_filter=source_filter)
        if not reranked_chunks:
            return ["No relevant information found."]

        if memory.waiting_treatment_risk_profile:
            memory.waiting_treatment_risk_profile = False
            memory.treatment_risk_profile_collected = True
            memory.mark_complete()
            diag_name = "the condition"
            if memory.last_diagnosis and "DIAGNOSIS:" in memory.last_diagnosis:
                diag_name = memory.last_diagnosis.split("DIAGNOSIS:")[1].split("\n")[0].strip()
            remedy_query = f"Ayurvedic conservative remedies, lifestyle support and precautions for {diag_name}"
            remedy_chunks = self.retriever.retrieve(remedy_query, k=config.K_RERANK, source_filter=source_filter)
            return self.generator.generate(remedy_query, remedy_chunks, conversation_history, mode="remedies")

        if memory.waiting_remedies_consent:
            if self._is_affirmative(question):
                memory.waiting_remedies_consent = False
                if getattr(config, "ENABLE_TREATMENT_RISK_GATE", True) and not memory.treatment_risk_profile_collected:
                    memory.waiting_treatment_risk_profile = True
                    return self.generator.generate(question, reranked_chunks, conversation_history, mode="risk_gate_question")

                memory.mark_complete()
                diag_name = "the condition"
                if memory.last_diagnosis and "DIAGNOSIS:" in memory.last_diagnosis:
                    diag_name = memory.last_diagnosis.split("DIAGNOSIS:")[1].split("\n")[0].strip()
                remedy_query = f"Ayurvedic conservative remedies, lifestyle support and precautions for {diag_name}"
                remedy_chunks = self.retriever.retrieve(remedy_query, k=config.K_RERANK, source_filter=source_filter)
                return self.generator.generate(remedy_query, remedy_chunks, conversation_history, mode="remedies")

            memory.waiting_remedies_consent = False
            memory.mark_complete()

            def farewell_gen():
                yield "Please let me know if you need anything else. Goodbye!"
            return farewell_gen()

        if memory.diagnosis_complete:
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="final")

        if user_turns < min_gathering:
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")

        if user_turns == min_gathering:
            return self._run_differential_decision(question, reranked_chunks, conversation_history, memory, final_attempt=False)

        if min_gathering < user_turns < final_diagnosis_turn:
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="gathering")

        return self._run_differential_decision(question, reranked_chunks, conversation_history, memory, final_attempt=True)
