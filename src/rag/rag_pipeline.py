import config
import json
import os
import re

from rag.retriever import Retriever
from rag.hybrid_fusion_retriever import HybridFusionRetriever
from rag.generator import Generator
from sentence_transformers import CrossEncoder, SentenceTransformer
from rag.symptom_extractor import extract_symptoms_llm
from rag.symptom_canonicalizer import canonicalize_symptoms_llm, fuzzy_canonicalizer
from rag.ollama_verifier import OllamaVerifier
from rag.emergency_detector import EmergencyDetector
from rag.symptom_weighting import build_frequency_index
from rag.weighted_query_builder import build_weighted_query
from rag.system_classifier import classify_query_system
from rag.risk_embeddings import RiskEmbeddingStore
from rag.dynamic_safety_engine import DynamicMedicalSafetyEngine
from rag.bayesian_engine import BayesianDiagnosticEngine


class RAGPipeline:
    """
    Enhanced RAG Pipeline with:
    - Medical safety detection for high-risk symptoms
    - Symptom extraction and weighted retrieval
    - Body system classification for metadata filtering
    - Multi-stage retrieval with reranking
    - Differential diagnosis + uncertainty + safety gates
    - Disease Index reasoning layer
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
        self.disease_index = []
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "disease_index.json"
        )
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                self.disease_index = json.load(f)
        
        self.bayesian_kb = self._build_bayesian_kb(self.disease_index)
        self.disease_categories = {
            entry.get("disease", "Unknown"): entry.get("category", "other").lower() 
            for entry in (self.disease_index or [])
        }
        # Pre-calc symptom difficulties (could also be stored in JSON)
        self.symptom_difficulty = self._estimate_symptom_difficulties(self.bayesian_kb)
        
        # Initialize Ollama Auditor
        self.ollama_auditor = OllamaVerifier(model_name=getattr(config, "OLLAMA_MODEL", "llama3"))
        
        # Initialize Emergency Detector
        self.emergency_detector = EmergencyDetector(self.generator)

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

    def _build_bayesian_kb(self, disease_index):
        kb = {}
        for entry in (disease_index or []):
            name = entry.get("disease", "Unknown")
            symptoms = entry.get("symptoms", [])
            # Map symptoms to fixed high probability P(S|D)
            kb[name] = {s.lower().strip(): 0.9 for s in symptoms if s}
            # Add category/system as a categorical symptom
            category = entry.get("category", "").lower()
            if category:
                kb[name][f"system_{category}"] = 0.95
        return kb

    def _estimate_symptom_difficulties(self, kb):
        """
        Assigns a heuristic difficulty to symptoms based on complexity.
        Simple logic: longer words or multiple words suggest more technical terms.
        This can be swapped for a real LLM-based labeling in a background task.
        """
        difficulties = {}
        all_symptoms = set()
        for d_syms in kb.values():
            all_symptoms.update(d_syms.keys())
            
        for sym in all_symptoms:
            # Heuristic: 
            # 1.0 = simple (1 word, < 6 chars)
            # 1.5 = moderate (2 words or > 6 chars)
            # 2.5 = hard (3+ words or very long words - technical terms)
            words = sym.split()
            if len(words) >= 3 or len(sym) > 15:
                difficulties[sym] = 2.5
            elif len(words) == 2 or len(sym) > 8:
                difficulties[sym] = 1.6
            else:
                difficulties[sym] = 1.0
        return difficulties

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
        value = (text or "").strip().lower()
        affirmative_tokens = {
            "yes", "y", "yeah", "yep", "ok", "okay", "sure", "please", "proceed",
            "continue", "go ahead", "remedies", "home remedies",
        }
        return any(token in value for token in affirmative_tokens)

    def _is_negative(self, text):
        value = (text or "").strip().lower()
        negative_tokens = {"no", "n", "nope", "not now", "later", "stop", "exit"}
        return any(token in value for token in negative_tokens)

    def _build_reranked_chunks(self, retrieval_query, source_filter=None, metadata_filter=None):
        retrieved_chunks = self.retriever.retrieve(
            retrieval_query,
            k=config.K_RETRIEVAL,
            source_filter=source_filter,
            metadata_filter=metadata_filter
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

    def _extract_diagnosis_name(self, diagnosis_report):
        diag_name = "the condition"
        if diagnosis_report and "DIAGNOSIS:" in diagnosis_report:
            diag_name = diagnosis_report.split("DIAGNOSIS:")[1].split("\n")[0].strip()
        return diag_name

    def _generate_diagnosis_and_remedies(self, report_payload, reranked_chunks, conversation_history, source_filter=None):
        def combined_gen():
            diagnosis_output = ""
            for chunk in self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="diagnosis"):
                diagnosis_output += chunk
                yield chunk

            diag_name = self._extract_diagnosis_name(diagnosis_output)
            remedy_query = f"Ayurvedic conservative remedies, lifestyle support and precautions for {diag_name}"
            remedy_chunks = self.retriever.retrieve(remedy_query, k=config.K_RERANK, source_filter=source_filter)
            if remedy_chunks:
                yield "\n\n"
                for chunk in self.generator.generate(remedy_query, remedy_chunks, conversation_history, mode="remedies"):
                    yield chunk
        return combined_gen()

    def _is_rejected_by_verifier(self, self_check):
        return (
            not self_check.get("diagnosis_valid", True)
            or not self_check.get("supported_by_chunks", True)
        )

    def _prepare_verification_plan(self, self_check):
        reasons = list(self_check.get("rejection_reasons", []))
        alternatives = list(self_check.get("alternative_conditions", []))
        questions = list(self_check.get("targeted_questions", []))

        while len(reasons) < 5:
            reasons.append("Current evidence does not sufficiently support the proposed diagnosis.")
        reasons = reasons[:5]

        if not alternatives:
            alternatives = ["Condition pattern A", "Condition pattern B", "Condition pattern C"]
        alternatives = alternatives[:3]

        max_questions = max(1, int(getattr(config, "MAX_VERIFICATION_TARGETED_QUESTIONS", 3)))
        questions = [q if q.endswith("?") else f"{q}?" for q in questions if q]
        if not questions:
            questions = ["Can you share one additional symptom detail that is still unclear?"]
        questions = questions[:max_questions]
        return reasons, alternatives, questions

    def _verification_intro_response(self, memory):
        reasons_block = "\n".join([f"- {reason}" for reason in memory.verification_reasons])
        alternatives_block = "\n".join([f"- {name}" for name in memory.verification_alternatives])
        first_question = memory.verification_questions[0] if memory.verification_questions else "Can you share one more specific symptom detail?"
        return (
            "--- USER-FRIENDLY OUTPUT ---\n"
            "VERIFICATION RESULT: Previous diagnosis is not reliable yet.\n\n"
            "WHY IT WAS REJECTED:\n"
            f"{reasons_block}\n\n"
            "OTHER POSSIBLE CONDITIONS:\n"
            f"{alternatives_block}\n\n"
            f"TARGETED QUESTION 1: {first_question}\n"
            "--- END USER OUTPUT ---"
        )

    def _verification_next_question_response(self, memory):
        idx = memory.verification_question_index
        question_number = idx + 1
        next_question = memory.verification_questions[idx]
        return (
            "--- USER-FRIENDLY OUTPUT ---\n"
            f"TARGETED QUESTION {question_number}: {next_question}\n"
            "--- END USER OUTPUT ---"
        )

    def _run_differential_decision(self, question, reranked_chunks, conversation_history, memory, final_attempt=False, source_filter=None):
        red_flags = self._detect_pre_diagnosis_red_flags(conversation_history)
        if red_flags:
            memory.escalation_active = True
            memory.waiting_remedies_consent = False
            memory.verification_active = False
            memory.last_diagnosis = f"DIAGNOSIS: Escalation advised\nREASONING: Red flags present: {', '.join(red_flags)}"
            self._trigger_verification("pre_diagnosis_escalation", question, conversation_history)
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="escalation")

        # --- Bayesian Hypothesis Generation ---
        # Initialize engine with the KB built from the index
        engine = BayesianDiagnosticEngine(self.bayesian_kb)
        # Restore priors and observations from memory to keep the state across turns
        if memory.bayesian_priors:
            engine.priors = memory.bayesian_priors
        engine.observations = memory.bayesian_observations

        # Convert engine state to a report payload compatible with the existing generator
        state = engine.get_diagnosis_state()
        
        # Build a "pseudo-differential" report from Bayesian probabilities
        differential = {
            "possible_conditions": [
                {
                    "name": d["disease"], 
                    "confidence": d["prob"],
                    "evidence_for": ["Supported by Bayesian likelihood over observed symptoms"],
                    "evidence_against": []
                } for d in state["differential_diagnoses"]
            ],
            "most_likely": state["top_diagnosis"],
            "most_likely_confidence": state["confidence"],
            "uncertainty_level": "low" if state["confidence"] > 0.7 else ("moderate" if state["confidence"] > 0.4 else "high"),
            "red_flags_present": [],
            "reasoning_summary": f"Calculated via Bayesian update with entropy {state['entropy']:.4f}."
        }

        # Still perform a self-check but now using the deterministic state
        self_check = self.generator.self_check_differential(differential, conversation_history, retrieved_chunks=reranked_chunks)

        confidence_cap = self_check.get("adjusted_confidence_cap", 1.0)
        differential["most_likely_confidence"] = min(
            differential.get("most_likely_confidence", 0.0),
            confidence_cap,
        )
        if self_check.get("requires_medical_escalation"):
            differential["red_flags_present"] = list(set(differential.get("red_flags_present", []) + ["auditor_escalation"]))

        if self._is_rejected_by_verifier(self_check):
            max_cycles = max(1, int(getattr(config, "MAX_VERIFICATION_REFINEMENT_CYCLES", 2)))
            memory.last_diagnosis = None
            if memory.verification_attempts >= max_cycles:
                memory.escalation_active = True
                memory.waiting_remedies_consent = False
                memory.verification_active = False
                self._trigger_verification("verification_exhausted_escalation", question, conversation_history)
                return self.generator.generate(question, reranked_chunks, conversation_history, mode="escalation")

            reasons, alternatives, questions = self._prepare_verification_plan(self_check)
            memory.verification_active = True
            memory.verification_reasons = reasons
            memory.verification_alternatives = alternatives
            memory.verification_questions = questions
            memory.verification_question_index = 0
            self._trigger_verification("diagnosis_rejected_start_refinement", question, conversation_history)

            def verification_intro_gen():
                yield self._verification_intro_response(memory)
            return verification_intro_gen()

        diagnosis_report = self.generator.format_differential_report(differential)
        
        # --- NEW: Ollama Auditing Layer (Algorithmic Verification) ---
        audit_result = self.ollama_auditor.verify(
            user_symptoms=memory.bayesian_observations,
            diagnosis_report=diagnosis_report,
            retrieved_chunks=reranked_chunks
        )
        
        if not audit_result.get("is_consistent", True):
            differential["red_flags_present"].extend(audit_result.get("contradictions_found", []))
            differential["reasoning_summary"] += f" [AUDIT ALERT: {audit_result.get('auditor_notes')}]"
            
        if audit_result.get("safety_risk_level") in ["high", "critical"]:
             differential["red_flags_present"].append("audit_safety_risk")

        memory.last_diagnosis = diagnosis_report

        confidence_threshold = getattr(config, "DIAGNOSIS_CONFIDENCE_THRESHOLD", 0.70)
        low_confidence = differential.get("most_likely_confidence", 0.0) < confidence_threshold
        high_uncertainty = differential.get("uncertainty_level") == "high"
        escalation_required = bool(differential.get("red_flags_present")) or self_check.get("requires_medical_escalation", False)

        report_payload = f"Differential report: {json.dumps(differential, ensure_ascii=False)}"

        if escalation_required:
            memory.escalation_active = True
            memory.waiting_remedies_consent = False
            memory.verification_active = False
            memory.last_diagnosis = None
            self._trigger_verification("differential_escalation", question, conversation_history)
            return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="escalation")

        memory.verification_active = False
        memory.verification_attempts = 0
        memory.verification_questions = []
        memory.verification_reasons = []
        memory.verification_alternatives = []
        memory.verification_question_index = 0
        memory.waiting_remedies_consent = False
        self._trigger_verification("diagnosis", question, conversation_history)
        if getattr(config, "AUTO_PRINT_REMEDIES_AFTER_VERIFIED_DIAGNOSIS", True):
            memory.mark_complete()
            return self._generate_diagnosis_and_remedies(
                report_payload,
                reranked_chunks,
                conversation_history,
                source_filter=source_filter,
            )

        if low_confidence or high_uncertainty or self_check.get("overconfident") or self_check.get("missing_differentials"):
            memory.waiting_remedies_consent = True
            if final_attempt:
                self._trigger_verification("uncertain_final", question, conversation_history)
                return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="uncertain_final")
            self._trigger_verification("uncertain", question, conversation_history)
            return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="uncertain")

        memory.waiting_remedies_consent = True
        return self.generator.generate(report_payload, reranked_chunks, conversation_history, mode="diagnosis")

    def run(self, question, memory, source_filter=None):
        conversation_history = memory.get_formatted_history()
        # --- NEW: Immediate Emergency Safety Check ---
        emergency_status = self.emergency_detector.evaluate(question, conversation_history)
        if emergency_status.get("is_emergency", False):
            # Terminate diagnostic cycle and output emergency message
            memory.escalation_active = True
            msg = (
                "--- EMERGENCY ALERT ---\n"
                f"POTENTIAL EMERGENCY DETECTED: {emergency_status.get('emergency_type', 'Medical Crisis')}\n"
                f"URGENCY: {emergency_status.get('urgency_score', 1.0)*100:.0f}%\n"
                f"ACTION: {emergency_status.get('recommended_action', 'SEEK MEDICAL CARE IMMEDIATELY')}\n"
                "--- END EMERGENCY ALERT ---"
            )
            def emergency_gen():
                yield msg
            return emergency_gen()

        # 2. CONVERSATION MEMORY & SYMPTOM EXTRACTOR
        symptom_json = extract_symptoms_llm(question, self.generator)
        
        # 3. BODY SYSTEM CLASSIFIER (Early filtering / RAG Query Prep)
        retrieval_query, body_system = self._classify_and_weight_query(question)

        # 4. INITIALIZE BAYESIAN ENGINE & RESTORE PRIOR STATE
        engine = BayesianDiagnosticEngine(
            self.bayesian_kb, 
            self.disease_categories, 
            self.symptom_difficulty
        )
        if memory.bayesian_priors:
            engine.priors = memory.bayesian_priors
        
        # 5. BODY SYSTEM PRE-FILTERING (Candidate Disease Filter)
        if body_system and body_system != "other":
            engine.filter_by_system(body_system)
            print(f"[Bayesian Engine: Filtered scope to '{body_system}' - {len(engine.diseases)} diseases active]")

        # 6. SYMPTOM CANONICALIZER & BAYESIAN UPDATE (Current Turn)
        kb_symptoms = engine.all_symptoms
        current_raw_symptoms = []
        for s_list in ["primary_symptoms", "secondary_symptoms", "systemic_symptoms"]:
            current_raw_symptoms.extend(symptom_json.get(s_list, []))

        mapping = canonicalize_symptoms_llm(current_raw_symptoms, kb_symptoms, self.generator)
        
        for raw_s in current_raw_symptoms:
            raw_s_clean = raw_s.lower().strip()
            # Try LLM mapping then fuzzy fallback
            canonical_s = mapping.get(raw_s, mapping.get(raw_s_clean))
            if not canonical_s:
                canonical_s = fuzzy_canonicalizer(raw_s_clean, kb_symptoms)
            
            if canonical_s:
                engine.record_observation(canonical_s, True)
                memory.bayesian_observations[canonical_s] = True

        # System boost
        if body_system and body_system != "other":
            system_sym = f"system_{body_system}"
            engine.record_observation(system_sym, True)
            memory.bayesian_observations[system_sym] = True
        
        # Save updated priors to Memory
        memory.bayesian_priors = engine.priors

        # 7. RAG RETRIEVER & CHUNK SEARCH
        metadata_filter = {"category": body_system} if body_system and body_system != "other" else None
        reranked_chunks = self._build_reranked_chunks(
            retrieval_query, 
            source_filter=source_filter,
            metadata_filter=metadata_filter
        )
        
        if not reranked_chunks:
            self._trigger_verification("no_retrieval_chunks", question, conversation_history)
            return ["No relevant information found matching this symptom profile."]

        # 8. DECISION: DIAGNOSIS VS QUESTION SELECTION
        state = engine.get_diagnosis_state()
        confidence_threshold = getattr(config, "DIAGNOSIS_CONFIDENCE_THRESHOLD", 0.85)
        
        if state["confidence"] >= confidence_threshold or memory.user_turn_count >= 15:
            # 9. FINAL DIAGNOSIS + EXPLANATION
            return self._run_differential_decision(
                question, 
                reranked_chunks, 
                conversation_history, 
                memory,
                source_filter=source_filter
            )
        else:
            # 10. INFORMATION GAIN QUESTION SELECTION & DIFFICULTY SCORING
            best_symptom, gain = engine.select_best_question(engine.priors)
            
            if not best_symptom:
                # Fallback to diagnosis if no better questions exist
                return self._run_differential_decision(question, reranked_chunks, conversation_history, memory)
                
            # 11. LLM NATURAL LANGUAGE WRAPPER
            prompt = (
                f"The algorithm needs to verify if the user has '{best_symptom}' to improve diagnostic confidence. "
                "Context: The user is in an Ayurvedic diagnostic session. "
                f"Textbook knowledge shows this symptom helps differentiate conditions in the {body_system or 'general'} system. "
                "Task: Phras a polite, concise Ayurvedic diagnostic question to ask the user if they have this symptom. "
                "Avoid complex jargon. Ask only about this ONE symptom."
            )
            
            def gathering_gen():
                for chunk in self.generator.generate_text(prompt, stream=True):
                    yield chunk
            return gathering_gen()

        if memory.verification_active:
            memory.verification_question_index += 1
            if memory.verification_question_index < len(memory.verification_questions):
                self._trigger_verification("verification_followup_question", question, conversation_history)

                def followup_question_gen():
                    yield self._verification_next_question_response(memory)
                return followup_question_gen()

            memory.verification_active = False
            memory.verification_attempts += 1
            memory.verification_question_index = 0
            self._trigger_verification("verification_refinement_recheck", question, conversation_history)
            return self._run_differential_decision(
                question,
                reranked_chunks,
                conversation_history,
                memory,
                final_attempt=False,
                source_filter=source_filter,
            )

        if memory.waiting_treatment_risk_profile:
            memory.waiting_treatment_risk_profile = False
            memory.treatment_risk_profile_collected = True
            memory.mark_complete()
            diag_name = "the condition"
            if memory.last_diagnosis and "DIAGNOSIS:" in memory.last_diagnosis:
                diag_name = memory.last_diagnosis.split("DIAGNOSIS:")[1].split("\n")[0].strip()
            remedy_query = f"Ayurvedic conservative remedies, lifestyle support and precautions for {diag_name}"
            remedy_chunks = self.retriever.retrieve(remedy_query, k=config.K_RERANK, source_filter=source_filter)
            self._trigger_verification("treatment_risk_profile_collected", question, conversation_history)
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
                self._trigger_verification("remedies_consent_yes", question, conversation_history)
                return self.generator.generate(remedy_query, remedy_chunks, conversation_history, mode="remedies")

            if not self._is_negative(question):
                self._trigger_verification("remedies_consent_clarification", question, conversation_history)
                return self.generator.generate(question, reranked_chunks, conversation_history, mode="consent_clarification")

            memory.waiting_remedies_consent = False
            memory.mark_complete()
            self._trigger_verification("remedies_consent_no", question, conversation_history)

            def farewell_gen():
                yield "Please let me know if you need anything else. Goodbye!"
            return farewell_gen()

        if memory.escalation_active:
            self._trigger_verification("escalation_followup", question, conversation_history)
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="escalation_followup")

        if memory.diagnosis_complete:
            self._trigger_verification("post_diagnosis_followup", question, conversation_history)
            return self.generator.generate(question, reranked_chunks, conversation_history, mode="final")

        # The core diagnostic logic is now handled in the first block of the run method.
        # This catch-all ensures we don't fall into old gathering/diagnosis branches.
        return self._run_differential_decision(question, reranked_chunks, conversation_history, memory)
