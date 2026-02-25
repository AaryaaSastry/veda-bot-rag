from sentence_transformers import SentenceTransformer
import numpy as np

class RiskEmbeddingStore:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.risk_concepts = {

    # --- Existing ---
    "vascular_infection": "limb swelling with systemic infection",
    "deep_vein_thrombosis": "unilateral leg swelling with clot risk",
    "cardiac_emergency": "chest pain with possible cardiac cause",
    "severe_infection": "high persistent fever with systemic symptoms",

    # --- Respiratory Emergencies ---
    "pulmonary_embolism": "sudden shortness of breath with chest discomfort and clot risk",
    "respiratory_failure": "severe breathing difficulty with low oxygen symptoms",
    "pneumonia_sepsis": "lung infection with systemic inflammatory response",

    # --- Neurological Emergencies ---
    "acute_stroke": "sudden neurological deficit with weakness or speech difficulty",
    "intracranial_event": "sudden severe headache with neurological symptoms",
    "altered_mental_status": "confusion or reduced consciousness with systemic cause",

    # --- Cardiovascular ---
    "heart_failure_exacerbation": "leg swelling with shortness of breath and fluid overload",
    "aortic_dissection": "sudden severe chest or back pain with vascular instability",
    "arrhythmia_instability": "palpitations with dizziness or fainting",

    # --- Abdominal Emergencies ---
    "acute_abdomen": "severe abdominal pain with guarding or systemic symptoms",
    "appendicitis_pattern": "localized abdominal pain with fever and inflammation",
    "bowel_obstruction": "abdominal distension with vomiting and inability to pass stool",

    # --- Infectious Critical ---
    "sepsis_pattern": "systemic infection with fever and organ dysfunction symptoms",
    "meningitis_pattern": "fever with neck stiffness and altered mental status",

    # --- Metabolic Emergencies ---
    "diabetic_ketoacidosis": "high blood sugar with dehydration and altered breathing",
    "hypoglycemic_event": "low blood sugar with confusion or loss of consciousness",

    # --- Vascular Limb Threat ---
    "acute_limb_ischemia": "sudden limb pain with coldness and reduced circulation",
    "compartment_syndrome": "severe limb pain with swelling and neurovascular compromise",

    # --- Pregnancy Related ---
    "ectopic_pregnancy": "early pregnancy with abdominal pain and internal bleeding risk",
    "preeclampsia_pattern": "pregnancy with high blood pressure and swelling",

    # --- Allergic / Immune ---
    "anaphylaxis": "allergic reaction with breathing difficulty and swelling",
    "angioedema_airway": "facial or throat swelling threatening airway",

    # --- Hemorrhage ---
    "internal_bleeding": "unexplained weakness with signs of blood loss",
    "gastrointestinal_bleeding": "vomiting blood or black stools with weakness",

    # --- Systemic Collapse ---
    "shock_state": "low blood pressure with dizziness and organ hypoperfusion",
    "multi_organ_failure": "systemic deterioration with multiple organ involvement"
}

        self.embeddings = self._encode_concepts()

    def _encode_concepts(self):
        concepts = list(self.risk_concepts.values())
        vectors = self.model.encode(concepts)
        return {
            name: vectors[i]
            for i, name in enumerate(self.risk_concepts.keys())
        }