import numpy as np
import json
from typing import Dict, List, Tuple

class BayesianDiagnosticEngine:
    def __init__(self, disease_kb: Dict[str, Dict[str, float]], disease_metadata: Dict[str, str] = None, symptom_difficulty: Dict[str, float] = None, confidence_threshold: float = 0.85):
        """
        disease_kb: { "disease_name": { "symptom_name": p_symptom_given_disease } }
        disease_metadata: { "disease_name": "category" }
        symptom_difficulty: { "symptom_name": difficulty_score (1.0 to 3.0) }
        """
        self.full_kb = disease_kb
        self.disease_metadata = disease_metadata or {}
        self.symptom_difficulty = symptom_difficulty or {}
        self.threshold = confidence_threshold
        
        # Initial active scope is everything
        self.diseases = sorted(list(self.full_kb.keys()))
        self.kb = {d: self.full_kb[d] for d in self.diseases}
        self.all_symptoms = sorted(list(set(s for d in self.kb.values() for s in d.keys())))
        
        # Initialize priors (Uniform)
        self.priors = {d: 1.0 / len(self.diseases) for d in self.diseases}
        self.observations = {} # {symptom: True/False}

    def filter_by_system(self, system: str):
        """Restricts logical scope to a specific body system."""
        if not system or system == "other" or not self.disease_metadata:
            return

        filtered_diseases = [
            d for d, cat in self.disease_metadata.items() 
            if cat.lower() == system.lower()
        ]
        
        if filtered_diseases:
            self.diseases = sorted(filtered_diseases)
            self.kb = {d: self.full_kb[d] for d in self.diseases}
            self.all_symptoms = sorted(list(set(s for d in self.kb.values() for s in d.keys())))
            
            # Re-normalize priors within the new scope to preserve relative weights
            current_sum = sum(self.priors.get(d, 0) for d in self.diseases)
            if current_sum > 0:
                self.priors = {d: self.priors.get(d, 0) / current_sum for d in self.diseases}
            else:
                # Reset to uniform if all previously scoped were zeroed (rare)
                self.priors = {d: 1.0 / len(self.diseases) for d in self.diseases}

    def calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """H(D) = -Σ P(d) log2 P(d)"""
        probs = np.array(list(probabilities.values()))
        probs = probs[probs > 0] # Avoid log(0)
        return -np.sum(probs * np.log2(probs))

    def update_probabilities(self, current_priors: Dict[str, float], symptom: str, observed: bool) -> Dict[str, float]:
        """
        P(D|S) = [P(S|D) * P(D)] / P(S)
        Where P(S) = Σ [P(S|Di) * P(Di)]
        """
        new_probs = {}
        total_evidence = 0
        
        # Precision constants to avoid zeroing out diagnoses entirely
        EPSILON = 0.01  # Minimum probability for a symptom not mentioned but possibly present
        MISSING_PENALTY = 0.05 # Likelihood of symptom being present if not in KB for this disease
        
        for d in self.diseases:
            # P(S|D): Probability of symptom given disease
            # If symptom is in KB, use its value (usually ~0.9 for primary symptoms)
            # If not in KB, assume a low baseline probability
            p_s_given_d = self.kb.get(d, {}).get(symptom, MISSING_PENALTY)
            
            if not observed:
                # Likelihood of symptom NOT being present given disease: P(~S|D) = 1 - P(S|D)
                likelihood = 1.0 - p_s_given_d
            else:
                # Likelihood of symptom being present given disease: P(S|D)
                likelihood = p_s_given_d
            
            # Bayes Numerator: P(S|D) * P(D)
            new_probs[d] = likelihood * current_priors.get(d, 1.0/len(self.diseases))
            total_evidence += new_probs[d]
            
        # Normalization: Divide by P(S) which is 'total_evidence'
        if total_evidence <= 0: 
            return current_priors 
        
        normalized_probs = {d: (p / total_evidence) for d, p in new_probs.items()}
        
        # Laplace smoothing to prevent absolute zero probabilities
        # ensures we never completely rule out a condition based on a single negative observation
        total_after_smoothing = 0
        for d in normalized_probs:
            normalized_probs[d] = max(normalized_probs[d], 1e-6)
            total_after_smoothing += normalized_probs[d]
            
        return {d: p / total_after_smoothing for d, p in normalized_probs.items()}

    def get_expected_entropy(self, current_priors: Dict[str, float], symptom: str) -> float:
        """E[H] = P(yes) * H(yes) + P(no) * H(no)"""
        # P(yes) = Σ P(S|Di) * P(Di)
        p_yes = sum(self.kb.get(d, {}).get(symptom, 0.05) * current_priors[d] for d in self.diseases)
        p_no = 1.0 - p_yes
        
        if p_yes == 0 or p_no == 0:
            return self.calculate_entropy(current_priors)
            
        h_yes = self.calculate_entropy(self.update_probabilities(current_priors, symptom, True))
        h_no = self.calculate_entropy(self.update_probabilities(current_priors, symptom, False))
        
        return (p_yes * h_yes) + (p_no * h_no)

    def select_best_question(self, current_priors: Dict[str, float]) -> Tuple[str, float]:
        """Choose symptom maximizing Information Gain weighted by Difficulty (IG/Difficulty)"""
        current_h = self.calculate_entropy(current_priors)
        best_symptom = None
        max_score = -1.0
        
        remaining_symptoms = [s for s in self.all_symptoms if s not in self.observations]
        
        for s in remaining_symptoms:
            expected_h = self.get_expected_entropy(current_priors, s)
            ig = max(0, current_h - expected_h)
            
            # Difficulty score: 1.0 (easy) to 3.0 (very hard technical term)
            # Default to 1.5 if unknown
            difficulty = self.symptom_difficulty.get(s, 1.5)
            
            # We want high IG and low difficulty
            score = ig / difficulty
            
            if score > max_score:
                max_score = score
                best_symptom = s
                
        return best_symptom, max_score

    def get_diagnosis_state(self):
        sorted_probs = sorted(self.priors.items(), key=lambda x: x[1], reverse=True)
        return {
            "top_diagnosis": sorted_probs[0][0],
            "confidence": sorted_probs[0][1],
            "differential_diagnoses": [{"disease": d, "prob": p} for d, p in sorted_probs[:5]],
            "entropy": self.calculate_entropy(self.priors),
            "complete": sorted_probs[0][1] >= self.threshold or len(self.observations) >= 10
        }

    def record_observation(self, symptom: str, observed: bool):
        """Record an observation and update priors."""
        # Check if the symptom should be mapped via fuzzy matching first
        matched_symptom = None
        if symptom in self.all_symptoms:
            matched_symptom = symptom
        else:
            # Simple fallback fuzzy logic inside record_observation
            s_low = symptom.lower().strip()
            for s in self.all_symptoms:
                if s_low in s or s in s_low:
                    matched_symptom = s
                    break
                    
        if matched_symptom:
            self.observations[matched_symptom] = observed
            self.priors = self.update_probabilities(self.priors, matched_symptom, observed)
        else:
            # If still not found, we don't update priors as it would be noise
            # unless it's a categorical boost system_...
            if symptom.startswith("system_"):
                self.observations[symptom] = observed
                self.priors = self.update_probabilities(self.priors, symptom, observed)
