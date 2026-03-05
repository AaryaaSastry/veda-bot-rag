import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rag.bayesian_engine import BayesianDiagnosticEngine

def build_kb_from_index(index_path):
    """
    Transform Disease Index into a Bayesian Knowledge Base P(Symptom|Disease).
    Mapping semantic descriptions to probabilistic weights.
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        disease_index = json.load(f)
    
    kb = {}
    for entry in disease_index:
        disease_name = entry.get("disease", "Unknown")
        symptoms = entry.get("symptoms", [])
        
        # In a real system, we'd map these to a standard symptom set.
        # Here we assign a high P(S|D) for mentioned symptoms (e.g., 0.9)
        # and a baseline epsilon for unmentioned ones.
        kb[disease_name] = {s.lower().strip(): 0.9 for idx, s in enumerate(symptoms)}
        
        # Also add Dosha/Srotas as categorical symptoms
        if entry.get("dosha"):
            kb[disease_name][f"dosha_{entry['dosha'].lower()}"] = 0.95
            
    return kb

def simulate_diagnostic_session():
    # 1. Load context from existing index
    index_path = os.path.join("data", "disease_index.json")
    if not os.path.exists(index_path):
        print("Run disease_index_builder.py first!")
        return
        
    kb = build_kb_from_index(index_path)
    # Filter KB to a subset for the demo
    demo_diseases = list(kb.keys())[:10]
    demo_kb = {d: kb[d] for d in demo_diseases}
    
    engine = BayesianDiagnosticEngine(demo_kb)
    
    print("--- STARTING BAYESIAN DIAGNOSTIC SESSION ---")
    print(f"Candidates: {len(demo_diseases)}")
    
    step = 1
    while not engine.get_diagnosis_state()["complete"]:
        state = engine.get_diagnosis_state()
        print(f"\nStep {step}: Current Top: {state['top_diagnosis']} (Conf: {state['confidence']:.2%}, Entropy: {state['entropy']:.2f})")
        
        best_symptom, ig = engine.select_best_question(engine.priors)
        if not best_symptom or ig <= 0:
            print("No more information gain possible.")
            break
            
        print(f"Algorithm Selected Question: 'Presence of {best_symptom}?' (IG: {ig:.4f})")
        
        # Simulation: User says 'yes' if the symptom exists in the first disease
        target_disease = demo_diseases[0]
        user_answer = best_symptom in demo_kb[target_disease]
        print(f"User Answer (Simulated): {'Yes' if user_answer else 'No'}")
        
        engine.record_observation(best_symptom, user_answer)
        step += 1
        
    final_state = engine.get_diagnosis_state()
    print("\n--- FINAL DIAGNOSIS ---")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    simulate_diagnostic_session()
