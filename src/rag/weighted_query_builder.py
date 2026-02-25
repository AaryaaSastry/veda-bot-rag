# rag/weighted_query_builder.py

from .symptom_weighting import compute_weight


def build_weighted_query(symptom_json, freq_index, total_chunks):
    """
    Build a weighted query from extracted symptoms.
    
    Args:
        symptom_json: Dict with primary_symptoms, secondary_symptoms, systemic_symptoms
        freq_index: Counter of word frequencies
        total_chunks: Total number of chunks
    
    Returns:
        Weighted query string with symptoms repeated based on IDF weight
    """
    all_symptoms = (
        symptom_json.get("primary_symptoms", []) +
        symptom_json.get("secondary_symptoms", []) +
        symptom_json.get("systemic_symptoms", [])
    )

    if not all_symptoms:
        return ""

    weighted_text = ""

    for symptom in all_symptoms:
        if not symptom or not isinstance(symptom, str):
            continue
            
        weight = compute_weight(symptom, freq_index, total_chunks)

        # repeat proportional to weight (dynamic scaling)
        repetitions = max(1, int(weight))

        weighted_text += (" " + symptom) * repetitions

    return weighted_text.strip()
