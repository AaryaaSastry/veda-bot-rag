import numpy as np

class DynamicMedicalSafetyEngine:

    def __init__(self, embedding_model, risk_store, threshold=0.65):
        self.model = embedding_model
        self.risk_store = risk_store
        self.threshold = threshold

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def evaluate(self, user_input):

        user_vector = self.model.encode([user_input])[0]

        triggered = []

        for risk_name, risk_vector in self.risk_store.embeddings.items():
            score = self.cosine_similarity(user_vector, risk_vector)

            if score >= self.threshold:
                triggered.append({
                    "risk_type": risk_name,
                    "similarity_score": float(score)
                })

        if triggered:
            return {
                "risk_detected": True,
                "matched_risks": triggered
            }

        return {
            "risk_detected": False,
            "matched_risks": []
        }