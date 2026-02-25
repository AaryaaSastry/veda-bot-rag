import numpy as np

class RareSymptomBooster:

    def __init__(self, embedding_model, corpus_stats):
        self.embedding_model = embedding_model
        self.stats = corpus_stats

    def build_weighted_vector(self, user_input):
        words = user_input.lower().split()

        base_vector = self.embedding_model.encode([user_input])[0]
        boosted_vector = np.zeros_like(base_vector)

        for word in words:
            word_vec = self.embedding_model.encode([word])[0]
            weight = self.stats.idf(word)

            boosted_vector += word_vec * weight

        final_vector = base_vector + boosted_vector

        return final_vector.astype("float32")