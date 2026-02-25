import glob
import json
from collections import Counter

class CorpusStats:

    def __init__(self, chunks_path="chunks/*.json"):
        self.freq = Counter()
        self.total_chunks = 0
        self._build_index(chunks_path)

    def _build_index(self, path):
        for file in glob.glob(path):
            with open(file, "r") as f:
                chunks = json.load(f)

            for chunk in chunks:
                self.total_chunks += 1
                words = set(chunk["text"].lower().split())

                for word in words:
                    self.freq[word] += 1

    def idf(self, term):
        import math
        f = self.freq.get(term.lower(), 1)
        return math.log((self.total_chunks + 1) / f)