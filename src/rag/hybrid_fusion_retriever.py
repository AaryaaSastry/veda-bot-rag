import json
import math
import os
import re
from collections import Counter, defaultdict

import config
from rag.retriever import Retriever


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class HybridFusionRetriever:
    """
    Hybrid retriever that fuses:
    - dense vector retrieval (existing FAISS retriever)
    - lexical BM25 retrieval (in-memory inverted index)
    using weighted Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_db_dir,
        dense_candidates=None,
        bm25_candidates=None,
        rrf_k=None,
        dense_weight=None,
        bm25_weight=None,
    ):
        self.vector_db_dir = vector_db_dir
        self.dense_retriever = Retriever(vector_db_dir)

        metadata_path = os.path.join(vector_db_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.dense_candidates = dense_candidates or getattr(config, "HYBRID_DENSE_CANDIDATES", 80)
        self.bm25_candidates = bm25_candidates or getattr(config, "HYBRID_BM25_CANDIDATES", 80)
        self.rrf_k = rrf_k or getattr(config, "HYBRID_RRF_K", 60)
        self.dense_weight = dense_weight or getattr(config, "HYBRID_DENSE_WEIGHT", 1.0)
        self.bm25_weight = bm25_weight or getattr(config, "HYBRID_BM25_WEIGHT", 1.0)

        self._build_bm25_index()

    def _tokenize(self, text):
        return TOKEN_PATTERN.findall((text or "").lower())

    def _build_bm25_index(self):
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_sources = []
        self.df = defaultdict(int)
        self.postings = defaultdict(list)

        for idx, item in enumerate(self.metadata):
            text = item.get("text", "")
            source = item.get("source", "")
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            self.doc_sources.append(source)

            for term in tf:
                self.df[term] += 1
                self.postings[term].append(idx)

        self.num_docs = len(self.metadata)
        self.avg_doc_len = (sum(self.doc_lengths) / self.num_docs) if self.num_docs else 0.0
        self.k1 = 1.2
        self.b = 0.75

    def _bm25_top_indices(self, query, top_k, source_filter=None):
        if self.num_docs == 0:
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        candidate_ids = set()
        for term in terms:
            candidate_ids.update(self.postings.get(term, []))

        if source_filter:
            candidate_ids = {i for i in candidate_ids if self.doc_sources[i] == source_filter}

        if not candidate_ids:
            return []

        scores = {}
        for doc_id in candidate_ids:
            tf = self.doc_term_freqs[doc_id]
            dl = self.doc_lengths[doc_id]
            if dl == 0:
                continue
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (self.avg_doc_len + 1e-9)))
                score += idf * (f * (self.k1 + 1.0)) / denom
            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]

    def retrieve(self, query, k=config.K_DEFAULT, source_filter=None):
        dense_k = max(k, self.dense_candidates)
        bm25_k = max(k, self.bm25_candidates)

        dense_results = self.dense_retriever.retrieve(query, k=dense_k, source_filter=source_filter)
        bm25_ids = self._bm25_top_indices(query, top_k=bm25_k, source_filter=source_filter)

        # Build rank maps for RRF fusion.
        dense_rank = {}
        for rank, item in enumerate(dense_results, start=1):
            idx = item.get("metadata_index")
            if idx is not None and idx not in dense_rank:
                dense_rank[idx] = rank

        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_ids, start=1)}

        candidate_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        if not candidate_ids:
            return []

        fused = []
        for doc_id in candidate_ids:
            score = 0.0
            if doc_id in dense_rank:
                score += self.dense_weight / (self.rrf_k + dense_rank[doc_id])
            if doc_id in bm25_rank:
                score += self.bm25_weight / (self.rrf_k + bm25_rank[doc_id])
            fused.append((doc_id, score))

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in fused[:k]:
            item = dict(self.metadata[doc_id])
            item["metadata_index"] = int(doc_id)
            item["fusion_score"] = float(score)
            item["source"] = item.get("source", "")
            item["text"] = item.get("text", "")
            results.append(item)

        return results
