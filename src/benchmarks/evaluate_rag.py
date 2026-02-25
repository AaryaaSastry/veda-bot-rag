"""
RAG Pipeline Evaluation Script

Evaluates:
1. Retrieval Accuracy - How well relevant chunks are retrieved
2. Safety Detection Accuracy - How well high-risk conditions are detected
3. Response Quality - Manual evaluation of generated responses
"""

import os
import sys
import json
import math
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_pipeline import RAGPipeline
from rag.memory import ConversationMemory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")
GOLD_RETRIEVAL_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "retrieval_gold.json")
API_KEY = "AIzaSyCXAd1eoAyTYB80xnPmi0dqg1rAQhvhz0U"

# Test cases for retrieval evaluation
RETRIEVAL_TEST_CASES = [
    {
        "query": "What are the symptoms of migraine headache?",
        "expected_keywords": ["headache", "migraine", "pain", "head"],
        "relevant_chunk_keywords": ["headache", "migraine", "pain"]
    },
    {
        "query": "Ayurvedic treatment for digestive problems",
        "expected_keywords": ["digestive", "stomach", "digestion", "agni"],
        "relevant_chunk_keywords": ["digestive", "stomach", "agni", "digestion"]
    },
    {
        "query": "Remedies for joint pain and arthritis",
        "expected_keywords": ["joint", "pain", "arthritis", "vata"],
        "relevant_chunk_keywords": ["joint", "pain", "arthritis", "vata"]
    },
    {
        "query": "Treatment for skin diseases in Ayurveda",
        "expected_keywords": ["skin", "disease", "treatment"],
        "relevant_chunk_keywords": ["skin", "kustha", "disease"]
    },
    {
        "query": "How to treat fever with Ayurvedic remedies",
        "expected_keywords": ["fever", "jvara", "temperature"],
        "relevant_chunk_keywords": ["fever", "jvara", "temperature"]
    }
]

# Test cases for safety detection
SAFETY_TEST_CASES = [
    {"query": "I have swelling in one leg with fever", "should_flag": True},
    {"query": "I have chest pain with difficulty breathing", "should_flag": True},
    {"query": "I have sudden severe headache with vision problems", "should_flag": True},
    {"query": "I have a mild headache", "should_flag": False},
    {"query": "I have slight indigestion after eating", "should_flag": False},
    {"query": "I have mild joint pain in my knee", "should_flag": False},
    {"query": "I have vomiting blood with weakness", "should_flag": True},
    {"query": "I have sudden weakness on one side of body", "should_flag": True},
]


def _make_match_key(chunk):
    source = (chunk.get("source") or "").strip().lower()
    chapter = (chunk.get("chapter") or "").strip().lower()
    if chapter:
        return f"{source}||{chapter}"
    return source


def _load_gold_cases():
    if not os.path.exists(GOLD_RETRIEVAL_PATH):
        return []

    with open(GOLD_RETRIEVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("cases", [])
    if isinstance(data, list):
        return data
    return []


def _recall_at_k(binary_relevance, k):
    if not binary_relevance:
        return 0.0
    return 1.0 if any(binary_relevance[:k]) else 0.0


def _mrr(binary_relevance):
    for rank, rel in enumerate(binary_relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(binary_relevance, k):
    gains = binary_relevance[:k]
    dcg = 0.0
    for i, rel in enumerate(gains, start=1):
        if rel:
            dcg += 1.0 / math.log2(i + 1.0)

    ideal_count = min(k, sum(binary_relevance))
    idcg = 0.0
    for i in range(1, ideal_count + 1):
        idcg += 1.0 / math.log2(i + 1.0)

    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval_with_gold(pipeline, top_ks=(1, 3, 5, 10)):
    """Evaluate retrieval with manually labeled relevance data."""
    print("\n=== RETRIEVAL EVALUATION (GOLD LABELS) ===")

    gold_cases = _load_gold_cases()
    if not gold_cases:
        print(f"No gold retrieval dataset found at: {GOLD_RETRIEVAL_PATH}")
        return None

    max_k = max(top_ks)
    per_case = []
    recall_totals = {k: 0.0 for k in top_ks}
    ndcg_totals = {k: 0.0 for k in top_ks}
    mrr_total = 0.0

    valid_cases = 0
    for i, case in enumerate(gold_cases):
        query = case.get("query", "").strip()
        relevant = case.get("relevant_chunks", [])
        if not query or not relevant:
            continue

        gold_keys = set()
        for item in relevant:
            key = _make_match_key(item)
            if key:
                gold_keys.add(key)

        if not gold_keys:
            continue

        retrieved = pipeline.retriever.retrieve(query, k=max_k)
        relevance = [1 if _make_match_key(c) in gold_keys else 0 for c in retrieved]

        recalls = {k: _recall_at_k(relevance, k) for k in top_ks}
        ndcgs = {k: _ndcg_at_k(relevance, k) for k in top_ks}
        case_mrr = _mrr(relevance)

        for k in top_ks:
            recall_totals[k] += recalls[k]
            ndcg_totals[k] += ndcgs[k]
        mrr_total += case_mrr

        valid_cases += 1
        per_case.append(
            {
                "query": query,
                "recall": recalls,
                "ndcg": ndcgs,
                "mrr": case_mrr,
                "hits_in_top_k": int(sum(relevance[:max_k])),
            }
        )

        print(f"\nTest {i+1}: {query}")
        print(f"  Recall@{top_ks}: {[round(recalls[k], 3) for k in top_ks]}")
        print(f"  nDCG@{top_ks}: {[round(ndcgs[k], 3) for k in top_ks]}")
        print(f"  MRR: {case_mrr:.3f}")

    if valid_cases == 0:
        print("Gold dataset exists but no valid test cases were found.")
        return None

    metrics = {
        "num_cases": valid_cases,
        "recall_at_k": {str(k): recall_totals[k] / valid_cases for k in top_ks},
        "ndcg_at_k": {str(k): ndcg_totals[k] / valid_cases for k in top_ks},
        "mrr": mrr_total / valid_cases,
    }

    print("\n>>> Gold Retrieval Metrics")
    for k in top_ks:
        print(f"Recall@{k}: {metrics['recall_at_k'][str(k)]:.3f}")
    for k in top_ks:
        print(f"nDCG@{k}: {metrics['ndcg_at_k'][str(k)]:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")

    return {
        "mode": "gold",
        "dataset_path": GOLD_RETRIEVAL_PATH,
        "metrics": metrics,
        "results": per_case,
    }


def evaluate_retrieval(pipeline):
    """Evaluate retrieval accuracy."""
    print("\n=== RETRIEVAL EVALUATION ===")
    
    results = []
    total_score = 0
    
    for i, test_case in enumerate(RETRIEVAL_TEST_CASES):
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        
        # Retrieve chunks
        memory = ConversationMemory()
        memory.add_turn("user", query)
        
        # Get retrieval query
        retrieval_query, _ = pipeline._classify_and_weight_query(query)
        
        # Retrieve without generation
        chunks = pipeline.retriever.retrieve(retrieval_query, k=5)
        
        # Check if retrieved chunks contain expected keywords
        retrieved_text = " ".join([c.get("text", "").lower() for c in chunks])
        
        keyword_hits = 0
        for keyword in expected_keywords:
            if keyword.lower() in retrieved_text:
                keyword_hits += 1
        
        keyword_score = keyword_hits / len(expected_keywords) * 100
        
        result = {
            "query": query,
            "keyword_score": keyword_score,
            "keywords_found": keyword_hits,
            "total_keywords": len(expected_keywords),
            "chunks_retrieved": len(chunks)
        }
        results.append(result)
        total_score += keyword_score
        
        print(f"\nTest {i+1}: {query}")
        print(f"  Keyword Score: {keyword_score:.1f}% ({keyword_hits}/{len(expected_keywords)} keywords found)")
        print(f"  Chunks Retrieved: {len(chunks)}")
    
    avg_score = total_score / len(RETRIEVAL_TEST_CASES)
    print(f"\n>>> Average Retrieval Score: {avg_score:.1f}%")
    
    return {
        "average_score": avg_score,
        "results": results
    }


def evaluate_safety(pipeline):
    """Evaluate safety detection accuracy."""
    print("\n=== SAFETY DETECTION EVALUATION ===")
    
    results = []
    correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i, test_case in enumerate(SAFETY_TEST_CASES):
        query = test_case["query"]
        should_flag = test_case["should_flag"]
        
        is_safe, safety_result = pipeline.check_safety(query)
        flagged = not is_safe
        
        # Calculate metrics
        if should_flag and flagged:
            true_positives += 1
            correct += 1
        elif not should_flag and not flagged:
            true_negatives += 1
            correct += 1
        elif should_flag and not flagged:
            false_negatives += 1
        else:  # not should_flag and flagged
            false_positives += 1
        
        result = {
            "query": query,
            "should_flag": should_flag,
            "flagged": flagged,
            "correct": flagged == should_flag,
            "matched_risks": safety_result.get("matched_risks", []) if safety_result else []
        }
        results.append(result)
        
        status = "[CORRECT]" if flagged == should_flag else "[WRONG]"
        print(f"\nTest {i+1}: {status}")
        print(f"  Query: {query}")
        print(f"  Expected: {'FLAG' if should_flag else 'SAFE'}, Got: {'FLAG' if flagged else 'SAFE'}")
        if safety_result and safety_result.get("matched_risks"):
            print(f"  Matched: {[r['risk_type'] for r in safety_result['matched_risks']]}")
    
    total = len(SAFETY_TEST_CASES)
    accuracy = correct / total * 100
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n>>> Safety Detection Accuracy: {accuracy:.1f}%")
    print(f">>> Precision: {precision:.1f}%")
    print(f">>> Recall: {recall:.1f}%")
    print(f">>> F1 Score: {f1:.1f}%")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "results": results
    }


def evaluate_response_quality(pipeline):
    """Evaluate response quality (requires manual inspection)."""
    print("\n=== RESPONSE QUALITY EVALUATION ===")
    print("This requires manual inspection. Generating sample responses...")
    
    test_queries = [
        "I have a headache for 3 days",
        "I feel bloated after eating",
        "I have joint pain in my knees"
    ]
    
    results = []
    
    for query in test_queries:
        memory = ConversationMemory()
        memory.add_turn("user", query)
        
        print(f"\nQuery: {query}")
        print("Response: ", end="", flush=True)
        
        full_response = ""
        for chunk in pipeline.run(query, memory):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        
        results.append({
            "query": query,
            "response": full_response
        })
        
        memory.add_turn("assistant", full_response)
    
    print("\n>>> Please manually evaluate the response quality above.")
    print(">>> Criteria: Is the question relevant? Is there no sympathy/filler? Is it professional?")
    
    return {
        "note": "Manual evaluation required",
        "sample_responses": results
    }


def run_evaluation():
    """Run full evaluation."""
    print("=" * 60)
    print("RAG PIPELINE EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = RAGPipeline(
        vector_db_path=VECTOR_DB_PATH,
        api_key=API_KEY,
        use_enhanced_retrieval=True,
        safety_threshold=0.65  # Slightly higher threshold to reduce false positives
    )
    print("Pipeline initialized!")
    
    # Run evaluations
    retrieval_results = evaluate_retrieval_with_gold(pipeline)
    if retrieval_results is None:
        print("Falling back to keyword-based retrieval evaluation.")
        retrieval_results = evaluate_retrieval(pipeline)
    safety_results = evaluate_safety(pipeline)
    response_results = evaluate_response_quality(pipeline)
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if retrieval_results.get("mode") == "gold":
        print(f"Retrieval MRR: {retrieval_results['metrics']['mrr']:.3f}")
        print(f"Retrieval Recall@5: {retrieval_results['metrics']['recall_at_k']['5']:.3f}")
        print(f"Retrieval nDCG@5: {retrieval_results['metrics']['ndcg_at_k']['5']:.3f}")
    else:
        print(f"Retrieval Accuracy: {retrieval_results['average_score']:.1f}%")
    print(f"Safety Detection Accuracy: {safety_results['accuracy']:.1f}%")
    print(f"Safety Precision: {safety_results['precision']:.1f}%")
    print(f"Safety Recall: {safety_results['recall']:.1f}%")
    print(f"Safety F1 Score: {safety_results['f1_score']:.1f}%")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "retrieval": retrieval_results,
        "safety": safety_results,
        "response_quality": response_results
    }
    
    output_path = os.path.join(PROJECT_ROOT, "data", "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_evaluation()
