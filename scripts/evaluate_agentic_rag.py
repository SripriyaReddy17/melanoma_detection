"""
Evaluation Framework for Agentic RAG (CEFM Melanoma Framework).

Provides mathematical formulations for information retrieval metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Evidence Sufficiency Rate
- Average Retrieval Rounds

IMPORTANT:
Does NOT fabricate benchmark results.
If no labeled ground-truth retrieval dataset exists, reports clearly that the
evaluation dataset is not available. Quantitative evaluation remains future work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Set, Dict, Any, Optional


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 3) -> float:
    """
    Calculate Precision@K: fraction of top-K retrieved items that are relevant.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 3) -> float:
    """
    Calculate Recall@K: fraction of relevant items that appear in top-K retrieved items.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / len(relevant_ids))


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant item.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int = 3) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    Relevance scores reflect graded clinical relevance (e.g., 0, 1, 2, 3).
    """
    if k <= 0 or not retrieved_ids or not relevance_scores:
        return 0.0

    top_k = retrieved_ids[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        rel = float(relevance_scores.get(item, 0.0))
        # Standard formulation: (2^rel - 1) / log2(rank + 1)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank + 1.0)
        dcg += gain / discount

    # Calculate Ideal DCG (IDCG)
    sorted_ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(sorted_ideal, start=1):
        gain = (2.0 ** float(rel)) - 1.0
        discount = math.log2(rank + 1.0)
        idcg += gain / discount

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def evidence_sufficiency_rate(evaluations: List[Dict[str, Any]]) -> float:
    """
    Calculate the percentage of cases achieving evidence sufficiency.
    """
    if not evaluations:
        return 0.0
    sufficient_count = sum(
        1 for e in evaluations
        if e.get("sufficiency") == "sufficient" or e.get("is_sufficient") is True
    )
    return float(sufficient_count / len(evaluations))


def average_retrieval_rounds(evaluations: List[Dict[str, Any]]) -> float:
    """
    Calculate the average number of retrieval rounds executed per case.
    """
    if not evaluations:
        return 0.0
    total_rounds = sum(int(e.get("rounds_executed", 1)) for e in evaluations)
    return float(total_rounds / len(evaluations))


def run_benchmark(dataset_path: Path, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """
    Execute quantitative retrieval evaluation on a ground-truth dataset.
    Reports if dataset does not exist.
    """
    if not dataset_path.exists():
        print("Evaluation dataset not available.")
        print(f"File not found: {dataset_path}")
        print(
            "Note: Quantitative retrieval benchmarking requires an expert-annotated "
            "clinical ground-truth dataset. Quantitative Agentic RAG evaluation remains future work."
        )
        return None

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Error loading evaluation dataset: {exc}")
        return None

    test_cases = data.get("test_cases", [])
    if not test_cases:
        print("Evaluation dataset not available (0 test cases present).")
        return None

    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    for case in test_cases:
        retrieved = case.get("retrieved_ids", [])
        relevant = set(case.get("relevant_ids", []))
        rel_scores = case.get("relevance_scores", {item: 1.0 for item in relevant})

        precisions.append(precision_at_k(retrieved, relevant, k=top_k))
        recalls.append(recall_at_k(retrieved, relevant, k=top_k))
        mrrs.append(mean_reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, rel_scores, k=top_k))

    n = len(test_cases)
    results = {
        "num_cases": n,
        "k": top_k,
        "mean_precision_at_k": sum(precisions) / n if n else 0.0,
        "mean_recall_at_k": sum(recalls) / n if n else 0.0,
        "mean_mrr": sum(mrrs) / n if n else 0.0,
        "mean_ndcg_at_k": sum(ndcgs) / n if n else 0.0,
        "sufficiency_rate": evidence_sufficiency_rate(test_cases),
        "average_rounds": average_retrieval_rounds(test_cases),
    }

    print("\n================ AGENTIC RAG EVALUATION RESULTS ================")
    print(f"Test cases evaluated: {results['num_cases']}")
    print(f"Precision@{top_k}:      {results['mean_precision_at_k']:.4f}")
    print(f"Recall@{top_k}:         {results['mean_recall_at_k']:.4f}")
    print(f"MRR:                {results['mean_mrr']:.4f}")
    print(f"NDCG@{top_k}:           {results['mean_ndcg_at_k']:.4f}")
    print(f"Sufficiency Rate:   {results['sufficiency_rate']:.2%}")
    print(f"Average Rounds:     {results['average_rounds']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Agentic RAG retrieval performance and evidence sufficiency."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/rag_benchmark.json",
        help="Path to labeled retrieval ground-truth JSON dataset.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K cutoff for retrieval metrics (default: 3).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    results = run_benchmark(dataset_path, top_k=args.top_k)
    if results is None:
        sys.exit(0)


if __name__ == "__main__":
    main()
