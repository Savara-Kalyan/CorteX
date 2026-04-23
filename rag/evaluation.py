from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int = 5) -> float:
    return len(set(retrieved_ids[:k]) & relevant_ids) / k if k > 0 else 0.0


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int = 5) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_batch(results: list[dict], k: int = 5) -> dict[str, float]:
    p_scores, mrr_scores, ndcg_scores = [], [], []

    for item in results:
        retrieved = item["retrieved_ids"]
        relevant = item["relevant_ids"]
        p_scores.append(precision_at_k(retrieved, relevant, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))

    n = len(results) or 1
    return {
        f"P@{k}": round(sum(p_scores) / n, 4),
        "MRR":    round(sum(mrr_scores) / n, 4),
        f"NDCG@{k}": round(sum(ndcg_scores) / n, 4),
    }
