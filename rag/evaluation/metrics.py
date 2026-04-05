"""
Pure metric calculation functions for RAG evaluation.

All functions are stateless and dependency-free — safe to import anywhere.

Retrieval metrics:
  precision_at_k   — fraction of top-K retrieved docs that are relevant
  recall_at_k      — fraction of all relevant docs found in top-K
  mean_reciprocal_rank — rank position of the first relevant result

Aggregation helpers:
  aggregate_retrieval_metrics — average a list of per-query retrieval dicts
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of the top-K retrieved documents that are relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant:  Set of ground-truth relevant document IDs.
        k:         Cut-off rank.

    Returns:
        Score in [0, 1].  Returns 0.0 when k <= 0.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of all relevant documents found in the top-K results.

    Vacuously returns 1.0 when the relevant set is empty (adversarial queries
    that have no ground-truth documents).

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant:  Set of ground-truth relevant document IDs.
        k:         Cut-off rank.

    Returns:
        Score in [0, 1].
    """
    if not relevant:
        return 1.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Reciprocal rank of the first relevant document in the retrieved list.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant:  Set of ground-truth relevant document IDs.

    Returns:
        1/(rank of first hit), or 0.0 if no relevant document is found.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _safe_avg(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval_scores(
    precision_scores: list[float],
    recall_scores: list[float],
    mrr_scores: list[float],
) -> dict[str, float]:
    """
    Average a collection of per-query retrieval scores.

    Args:
        precision_scores: List of P@K values (one per query).
        recall_scores:    List of Recall@K values.
        mrr_scores:       List of MRR values.

    Returns:
        Dict with keys ``precision_at_k``, ``recall_at_k``, ``mrr``.
    """
    return {
        "precision_at_k": _safe_avg(precision_scores),
        "recall_at_k": _safe_avg(recall_scores),
        "mrr": _safe_avg(mrr_scores),
    }


def aggregate_generation_scores(
    faithfulness_scores: list[float],
    relevancy_scores: list[float],
    context_precision_scores: list[float],
    context_recall_scores: list[float],
) -> dict[str, float]:
    """
    Average a collection of per-query generation scores.

    Args:
        faithfulness_scores:       Faithfulness per query.
        relevancy_scores:          Answer relevancy per query.
        context_precision_scores:  Context precision per query.
        context_recall_scores:     Context recall per query.

    Returns:
        Dict with keys ``faithfulness``, ``answer_relevancy``,
        ``context_precision``, ``context_recall``.
    """
    return {
        "faithfulness": _safe_avg(faithfulness_scores),
        "answer_relevancy": _safe_avg(relevancy_scores),
        "context_precision": _safe_avg(context_precision_scores),
        "context_recall": _safe_avg(context_recall_scores),
    }
