"""
Retrieval evaluation module.

RetrievalEvaluator accepts a pluggable search function and evaluates it
against a list of GoldenItems, computing P@K, Recall@K, and MRR for each
query and aggregating results per category.

Usage::

    from rag.hybrid_search.service import HybridSearchService
    from rag.evaluation.retrieval_evaluator import RetrievalEvaluator

    async def my_search(query: str, top_k: int) -> list[str]:
        svc = HybridSearchService()
        resp = await svc.search(HybridSearchRequest(query=query, top_k=top_k))
        return [r.id for r in resp.results]

    evaluator = RetrievalEvaluator(search_fn=my_search, k=5)
    results = await evaluator.evaluate(golden_items)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from rag.evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank
from rag.evaluation.models import GoldenItem, RetrievalMetrics

logger = logging.getLogger(__name__)

# Type alias: async function that takes (query, top_k) and returns ordered doc IDs
SearchFn = Callable[[str, int], Awaitable[list[str]]]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetrievalEvaluationError(Exception):
    """Raised when retrieval evaluation cannot proceed."""


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------


class RetrievalEvaluator:
    """
    Evaluates retrieval quality of a RAG system against golden items.

    Args:
        search_fn:   Async callable ``(query: str, top_k: int) -> list[str]``
                     returning ordered document IDs.
        k:           Rank cut-off for P@K and Recall@K (default 5).
        concurrency: Maximum number of concurrent search calls (default 5).
    """

    def __init__(
        self,
        search_fn: SearchFn,
        k: int = 5,
        concurrency: int = 5,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        self._search_fn = search_fn
        self.k = k
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(self, items: list[GoldenItem]) -> list[RetrievalMetrics]:
        """
        Evaluate retrieval metrics for every item in the list.

        Each item is evaluated independently.  Failures for individual items
        are logged at ERROR level and that item receives zero scores so the
        overall run still completes.

        Args:
            items: List of GoldenItem objects to evaluate.

        Returns:
            List of RetrievalMetrics in the same order as ``items``.

        Raises:
            RetrievalEvaluationError: when ``items`` is empty.
        """
        if not items:
            raise RetrievalEvaluationError("No golden items provided for retrieval evaluation.")

        logger.info("Starting retrieval evaluation: items=%s k=%s", len(items), self.k)

        results = await asyncio.gather(
            *[self._evaluate_one(item) for item in items],
            return_exceptions=False,
        )

        logger.info("Retrieval evaluation complete: evaluated=%s", len(results))
        return list(results)

    async def evaluate_one(self, item: GoldenItem) -> RetrievalMetrics:
        """Evaluate a single golden item (public convenience wrapper)."""
        return await self._evaluate_one(item)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate_one(self, item: GoldenItem) -> RetrievalMetrics:
        """Run search for a single item and compute its metrics."""
        relevant = set(item.relevant_doc_ids)

        async with self._semaphore:
            try:
                retrieved = await self._search_fn(item.question, self.k)
            except Exception as exc:
                logger.error(
                    "Search failed for question %r: %s — assigning zero scores.",
                    item.question[:60], exc,
                )
                retrieved = []

        p = precision_at_k(retrieved, relevant, self.k)
        r = recall_at_k(retrieved, relevant, self.k)
        m = mean_reciprocal_rank(retrieved, relevant)

        logger.debug(
            "Retrieval metrics: question=%r P@%s=%.3f R@%s=%.3f MRR=%.3f",
            item.question[:60], self.k, p, self.k, r, m,
        )

        return RetrievalMetrics(
            question=item.question,
            retrieved_ids=retrieved,
            relevant_ids=list(relevant),
            precision_at_k=p,
            recall_at_k=r,
            mrr=m,
            k=self.k,
        )
