"""
HybridSearchService — orchestrates vector + BM25 search with RRF fusion.

Both search paths run concurrently via asyncio.gather. If one path fails
the other's results are still fused and returned (degraded mode). Only
when BOTH paths fail is HybridSearchError raised.

Usage::

    svc = HybridSearchService()

    response = await svc.search(
        HybridSearchRequest(query="AWS EC2 pricing November 2024", top_k=5)
    )

    for result in response.results:
        print(result.rrf_score, result.content[:80])
"""

import asyncio
import logging

from rag.embeddings.service import EmbeddingService
from rag.hybrid_search.bm25 import KeywordSearcher, KeywordSearchError
from rag.hybrid_search.models import HybridSearchRequest, HybridSearchResponse
from rag.hybrid_search.reranker import RRFReranker
from rag.hybrid_search.vector import VectorSearcher, VectorSearchError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HybridSearchError(Exception):
    """Raised when both the vector and BM25 search paths fail."""


# ---------------------------------------------------------------------------
# HybridSearchService
# ---------------------------------------------------------------------------


class HybridSearchService:
    """
    Hybrid search: semantic vector search + BM25 keyword search, merged via RRF.

    Args:
        embed_service: Optional pre-built EmbeddingService (useful for testing).
        rrf_k:         RRF smoothing constant forwarded to RRFReranker (default 60).
    """

    def __init__(
        self,
        embed_service: EmbeddingService | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._vector = VectorSearcher(embed_service=embed_service)
        self._keyword = KeywordSearcher()
        self._reranker = RRFReranker(k=rrf_k)

    async def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        """
        Run hybrid search for the given request.

        Both paths run concurrently. A failed path is logged at ERROR level
        and treated as an empty result list so fusion can still proceed.
        HybridSearchError is raised only when both paths fail.

        Args:
            request: HybridSearchRequest with query, top_k, fetch_k, rrf_k.

        Returns:
            HybridSearchResponse with RRF-ranked results and candidate counts.

        Raises:
            HybridSearchError: when both vector search and BM25 search fail.
        """
        logger.info(
            "Hybrid search: query=%r top_k=%s fetch_k=%s rrf_k=%s",
            request.query, request.top_k, request.fetch_k, request.rrf_k,
        )

        vector_outcome, bm25_outcome = await asyncio.gather(
            self._safe_vector_search(request.query, request.fetch_k),
            self._safe_keyword_search(request.query, request.fetch_k),
            return_exceptions=True,
        )

        vector_error = isinstance(vector_outcome, BaseException)
        bm25_error = isinstance(bm25_outcome, BaseException)

        if vector_error:
            logger.error("Vector search path failed: %s", vector_outcome)
            vector_results: list[dict] = []
        else:
            vector_results = vector_outcome  # type: ignore[assignment]

        if bm25_error:
            logger.error("BM25 search path failed: %s", bm25_outcome)
            bm25_results: list[dict] = []
        else:
            bm25_results = bm25_outcome  # type: ignore[assignment]

        if vector_error and bm25_error:
            raise HybridSearchError(
                f"Both search paths failed. "
                f"vector_error={vector_outcome!r} bm25_error={bm25_outcome!r}"
            )

        reranker = RRFReranker(k=request.rrf_k)
        results = reranker.fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=request.top_k,
        )

        logger.info(
            "Hybrid search complete: result_count=%s vector_candidates=%s bm25_candidates=%s",
            len(results), len(vector_results), len(bm25_results),
        )

        return HybridSearchResponse(
            query=request.query,
            results=results,
            total_vector_candidates=len(vector_results),
            total_bm25_candidates=len(bm25_results),
        )

    # ------------------------------------------------------------------
    # Internal helpers — convert exceptions so gather can distinguish them
    # ------------------------------------------------------------------

    async def _safe_vector_search(self, query: str, top_k: int) -> list[dict]:
        try:
            return await self._vector.search(query, top_k=top_k)
        except VectorSearchError:
            raise
        except Exception as e:
            raise VectorSearchError(f"Unexpected error in vector search: {e}") from e

    async def _safe_keyword_search(self, query: str, top_k: int) -> list[dict]:
        try:
            return await self._keyword.search(query, top_k=top_k)
        except KeywordSearchError:
            raise
        except Exception as e:
            raise KeywordSearchError(f"Unexpected error in keyword search: {e}") from e
