"""
Reciprocal Rank Fusion (RRF) — merges multiple ranked result lists.

Formula (from Cormack et al., 2009):
    RRF_score(d) = Σ  1 / (rank_i(d) + k)

where rank_i(d) is the 1-based rank of document d in list i and k is a
smoothing constant. k=60 is the value validated in the original paper and
widely used in production hybrid-search systems.

Documents appearing in only one list still get an RRF contribution from
that list; appearing in both doubles their boost — exactly the desired
behaviour for hybrid search.
"""

import logging

from rag.hybrid_search.models import SearchResult

logger = logging.getLogger(__name__)


class RRFReranker:
    """
    Merge two ranked result lists (vector + BM25) via Reciprocal Rank Fusion.

    Args:
        k: RRF smoothing constant (default 60). Higher values reduce the
           influence of top-ranked documents; lower values amplify it.
    """

    def __init__(self, k: int = 60) -> None:
        if k < 1:
            raise ValueError(f"RRF constant k must be >= 1, got {k}")
        self.k = k

    def fuse(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Combine vector and BM25 ranked lists and return the top_k results.

        Args:
            vector_results: Ordered list of dicts from VectorSearcher.search().
            bm25_results:   Ordered list of dicts from KeywordSearcher.search().
            top_k:          Number of results to return after fusion.

        Returns:
            List of SearchResult objects sorted by descending RRF score.
            An empty list is returned when both input lists are empty.
        """
        if not vector_results and not bm25_results:
            logger.warning("RRFReranker.fuse called with both lists empty")
            return []

        doc_scores: dict[str, dict] = {}

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc["id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {**doc, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None}
            doc_scores[doc_id]["rrf_score"] += 1.0 / (rank + self.k)
            doc_scores[doc_id]["vector_rank"] = rank

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {**doc, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None}
            doc_scores[doc_id]["rrf_score"] += 1.0 / (rank + self.k)
            doc_scores[doc_id]["bm25_rank"] = rank

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        logger.info(
            "RRF fusion complete: total_candidates=%s returning=%s k=%s",
            len(sorted_docs), min(top_k, len(sorted_docs)), self.k,
        )

        return [
            SearchResult(
                id=item["id"],
                content=item["content"],
                source_file=item.get("source_file"),
                page_number=item.get("page_number"),
                metadata=item.get("metadata", {}),
                rrf_score=item["rrf_score"],
                vector_rank=item["vector_rank"],
                bm25_rank=item["bm25_rank"],
            )
            for item in sorted_docs[:top_k]
        ]
