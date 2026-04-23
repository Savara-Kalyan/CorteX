"""
Hybrid search: semantic vector + BM25 keyword search, merged via Reciprocal Rank Fusion.

  SearchResult, HybridSearchRequest, HybridSearchResponse  — data models
  KeywordSearcher    — PostgreSQL full-text search (BM25-style)
  VectorSearcher     — pgvector cosine similarity
  RRFReranker        — merges both ranked lists
  HybridSearchService — orchestrates both paths concurrently
"""

import asyncio
import logging

import psycopg
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel, Field

from settings import settings
from rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


# Models

class SearchResult(BaseModel):
    id: str
    content: str
    source_file: str | None = None
    page_number: int | None = None
    metadata: dict = Field(default_factory=dict)
    rrf_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    fetch_k: int = Field(default=40, ge=1)
    user_tier: str = "public"
    domain: str | None = None


class HybridSearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_vector_candidates: int
    total_bm25_candidates: int


# Exceptions

class KeywordSearchError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[KeywordSearch] {reason}")


class VectorSearchError(Exception):
    """Base exception for vector-search errors."""


class VectorConnectionError(VectorSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[VectorSearch] DB connection failed: {reason}")


class VectorEmbeddingError(VectorSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[VectorSearch] Embedding generation failed: {reason}")


class VectorQueryError(VectorSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[VectorSearch] Query execution failed: {reason}")


class HybridSearchError(Exception):
    """Raised when both vector and BM25 search paths fail."""


# KeywordSearcher — PostgreSQL FTS (BM25-style)

class KeywordSearcher:

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        try:
            conn = await psycopg.AsyncConnection.connect(settings.vector_store.connection_string)
            await register_vector_async(conn)
            return conn
        except psycopg.OperationalError as e:
            raise KeywordSearchError(f"DB connection failed: {e}") from e

    async def search(self, query: str, top_k: int = 40, domain: str | None = None) -> list[dict]:
        if not query.strip():
            return []
        try:
            async with await self._connect() as conn:
                async with conn.cursor() as cur:
                    if domain:
                        await cur.execute(
                            """
                            SELECT id::text, content, source_file, page_number, metadata,
                                   ts_rank_cd(to_tsvector('english', content),
                                              plainto_tsquery('english', %s)) AS bm25_score
                            FROM documents
                            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                              AND metadata->>'domain' = %s
                            ORDER BY bm25_score DESC
                            LIMIT %s;
                            """,
                            (query, query, domain, top_k),
                        )
                    else:
                        await cur.execute(
                            """
                            SELECT id::text, content, source_file, page_number, metadata,
                                   ts_rank_cd(to_tsvector('english', content),
                                              plainto_tsquery('english', %s)) AS bm25_score
                            FROM documents
                            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                            ORDER BY bm25_score DESC
                            LIMIT %s;
                            """,
                            (query, query, top_k),
                        )
                    rows = await cur.fetchall()
        except psycopg.OperationalError as e:
            raise KeywordSearchError(f"DB connection failed: {e}") from e
        except psycopg.DatabaseError as e:
            raise KeywordSearchError(f"Query failed: {e}") from e

        return [
            {"id": r[0], "content": r[1], "source_file": r[2], "page_number": r[3],
             "metadata": r[4] or {}, "bm25_score": float(r[5]) if r[5] is not None else 0.0}
            for r in rows
        ]


# VectorSearcher — pgvector cosine distance

class VectorSearcher:

    def __init__(self, embed_service: EmbeddingService | None = None) -> None:
        self._embed = embed_service or EmbeddingService()

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        try:
            conn = await psycopg.AsyncConnection.connect(settings.vector_store.connection_string)
            await register_vector_async(conn)
            return conn
        except psycopg.OperationalError as e:
            raise VectorConnectionError(str(e)) from e

    async def search(self, query: str, top_k: int = 40, domain: str | None = None) -> list[dict]:
        if not query.strip():
            return []
        try:
            query_embedding = await self._embed.embed_query(query)
        except Exception as e:
            raise VectorEmbeddingError(str(e)) from e

        try:
            async with await self._connect() as conn:
                async with conn.cursor() as cur:
                    if domain:
                        await cur.execute(
                            """
                            SELECT id::text, content, source_file, page_number, metadata,
                                   1 - (embedding <=> %s::vector) AS vector_score
                            FROM documents
                            WHERE metadata->>'domain' = %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                            """,
                            (query_embedding, domain, query_embedding, top_k),
                        )
                    else:
                        await cur.execute(
                            """
                            SELECT id::text, content, source_file, page_number, metadata,
                                   1 - (embedding <=> %s::vector) AS vector_score
                            FROM documents
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                            """,
                            (query_embedding, query_embedding, top_k),
                        )
                    rows = await cur.fetchall()
        except psycopg.OperationalError as e:
            raise VectorConnectionError(str(e)) from e
        except psycopg.DatabaseError as e:
            raise VectorQueryError(str(e)) from e

        return [
            {"id": r[0], "content": r[1], "source_file": r[2], "page_number": r[3],
             "metadata": r[4] or {}, "vector_score": float(r[5]) if r[5] is not None else 0.0}
            for r in rows
        ]


# RRFReranker — Reciprocal Rank Fusion

class RRFReranker:

    def __init__(self, k: int = 60) -> None:
        if k < 1:
            raise ValueError(f"RRF constant k must be >= 1, got {k}")
        self.k = k

    def fuse(self, vector_results: list[dict], bm25_results: list[dict], top_k: int = 5) -> list[SearchResult]:
        if not vector_results and not bm25_results:
            return []

        doc_scores: dict[str, dict] = {}

        for rank, doc in enumerate(vector_results, start=1):
            did = doc["id"]
            if did not in doc_scores:
                doc_scores[did] = {**doc, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None}
            doc_scores[did]["rrf_score"] += 1.0 / (rank + self.k)
            doc_scores[did]["vector_rank"] = rank

        for rank, doc in enumerate(bm25_results, start=1):
            did = doc["id"]
            if did not in doc_scores:
                doc_scores[did] = {**doc, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None}
            doc_scores[did]["rrf_score"] += 1.0 / (rank + self.k)
            doc_scores[did]["bm25_rank"] = rank

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        return [
            SearchResult(
                id=item["id"], content=item["content"],
                source_file=item.get("source_file"), page_number=item.get("page_number"),
                metadata=item.get("metadata", {}), rrf_score=item["rrf_score"],
                vector_rank=item["vector_rank"], bm25_rank=item["bm25_rank"],
            )
            for item in sorted_docs[:top_k]
        ]


# HybridSearchService

class HybridSearchService:

    def __init__(self, embed_service: EmbeddingService | None = None, rrf_k: int = 60) -> None:
        self._vector = VectorSearcher(embed_service=embed_service)
        self._keyword = KeywordSearcher()
        self._reranker = RRFReranker(k=rrf_k)

    async def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        vector_outcome, bm25_outcome = await asyncio.gather(
            self._safe_vector(request.query, request.fetch_k, request.domain),
            self._safe_keyword(request.query, request.fetch_k, request.domain),
            return_exceptions=True,
        )

        vector_error = isinstance(vector_outcome, BaseException)
        bm25_error = isinstance(bm25_outcome, BaseException)

        if vector_error and bm25_error:
            raise HybridSearchError(
                f"Both search paths failed: vector={vector_outcome!r} bm25={bm25_outcome!r}"
            )

        vector_results: list[dict] = [] if vector_error else vector_outcome  # type: ignore
        bm25_results: list[dict] = [] if bm25_error else bm25_outcome  # type: ignore

        reranker = RRFReranker(k=request.rrf_k)
        results = reranker.fuse(vector_results, bm25_results, top_k=request.top_k)

        return HybridSearchResponse(
            query=request.query, results=results,
            total_vector_candidates=len(vector_results),
            total_bm25_candidates=len(bm25_results),
        )

    async def _safe_vector(self, query: str, top_k: int, domain: str | None = None) -> list[dict]:
        try:
            return await self._vector.search(query, top_k=top_k, domain=domain)
        except VectorSearchError:
            raise
        except Exception as e:
            raise VectorSearchError(f"Unexpected error: {e}") from e

    async def _safe_keyword(self, query: str, top_k: int, domain: str | None = None) -> list[dict]:
        try:
            return await self._keyword.search(query, top_k=top_k, domain=domain)
        except KeywordSearchError:
            raise
        except Exception as e:
            raise KeywordSearchError(f"Unexpected error: {e}") from e
