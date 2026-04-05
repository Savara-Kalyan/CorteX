"""
Semantic vector search using pgvector cosine distance.

Mirrors the query pattern used in AccessControlService so the two can be
composed cleanly. Keeps the embedding step explicit here rather than
delegating to VectorStoreService so that we get the id column back (the
base VectorStoreService.search does not return it).
"""

import logging

import psycopg
from pgvector.psycopg import register_vector_async

from settings import settings
from rag.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VectorSearchError(Exception):
    """Base exception for all vector-search errors."""


class VectorConnectionError(VectorSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[VectorSearch] DB connection failed: {reason}")


class VectorEmbeddingError(VectorSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[VectorSearch] Embedding generation failed: {reason}")


class VectorQueryError(VectorSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[VectorSearch] Query execution failed: {reason}")


# ---------------------------------------------------------------------------
# VectorSearcher
# ---------------------------------------------------------------------------


class VectorSearcher:
    """
    Semantic similarity search over the documents table using pgvector.

    The query is embedded via the configured EmbeddingService, then a
    cosine-distance ORDER BY pulls the top_k nearest neighbours.

    Result dicts contain:
        id, content, source_file, page_number, metadata, vector_score
    where vector_score = 1 − cosine_distance (higher is more similar).
    """

    def __init__(self, embed_service: EmbeddingService | None = None) -> None:
        self._embed = embed_service or EmbeddingService()

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        try:
            conn = await psycopg.AsyncConnection.connect(
                settings.vector_store.connection_string
            )
            await register_vector_async(conn)
            return conn
        except psycopg.OperationalError as e:
            raise VectorConnectionError(str(e)) from e

    async def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Embed the query and return the top_k most similar documents.

        Args:
            query:  Natural-language query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of result dicts ordered by descending cosine similarity.

        Raises:
            VectorEmbeddingError:  if the embedding model call fails.
            VectorConnectionError: on DB connectivity failure.
            VectorQueryError:      on SQL execution failure.
        """
        if not query.strip():
            logger.warning("VectorSearcher.search called with empty query — returning []")
            return []

        logger.info("Vector search: query=%r top_k=%s", query, top_k)

        try:
            query_embedding = await self._embed.embed_query(query)
        except Exception as e:
            logger.error("Embedding failed: error=%s query=%r", e, query)
            raise VectorEmbeddingError(str(e)) from e

        try:
            async with await self._connect() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT
                            id::text,
                            content,
                            source_file,
                            page_number,
                            metadata,
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
            logger.error("Vector search query failed: error=%s query=%r", e, query)
            raise VectorQueryError(str(e)) from e

        results = [
            {
                "id": row[0],
                "content": row[1],
                "source_file": row[2],
                "page_number": row[3],
                "metadata": row[4] or {},
                "vector_score": float(row[5]) if row[5] is not None else 0.0,
            }
            for row in rows
        ]
        logger.info("Vector search complete: result_count=%s", len(results))
        return results
