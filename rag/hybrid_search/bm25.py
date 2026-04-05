"""
Keyword search using PostgreSQL full-text search (tsvector / ts_rank_cd).

PostgreSQL's FTS is BM25-flavoured under the hood and runs directly on the
documents table — no in-memory corpus needed, scales with the DB.

ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', query))
gives a normalised relevance score that increases when query terms appear
more frequently and in smaller documents, closely matching BM25 intuition.
"""

import logging

import psycopg
from pgvector.psycopg import register_vector_async

from settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KeywordSearchError(Exception):
    """Base exception for all keyword-search errors."""


class KeywordConnectionError(KeywordSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[KeywordSearch] DB connection failed: {reason}")


class KeywordQueryError(KeywordSearchError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[KeywordSearch] Query execution failed: {reason}")


# ---------------------------------------------------------------------------
# KeywordSearcher
# ---------------------------------------------------------------------------


class KeywordSearcher:
    """
    BM25-style keyword search backed by PostgreSQL full-text search.

    Each call opens a fresh async connection (consistent with the existing
    PGVectorStore and AccessControlService patterns in this codebase).

    Result dicts contain:
        id, content, source_file, page_number, metadata, bm25_score
    """

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        try:
            conn = await psycopg.AsyncConnection.connect(
                settings.vector_store.connection_string
            )
            await register_vector_async(conn)
            return conn
        except psycopg.OperationalError as e:
            raise KeywordConnectionError(str(e)) from e

    async def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Full-text search against the documents table.

        Args:
            query:  Raw natural-language query (passed to plainto_tsquery).
            top_k:  Maximum number of results to return.

        Returns:
            List of result dicts ordered by descending BM25 score.
            Documents with no matching terms are excluded entirely.

        Raises:
            KeywordConnectionError: on DB connectivity failure.
            KeywordQueryError:      on SQL execution failure.
        """
        if not query.strip():
            logger.warning("KeywordSearcher.search called with empty query — returning []")
            return []

        logger.info("Keyword search: query=%r top_k=%s", query, top_k)

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
                            ts_rank_cd(
                                to_tsvector('english', content),
                                plainto_tsquery('english', %s)
                            ) AS bm25_score
                        FROM documents
                        WHERE to_tsvector('english', content)
                              @@ plainto_tsquery('english', %s)
                        ORDER BY bm25_score DESC
                        LIMIT %s;
                        """,
                        (query, query, top_k),
                    )
                    rows = await cur.fetchall()

        except psycopg.OperationalError as e:
            raise KeywordConnectionError(str(e)) from e
        except psycopg.DatabaseError as e:
            logger.error("Keyword search query failed: error=%s query=%r", e, query)
            raise KeywordQueryError(str(e)) from e

        results = [
            {
                "id": row[0],
                "content": row[1],
                "source_file": row[2],
                "page_number": row[3],
                "metadata": row[4] or {},
                "bm25_score": float(row[5]) if row[5] is not None else 0.0,
            }
            for row in rows
        ]
        logger.info("Keyword search complete: result_count=%s", len(results))
        return results
