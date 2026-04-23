import asyncio
import logging

import psycopg
from pgvector.psycopg import register_vector_async

from settings import settings
from rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

_ACCESS_LEVELS: list[str] = ["public", "internal", "confidential", "restricted"]

_DOMAIN_ACCESS_MAP: dict[str, str] = {
    "hr": "confidential",
    "engineering": "internal",
    "culture": "public",
    "general": "internal",
}


class AccessControlError(Exception):
    """Base exception for all access-control errors."""


class DatabaseConnectionError(AccessControlError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[AccessControl] Database connection failed: {reason}")


class IndexCreationError(AccessControlError):
    def __init__(self, index_name: str, reason: str) -> None:
        super().__init__(f"[AccessControl] Failed to create index '{index_name}': {reason}")


class SearchError(AccessControlError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[AccessControl] Search failed: {reason}")


class PermissionDeniedError(AccessControlError):
    def __init__(self, domain: str, required_level: str, user_level: str) -> None:
        super().__init__(
            f"[AccessControl] Access denied: domain='{domain}' "
            f"requires '{required_level}', user has '{user_level}'"
        )


class AccessPolicy:

    def __init__(
        self,
        domain_access_map: dict[str, str] | None = None,
        access_levels: list[str] | None = None,
    ) -> None:
        self._domain_access_map = domain_access_map or _DOMAIN_ACCESS_MAP
        self._levels = access_levels or _ACCESS_LEVELS

    def _rank(self, level: str) -> int:
        try:
            return self._levels.index(level)
        except ValueError:
            logger.warning("Unknown access level '%s', treating as lowest privilege.", level)
            return 0

    def can_access_domain(self, domain: str, user_access_level: str) -> bool:
        required = self._domain_access_map.get(domain, "internal")
        return self._rank(user_access_level) >= self._rank(required)

    def required_level_for(self, domain: str) -> str:
        return self._domain_access_map.get(domain, "internal")

    def accessible_domains(self, user_access_level: str) -> list[str]:
        return [
            domain for domain, required in self._domain_access_map.items()
            if self._rank(user_access_level) >= self._rank(required)
        ]


class IndexManager:

    _INDEXES: list[tuple[str, str]] = [
        (
            "documents_hnsw_idx",
            "CREATE INDEX IF NOT EXISTS documents_hnsw_idx "
            "ON documents USING hnsw (embedding vector_cosine_ops);",
        ),
        (
            "documents_metadata_gin",
            "CREATE INDEX IF NOT EXISTS documents_metadata_gin "
            "ON documents USING gin (metadata);",
        ),
        (
            "documents_domain_expr_idx",
            "CREATE INDEX IF NOT EXISTS documents_domain_expr_idx "
            "ON documents ((metadata->>'domain'));",
        ),
        (
            "documents_access_level_idx",
            "CREATE INDEX IF NOT EXISTS documents_access_level_idx "
            "ON documents (access_level);",
        ),
    ]

    async def ensure_indexes(self, conn: psycopg.AsyncConnection) -> None:  # type: ignore[type-arg]
        for name, ddl in self._INDEXES:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(ddl)
                await conn.commit()
                logger.info("Index ensured: name=%s", name)
            except psycopg.DatabaseError as e:
                logger.error("Index creation failed: name=%s error=%s", name, e)
                raise IndexCreationError(name, str(e)) from e


class AccessControlService:

    def __init__(
        self,
        embed_service: EmbeddingService | None = None,
        policy: AccessPolicy | None = None,
    ) -> None:
        self._embed = embed_service or EmbeddingService()
        self._policy = policy or AccessPolicy()
        self._index_manager = IndexManager()

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        try:
            conn = await psycopg.AsyncConnection.connect(settings.vector_store.connection_string)
            await register_vector_async(conn)
            return conn
        except psycopg.OperationalError as e:
            raise DatabaseConnectionError(str(e)) from e

    async def ensure_indexes(self) -> None:
        async with await self._connect() as conn:
            await self._index_manager.ensure_indexes(conn)

    async def filter(self, chunks: list, user_tier: str = "internal") -> list:
        """Post-retrieval access filter for pipeline use."""
        return [
            c for c in chunks
            if self._policy.can_access_domain(
                c.metadata.get("domain", "general"), user_tier
            )
        ]

    async def search_by_domain(
        self,
        query: str,
        domain: str,
        user_access_level: str = "internal",
        top_k: int = 5,
    ) -> list[dict]:
        if not self._policy.can_access_domain(domain, user_access_level):
            required = self._policy.required_level_for(domain)
            raise PermissionDeniedError(domain, required, user_access_level)

        try:
            query_embedding = await self._embed.embed_query(query)
        except Exception as e:
            raise SearchError(f"Embedding generation failed: {e}") from e

        try:
            async with await self._connect() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT id, content, access_level, source_file,
                               metadata->>'domain' AS domain,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM documents
                        WHERE metadata->>'domain' = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (query_embedding, domain, query_embedding, top_k),
                    )
                    rows = await cur.fetchall()
        except psycopg.OperationalError as e:
            raise DatabaseConnectionError(str(e)) from e
        except psycopg.DatabaseError as e:
            raise SearchError(f"Query execution failed: {e}") from e

        return [
            {"id": r[0], "content": r[1], "access_level": r[2], "source_file": r[3],
             "domain": r[4], "similarity": float(r[5]) if r[5] is not None else 0.0}
            for r in rows
        ]

    async def search_multi_domain(
        self,
        query: str,
        domains: list[str],
        user_access_level: str = "internal",
        top_k_per_domain: int = 3,
    ) -> dict[str, list[dict]]:
        async def _safe(domain: str) -> tuple[str, list[dict]]:
            if not self._policy.can_access_domain(domain, user_access_level):
                return domain, []
            try:
                return domain, await self.search_by_domain(query, domain, user_access_level, top_k=top_k_per_domain)
            except AccessControlError:
                return domain, []

        return dict(await asyncio.gather(*[_safe(d) for d in domains]))

    async def search_accessible(
        self,
        query: str,
        user_access_level: str = "internal",
        top_k: int = 5,
    ) -> list[dict]:
        accessible = self._policy.accessible_domains(user_access_level)
        if not accessible:
            return []
        per_domain = await self.search_multi_domain(query, accessible, user_access_level, top_k)
        merged = [r for results in per_domain.values() for r in results]
        merged.sort(key=lambda r: r["similarity"], reverse=True)
        return merged[:top_k]
