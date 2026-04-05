"""
Access Control Service - Domain-filtered semantic search with role-based access control.

Architecture:
  - AccessPolicy    : maps domains to required access levels; enforces who can query what
  - IndexManager    : idempotently creates all DB indexes needed for fast filtered search
  - AccessControlService : async facade — embed query, enforce policy, run filtered search

Domain → access-level defaults (configurable via AccessPolicy):
  hr          → confidential
  engineering → internal
  culture     → public
  general     → internal

Access level hierarchy (least → most privileged):
  public < internal < confidential < restricted
"""

import asyncio
import logging

import psycopg
from pgvector.psycopg import register_vector_async

from settings import settings
from rag.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Access level ordering
# ---------------------------------------------------------------------------

_ACCESS_LEVELS: list[str] = ["public", "internal", "confidential", "restricted"]

# Default domain → minimum access level required to read that domain
_DOMAIN_ACCESS_MAP: dict[str, str] = {
    "hr": "confidential",
    "engineering": "internal",
    "culture": "public",
    "general": "internal",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AccessPolicy
# ---------------------------------------------------------------------------


class AccessPolicy:
    """
    Determines which domains a given access level can query.

    Args:
        domain_access_map: Mapping of domain → minimum access level required.
                           Defaults to module-level _DOMAIN_ACCESS_MAP.
        access_levels:     Ordered list from least to most privileged.
                           Defaults to module-level _ACCESS_LEVELS.
    """

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
            domain
            for domain, required in self._domain_access_map.items()
            if self._rank(user_access_level) >= self._rank(required)
        ]


# ---------------------------------------------------------------------------
# IndexManager
# ---------------------------------------------------------------------------


class IndexManager:
    """
    Idempotently creates all indexes required for domain-filtered vector search.

    Indexes:
      - HNSW on embedding (cosine)   → fast approximate nearest-neighbour
      - GIN  on metadata             → fast JSONB key/value lookups
      - B-tree on metadata->>'domain' → fast domain equality filter
      - B-tree on access_level        → fast access level filter
    """

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
        """Create all indexes if they do not already exist."""
        for name, ddl in self._INDEXES:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(ddl)
                await conn.commit()
                logger.info("Index ensured: name=%s", name)
            except psycopg.DatabaseError as e:
                logger.error("Index creation failed: name=%s error=%s", name, e)
                raise IndexCreationError(name, str(e)) from e


# ---------------------------------------------------------------------------
# AccessControlService
# ---------------------------------------------------------------------------


class AccessControlService:
    """
    Domain-filtered semantic search with role-based access control.

    All public methods are async. The service is safe to instantiate once and
    reuse — it holds no per-request state.

    Example::

        svc = AccessControlService()
        await svc.ensure_indexes()

        # Single-domain search
        results = await svc.search_by_domain(
            query="parental leave policy",
            domain="hr",
            user_access_level="confidential",
        )

        # All accessible domains merged and re-ranked
        results = await svc.search_accessible(
            query="career growth",
            user_access_level="internal",
        )
    """

    def __init__(
        self,
        embed_service: EmbeddingService | None = None,
        policy: AccessPolicy | None = None,
    ) -> None:
        self._embed = embed_service or EmbeddingService()
        self._policy = policy or AccessPolicy()
        self._index_manager = IndexManager()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    async def _connect(self) -> psycopg.AsyncConnection:  # type: ignore[type-arg]
        try:
            conn = await psycopg.AsyncConnection.connect(
                settings.vector_store.connection_string
            )
            await register_vector_async(conn)
            return conn
        except psycopg.OperationalError as e:
            raise DatabaseConnectionError(str(e)) from e

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create all required database indexes (idempotent)."""
        logger.info("Ensuring access control indexes.")
        async with await self._connect() as conn:
            await self._index_manager.ensure_indexes(conn)
        logger.info("All indexes verified.")

    # ------------------------------------------------------------------
    # Search: single domain
    # ------------------------------------------------------------------

    async def search_by_domain(
        self,
        query: str,
        domain: str,
        user_access_level: str = "internal",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Semantic search within a single domain, enforcing access control.

        Args:
            query:             Natural-language query string.
            domain:            Target domain (e.g. "hr", "engineering", "culture").
            user_access_level: Caller's access level.
            top_k:             Maximum results to return.

        Returns:
            List of result dicts with keys:
            ``id``, ``content``, ``access_level``, ``source_file``,
            ``domain``, ``similarity``.

        Raises:
            PermissionDeniedError: when user_access_level is insufficient.
            SearchError:           on embedding or query failure.
            DatabaseConnectionError: on DB connectivity issues.
        """
        if not self._policy.can_access_domain(domain, user_access_level):
            required = self._policy.required_level_for(domain)
            raise PermissionDeniedError(domain, required, user_access_level)

        logger.info(
            "Searching domain: domain=%s user_level=%s top_k=%s",
            domain, user_access_level, top_k,
        )

        try:
            query_embedding = await self._embed.embed_query(query)
        except Exception as e:
            logger.error("Embedding failed during access-control search: error=%s", e)
            raise SearchError(f"Embedding generation failed: {e}") from e

        try:
            async with await self._connect() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT
                            id,
                            content,
                            access_level,
                            source_file,
                            metadata->>'domain'          AS domain,
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
            logger.error("Search query failed: domain=%s error=%s", domain, e)
            raise SearchError(f"Query execution failed: {e}") from e

        results = [
            {
                "id": r[0],
                "content": r[1],
                "access_level": r[2],
                "source_file": r[3],
                "domain": r[4],
                "similarity": float(r[5]) if r[5] is not None else 0.0,
            }
            for r in rows
        ]
        logger.info(
            "Domain search complete: domain=%s result_count=%s", domain, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Search: multiple domains in parallel
    # ------------------------------------------------------------------

    async def search_multi_domain(
        self,
        query: str,
        domains: list[str],
        user_access_level: str = "internal",
        top_k_per_domain: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Run the same query against multiple domains concurrently.

        Domains the user cannot access are silently skipped (logged at WARNING).
        Individual domain failures are logged at ERROR and return an empty list —
        they do not abort the entire search.

        Returns:
            Mapping of domain → result list.
        """
        async def _safe_search(domain: str) -> tuple[str, list[dict]]:
            if not self._policy.can_access_domain(domain, user_access_level):
                logger.warning(
                    "Skipping inaccessible domain: domain=%s user_level=%s",
                    domain, user_access_level,
                )
                return domain, []
            try:
                results = await self.search_by_domain(
                    query, domain, user_access_level, top_k=top_k_per_domain
                )
                return domain, results
            except AccessControlError as e:
                logger.error("Domain search failed: domain=%s error=%s", domain, e)
                return domain, []

        pairs = await asyncio.gather(*[_safe_search(d) for d in domains])
        return dict(pairs)

    # ------------------------------------------------------------------
    # Search: all accessible domains, merged
    # ------------------------------------------------------------------

    async def search_accessible(
        self,
        query: str,
        user_access_level: str = "internal",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search across all domains accessible to the user and return results
        merged and re-ranked by cosine similarity.

        Args:
            query:             Natural-language query string.
            user_access_level: Caller's access level.
            top_k:             Total results to return after merging.

        Returns:
            List of result dicts (same shape as search_by_domain), ranked by similarity.
        """
        accessible = self._policy.accessible_domains(user_access_level)
        if not accessible:
            logger.warning(
                "No accessible domains for user_level=%s", user_access_level
            )
            return []

        logger.info(
            "Searching accessible domains: domains=%s user_level=%s",
            accessible, user_access_level,
        )

        per_domain = await self.search_multi_domain(
            query,
            domains=accessible,
            user_access_level=user_access_level,
            top_k_per_domain=top_k,
        )

        merged = [result for results in per_domain.values() for result in results]
        merged.sort(key=lambda r: r["similarity"], reverse=True)
        return merged[:top_k]
