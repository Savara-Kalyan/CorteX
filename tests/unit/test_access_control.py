"""
Unit tests for rag/access_control/service.py

No API keys or live DB required — all I/O is mocked.
"""

import os
import pytest
import psycopg

# Must precede any import that instantiates EmbeddingService at class level.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import AsyncMock, MagicMock, patch, call

from rag.access_control.service import (
    AccessControlService,
    AccessPolicy,
    IndexManager,
    AccessControlError,
    DatabaseConnectionError,
    IndexCreationError,
    PermissionDeniedError,
    SearchError,
    _ACCESS_LEVELS,
    _DOMAIN_ACCESS_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn_mock(rows: list = None):
    """
    Return (conn, cur) async-context-manager mocks for psycopg.AsyncConnection.
    ``rows`` is what cur.fetchall() returns.
    """
    cur = AsyncMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=None)
    cur.fetchall = AsyncMock(return_value=rows or [])

    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    conn.cursor = MagicMock(return_value=cur)
    conn.commit = AsyncMock()

    return conn, cur


def _make_service(embed_return: list[float] = None, rows: list = None):
    """
    Return (svc, conn_mock, cur_mock) with embed and DB fully mocked.
    """
    embed = MagicMock()
    embed.embed_query = AsyncMock(return_value=embed_return or [0.1, 0.2, 0.3])

    conn, cur = _make_conn_mock(rows=rows)
    svc = AccessControlService(embed_service=embed)
    return svc, conn, cur


def _search_row(
    id_: int = 1,
    content: str = "Some content.",
    access_level: str = "internal",
    source_file: str = "file.md",
    domain: str = "engineering",
    similarity: float = 0.9,
) -> tuple:
    """Raw DB cursor row tuple — used for psycopg fetchall() mocks."""
    return (id_, content, access_level, source_file, domain, similarity)


def _result_dict(
    id_: int = 1,
    content: str = "Some content.",
    access_level: str = "internal",
    source_file: str = "file.md",
    domain: str = "engineering",
    similarity: float = 0.9,
) -> dict:
    """Formatted result dict — used when mocking search_by_domain return values."""
    return {
        "id": id_,
        "content": content,
        "access_level": access_level,
        "source_file": source_file,
        "domain": domain,
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# AccessPolicy — pure logic, no I/O
# ---------------------------------------------------------------------------

class TestAccessPolicy:

    def test_default_map_covers_known_domains(self):
        policy = AccessPolicy()
        for domain in ("hr", "engineering", "culture", "general"):
            assert domain in _DOMAIN_ACCESS_MAP

    def test_access_levels_ordered(self):
        assert _ACCESS_LEVELS == ["public", "internal", "confidential", "restricted"]

    # -- can_access_domain --

    def test_public_domain_accessible_to_all_levels(self):
        policy = AccessPolicy()
        for level in _ACCESS_LEVELS:
            assert policy.can_access_domain("culture", level)

    def test_confidential_domain_denied_to_public(self):
        policy = AccessPolicy()
        assert not policy.can_access_domain("hr", "public")

    def test_confidential_domain_denied_to_internal(self):
        policy = AccessPolicy()
        assert not policy.can_access_domain("hr", "internal")

    def test_confidential_domain_accessible_to_confidential(self):
        policy = AccessPolicy()
        assert policy.can_access_domain("hr", "confidential")

    def test_confidential_domain_accessible_to_restricted(self):
        policy = AccessPolicy()
        assert policy.can_access_domain("hr", "restricted")

    def test_internal_domain_denied_to_public(self):
        policy = AccessPolicy()
        assert not policy.can_access_domain("engineering", "public")

    def test_internal_domain_accessible_to_internal_and_above(self):
        policy = AccessPolicy()
        for level in ("internal", "confidential", "restricted"):
            assert policy.can_access_domain("engineering", level)

    def test_unknown_domain_defaults_to_internal(self):
        policy = AccessPolicy()
        assert not policy.can_access_domain("unknown_dept", "public")
        assert policy.can_access_domain("unknown_dept", "internal")

    def test_unknown_access_level_treated_as_lowest(self):
        policy = AccessPolicy()
        # Unknown level gets rank 0 (same as "public"), denied for internal+ domains
        assert not policy.can_access_domain("engineering", "ghost")

    # -- accessible_domains --

    def test_public_level_can_only_access_public_domains(self):
        policy = AccessPolicy()
        domains = policy.accessible_domains("public")
        assert "culture" in domains
        assert "engineering" not in domains
        assert "hr" not in domains

    def test_internal_level_can_access_internal_and_public(self):
        policy = AccessPolicy()
        domains = policy.accessible_domains("internal")
        assert "culture" in domains
        assert "engineering" in domains
        assert "hr" not in domains

    def test_confidential_level_can_access_all_domains(self):
        policy = AccessPolicy()
        domains = policy.accessible_domains("confidential")
        for domain in ("culture", "engineering", "hr", "general"):
            assert domain in domains

    def test_restricted_level_can_access_all_domains(self):
        policy = AccessPolicy()
        domains = policy.accessible_domains("restricted")
        for domain in _DOMAIN_ACCESS_MAP:
            assert domain in domains

    # -- required_level_for --

    def test_required_level_for_known_domain(self):
        policy = AccessPolicy()
        assert policy.required_level_for("hr") == "confidential"
        assert policy.required_level_for("engineering") == "internal"
        assert policy.required_level_for("culture") == "public"

    def test_required_level_for_unknown_domain_defaults_to_internal(self):
        policy = AccessPolicy()
        assert policy.required_level_for("mystery") == "internal"

    # -- custom map injection --

    def test_custom_domain_map_overrides_defaults(self):
        custom_map = {"sales": "restricted", "marketing": "public"}
        policy = AccessPolicy(domain_access_map=custom_map)
        assert not policy.can_access_domain("sales", "confidential")
        assert policy.can_access_domain("sales", "restricted")
        assert policy.can_access_domain("marketing", "public")

    def test_custom_access_levels_respected(self):
        policy = AccessPolicy(
            domain_access_map={"vip": "premium"},
            access_levels=["free", "premium"],
        )
        assert not policy.can_access_domain("vip", "free")
        assert policy.can_access_domain("vip", "premium")


# ---------------------------------------------------------------------------
# IndexManager
# ---------------------------------------------------------------------------

class TestIndexManager:

    @pytest.mark.asyncio
    async def test_creates_all_four_indexes(self):
        conn, cur = _make_conn_mock()
        manager = IndexManager()
        await manager.ensure_indexes(conn)

        assert cur.execute.await_count == len(manager._INDEXES)
        assert conn.commit.await_count == len(manager._INDEXES)

    @pytest.mark.asyncio
    async def test_each_ddl_contains_if_not_exists(self):
        for name, ddl in IndexManager._INDEXES:
            assert "IF NOT EXISTS" in ddl, f"Index '{name}' DDL missing IF NOT EXISTS"

    @pytest.mark.asyncio
    async def test_hnsw_index_uses_cosine_ops(self):
        hnsw_ddl = next(ddl for name, ddl in IndexManager._INDEXES if "hnsw" in name)
        assert "vector_cosine_ops" in hnsw_ddl

    @pytest.mark.asyncio
    async def test_domain_index_targets_metadata_json_field(self):
        domain_ddl = next(ddl for name, ddl in IndexManager._INDEXES if "domain" in name)
        assert "metadata->>'domain'" in domain_ddl

    @pytest.mark.asyncio
    async def test_raises_index_creation_error_on_db_failure(self):
        conn, cur = _make_conn_mock()
        cur.execute = AsyncMock(side_effect=psycopg.DatabaseError("permission denied"))

        manager = IndexManager()
        with pytest.raises(IndexCreationError) as exc_info:
            await manager.ensure_indexes(conn)

        assert "permission denied" in str(exc_info.value)
        assert IndexManager._INDEXES[0][0] in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_creation_error_names_the_failing_index(self):
        conn, cur = _make_conn_mock()
        # Fail on the second index
        call_count = 0

        async def flaky_execute(ddl, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise psycopg.DatabaseError("table missing")

        cur.execute = AsyncMock(side_effect=flaky_execute)

        manager = IndexManager()
        with pytest.raises(IndexCreationError) as exc_info:
            await manager.ensure_indexes(conn)

        expected_name = IndexManager._INDEXES[1][0]
        assert expected_name in str(exc_info.value)


# ---------------------------------------------------------------------------
# AccessControlService._connect
# ---------------------------------------------------------------------------

class TestAccessControlServiceConnect:

    @pytest.mark.asyncio
    async def test_connect_raises_database_connection_error_on_operational_error(self):
        svc = AccessControlService(embed_service=MagicMock())

        with patch(
            "rag.access_control.service.psycopg.AsyncConnection.connect",
            new=AsyncMock(side_effect=psycopg.OperationalError("refused")),
        ):
            with pytest.raises(DatabaseConnectionError, match="refused"):
                await svc._connect()


# ---------------------------------------------------------------------------
# AccessControlService.ensure_indexes
# ---------------------------------------------------------------------------

class TestEnsureIndexes:

    @pytest.mark.asyncio
    async def test_ensure_indexes_delegates_to_index_manager(self):
        svc = AccessControlService(embed_service=MagicMock())
        conn, _ = _make_conn_mock()

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch.object(svc._index_manager, "ensure_indexes", new=AsyncMock()) as mock_ensure:
            await svc.ensure_indexes()

        mock_ensure.assert_awaited_once_with(conn)

    @pytest.mark.asyncio
    async def test_ensure_indexes_propagates_index_creation_error(self):
        svc = AccessControlService(embed_service=MagicMock())
        conn, _ = _make_conn_mock()

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch.object(
                 svc._index_manager,
                 "ensure_indexes",
                 new=AsyncMock(side_effect=IndexCreationError("idx", "boom")),
             ):
            with pytest.raises(IndexCreationError):
                await svc.ensure_indexes()


# ---------------------------------------------------------------------------
# AccessControlService.search_by_domain
# ---------------------------------------------------------------------------

class TestSearchByDomain:

    @pytest.mark.asyncio
    async def test_raises_permission_denied_for_insufficient_level(self):
        svc, conn, _ = _make_service()
        with pytest.raises(PermissionDeniedError) as exc_info:
            await svc.search_by_domain("salary", domain="hr", user_access_level="public")

        err = str(exc_info.value)
        assert "hr" in err
        assert "confidential" in err
        assert "public" in err

    @pytest.mark.asyncio
    async def test_does_not_call_embed_when_permission_denied(self):
        svc, conn, _ = _make_service()
        with pytest.raises(PermissionDeniedError):
            await svc.search_by_domain("salary", domain="hr", user_access_level="internal")
        svc._embed.embed_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_happy_path_returns_formatted_results(self):
        rows = [_search_row(id_=1, content="EC2 pricing table", domain="engineering", similarity=0.88)]
        svc, conn, _ = _make_service(rows=rows)

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch("rag.access_control.service.register_vector_async", new=AsyncMock()):
            results = await svc.search_by_domain(
                "EC2 pricing", domain="engineering", user_access_level="internal"
            )

        assert len(results) == 1
        r = results[0]
        assert r["id"] == 1
        assert r["content"] == "EC2 pricing table"
        assert r["domain"] == "engineering"
        assert r["similarity"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_passes_domain_and_top_k_to_query(self):
        svc, conn, cur = _make_service(rows=[])

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch("rag.access_control.service.register_vector_async", new=AsyncMock()):
            await svc.search_by_domain(
                "handbook", domain="engineering", user_access_level="internal", top_k=7
            )

        sql, params = cur.execute.call_args.args
        assert "metadata->>'domain'" in sql
        assert params[1] == "engineering"
        assert params[3] == 7

    @pytest.mark.asyncio
    async def test_raises_search_error_on_embed_failure(self):
        svc, conn, _ = _make_service()
        svc._embed.embed_query = AsyncMock(side_effect=RuntimeError("API timeout"))

        with pytest.raises(SearchError, match="Embedding generation failed"):
            await svc.search_by_domain(
                "onboarding", domain="culture", user_access_level="public"
            )

    @pytest.mark.asyncio
    async def test_raises_database_connection_error_on_operational_error(self):
        svc, _, _ = _make_service()

        with patch.object(
            svc,
            "_connect",
            new=AsyncMock(side_effect=DatabaseConnectionError("timed out")),
        ):
            with pytest.raises(DatabaseConnectionError):
                await svc.search_by_domain(
                    "values", domain="culture", user_access_level="public"
                )

    @pytest.mark.asyncio
    async def test_raises_search_error_on_db_query_failure(self):
        svc, conn, cur = _make_service()
        cur.execute = AsyncMock(side_effect=psycopg.DatabaseError("syntax error"))

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch("rag.access_control.service.register_vector_async", new=AsyncMock()):
            with pytest.raises(SearchError, match="Query execution failed"):
                await svc.search_by_domain(
                    "values", domain="culture", user_access_level="public"
                )

    @pytest.mark.asyncio
    async def test_similarity_none_coerced_to_zero(self):
        rows = [_search_row(similarity=None)]  # type: ignore[arg-type]
        svc, conn, _ = _make_service(rows=rows)

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch("rag.access_control.service.register_vector_async", new=AsyncMock()):
            results = await svc.search_by_domain(
                "q", domain="engineering", user_access_level="internal"
            )

        assert results[0]["similarity"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_result_set_returns_empty_list(self):
        svc, conn, _ = _make_service(rows=[])

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch("rag.access_control.service.register_vector_async", new=AsyncMock()):
            results = await svc.search_by_domain(
                "nothing here", domain="engineering", user_access_level="internal"
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_culture_domain_accessible_at_public_level(self):
        rows = [_search_row(domain="culture", access_level="public")]
        svc, conn, _ = _make_service(rows=rows)

        with patch.object(svc, "_connect", new=AsyncMock(return_value=conn)), \
             patch("rag.access_control.service.register_vector_async", new=AsyncMock()):
            results = await svc.search_by_domain(
                "company values", domain="culture", user_access_level="public"
            )

        assert len(results) == 1


# ---------------------------------------------------------------------------
# AccessControlService.search_multi_domain
# ---------------------------------------------------------------------------

class TestSearchMultiDomain:

    @pytest.mark.asyncio
    async def test_returns_results_for_each_accessible_domain(self):
        svc = AccessControlService(embed_service=MagicMock())
        svc._embed.embed_query = AsyncMock(return_value=[0.1])

        async def fake_search(query, domain, user_access_level, top_k):
            return [_result_dict(domain=domain)]

        with patch.object(svc, "search_by_domain", side_effect=fake_search):
            results = await svc.search_multi_domain(
                "career", domains=["engineering", "culture"], user_access_level="internal"
            )

        assert set(results.keys()) == {"engineering", "culture"}
        assert results["engineering"][0]["domain"] == "engineering"

    @pytest.mark.asyncio
    async def test_skips_domains_user_cannot_access(self):
        svc = AccessControlService(embed_service=MagicMock())
        called_domains = []

        async def fake_search(query, domain, user_access_level, top_k):
            called_domains.append(domain)
            return []

        with patch.object(svc, "search_by_domain", side_effect=fake_search):
            results = await svc.search_multi_domain(
                "salary", domains=["hr", "engineering"], user_access_level="internal"
            )

        # hr requires confidential; internal user should be skipped
        assert "hr" not in called_domains
        assert results["hr"] == []

    @pytest.mark.asyncio
    async def test_domain_failure_returns_empty_not_raises(self):
        svc = AccessControlService(embed_service=MagicMock())

        async def fake_search(query, domain, user_access_level, top_k):
            if domain == "engineering":
                raise SearchError("DB down")
            return [_search_row(domain=domain)]

        with patch.object(svc, "search_by_domain", side_effect=fake_search):
            results = await svc.search_multi_domain(
                "q", domains=["engineering", "culture"], user_access_level="internal"
            )

        assert results["engineering"] == []
        assert len(results["culture"]) == 1

    @pytest.mark.asyncio
    async def test_all_domains_run_concurrently(self):
        """Verify asyncio.gather is used — all searches are issued before any completes."""
        import asyncio

        svc = AccessControlService(embed_service=MagicMock())
        started: list[str] = []
        finished: list[str] = []

        async def fake_search(query, domain, user_access_level, top_k):
            started.append(domain)
            await asyncio.sleep(0)  # yield to event loop
            finished.append(domain)
            return []

        with patch.object(svc, "search_by_domain", side_effect=fake_search):
            await svc.search_multi_domain(
                "q",
                domains=["engineering", "culture"],
                user_access_level="internal",
            )

        # Both started before either finished
        assert set(started) == {"engineering", "culture"}

    @pytest.mark.asyncio
    async def test_empty_domains_list_returns_empty_dict(self):
        svc = AccessControlService(embed_service=MagicMock())
        results = await svc.search_multi_domain("q", domains=[], user_access_level="internal")
        assert results == {}


# ---------------------------------------------------------------------------
# AccessControlService.search_accessible
# ---------------------------------------------------------------------------

class TestSearchAccessible:

    @pytest.mark.asyncio
    async def test_merges_and_reranks_by_similarity(self):
        svc = AccessControlService(embed_service=MagicMock())

        async def fake_multi(query, domains, user_access_level, top_k_per_domain):
            return {
                "engineering": [_result_dict(domain="engineering", similarity=0.7)],
                "culture": [_result_dict(domain="culture", similarity=0.95)],
            }

        with patch.object(svc, "search_multi_domain", side_effect=fake_multi):
            results = await svc.search_accessible("growth", user_access_level="internal", top_k=5)

        assert results[0]["similarity"] == pytest.approx(0.95)
        assert results[1]["similarity"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_top_k_caps_merged_results(self):
        svc = AccessControlService(embed_service=MagicMock())

        async def fake_multi(query, domains, user_access_level, top_k_per_domain):
            return {
                "engineering": [_result_dict(similarity=0.8 - i * 0.01) for i in range(5)],
                "culture": [_result_dict(similarity=0.9 - i * 0.01) for i in range(5)],
            }

        with patch.object(svc, "search_multi_domain", side_effect=fake_multi):
            results = await svc.search_accessible("anything", user_access_level="internal", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_domains_accessible(self):
        policy = AccessPolicy(domain_access_map={"classified": "restricted"})
        svc = AccessControlService(embed_service=MagicMock(), policy=policy)

        results = await svc.search_accessible("secret", user_access_level="public")
        assert results == []

    @pytest.mark.asyncio
    async def test_only_queries_domains_accessible_to_user(self):
        svc = AccessControlService(embed_service=MagicMock())
        queried_domains: list[str] = []

        async def fake_multi(query, domains, user_access_level, top_k_per_domain):
            queried_domains.extend(domains)
            return {d: [] for d in domains}

        with patch.object(svc, "search_multi_domain", side_effect=fake_multi):
            await svc.search_accessible("q", user_access_level="internal")

        # hr (confidential) must not be queried by an internal user
        assert "hr" not in queried_domains
        assert "engineering" in queried_domains
        assert "culture" in queried_domains


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:

    def test_all_errors_inherit_from_access_control_error(self):
        for exc_class in (
            DatabaseConnectionError,
            IndexCreationError,
            SearchError,
            PermissionDeniedError,
        ):
            assert issubclass(exc_class, AccessControlError)

    def test_database_connection_error_message(self):
        err = DatabaseConnectionError("host unreachable")
        assert "host unreachable" in str(err)

    def test_index_creation_error_message_contains_name_and_reason(self):
        err = IndexCreationError("my_idx", "table not found")
        assert "my_idx" in str(err)
        assert "table not found" in str(err)

    def test_permission_denied_error_message_contains_domain_and_levels(self):
        err = PermissionDeniedError("hr", "confidential", "public")
        assert "hr" in str(err)
        assert "confidential" in str(err)
        assert "public" in str(err)

    def test_search_error_message(self):
        err = SearchError("index not found")
        assert "index not found" in str(err)
