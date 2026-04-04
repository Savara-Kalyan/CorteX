import hashlib
import os
import pytest
import psycopg

# Must be set before importing any module that instantiates EmbeddingService at class level.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from rag.vector_store.model import DocumentInsert
from rag.vector_store.service import VectorStoreService
from rag.vector_store.pgvector import PGVectorStore
from rag.vector_store.qdrant import QdrantVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(content: str = "Sample content.", **meta) -> Document:
    base = {"source": "test.pdf", "page": 1, "file_type": "pdf"}
    return Document(page_content=content, metadata={**base, **meta})


def _make_pg_conn_mock():
    """Return (conn_mock, cur_mock) async-context-manager pair for psycopg."""
    cur = AsyncMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=None)
    cur.fetchall = AsyncMock(return_value=[("content", "test.pdf", 1, {}, 0.1)])

    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    conn.cursor = MagicMock(return_value=cur)
    conn.commit = AsyncMock()

    return conn, cur


def _make_qdrant_client_mock():
    """Return an AsyncQdrantClient-like async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[MagicMock(id="1", score=0.9)])
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure VectorStoreService singleton is cleared before and after each test."""
    VectorStoreService._instance = None
    yield
    VectorStoreService._instance = None


@pytest.fixture
def pg_store():
    store = PGVectorStore()
    store.embed_model = MagicMock()
    store.embed_model.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return store


@pytest.fixture
def qdrant_store():
    store = QdrantVectorStore()
    store.embed_model = MagicMock()
    store.embed_model.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return store


# ---------------------------------------------------------------------------
# DocumentInsert model
# ---------------------------------------------------------------------------

class TestDocumentInsert:

    def test_doc_hash_is_sha256_of_content(self):
        row = DocumentInsert(
            content="hello",
            embedding=[0.1],
            chunk_index=0,
            total_chunks=1,
            created_by="test",
            chunk_length=5,
        )
        expected = hashlib.sha256("hello".encode()).hexdigest()
        assert row.doc_hash == expected

    def test_doc_hash_differs_for_different_content(self):
        def _row(content):
            return DocumentInsert(
                content=content,
                embedding=[0.1],
                chunk_index=0,
                total_chunks=1,
                created_by="test",
                chunk_length=len(content),
            )

        assert _row("foo").doc_hash != _row("bar").doc_hash

    def test_defaults(self):
        row = DocumentInsert(
            content="x",
            embedding=[0.0],
            chunk_index=0,
            total_chunks=1,
            created_by="test",
            chunk_length=1,
        )
        assert row.access_level == "internal"
        assert row.doc_type == "unknown"
        assert row.chunk_type == "text"
        assert row.extraction_method == "docling"
        assert row.extraction_confidence == 0.95
        assert row.metadata == {}


# ---------------------------------------------------------------------------
# VectorStoreService
# ---------------------------------------------------------------------------

class TestVectorStoreService:

    def test_loads_pgvector_backend(self):
        with patch("rag.vector_store.service.settings") as mock_settings, \
             patch("rag.vector_store.pgvector.PGVectorStore") as MockPG:
            mock_settings.vector_store.provider = "pgvector"
            svc = VectorStoreService()
            MockPG.assert_called_once()
            assert svc._backend is MockPG.return_value

    def test_loads_qdrant_backend(self):
        with patch("rag.vector_store.service.settings") as mock_settings, \
             patch("rag.vector_store.qdrant.QdrantVectorStore") as MockQdrant:
            mock_settings.vector_store.provider = "qdrant"
            svc = VectorStoreService()
            MockQdrant.assert_called_once()
            assert svc._backend is MockQdrant.return_value

    def test_unsupported_provider_raises(self):
        with patch("rag.vector_store.service.settings") as mock_settings:
            mock_settings.vector_store.provider = "weaviate"
            with pytest.raises(ValueError, match="Unsupported vector store provider"):
                VectorStoreService()

    def test_singleton_returns_same_instance(self):
        with patch("rag.vector_store.service.settings") as mock_settings, \
             patch("rag.vector_store.pgvector.PGVectorStore"):
            mock_settings.vector_store.provider = "pgvector"
            svc1 = VectorStoreService()
            svc2 = VectorStoreService()
            assert svc1 is svc2

    def test_singleton_resets_on_init_failure(self):
        with patch("rag.vector_store.service.settings") as mock_settings:
            mock_settings.vector_store.provider = "bad"
            with pytest.raises(ValueError):
                VectorStoreService()
        assert VectorStoreService._instance is None

    @pytest.mark.asyncio
    async def test_insert_delegates_to_backend(self):
        mock_backend = MagicMock()
        mock_backend.insert = AsyncMock()
        docs = [_make_doc()]

        with patch.object(VectorStoreService, "_load_backend", return_value=mock_backend):
            svc = VectorStoreService()
            await svc.insert(docs)
            mock_backend.insert.assert_awaited_once_with(docs)

    @pytest.mark.asyncio
    async def test_insert_propagates_backend_exception(self):
        mock_backend = MagicMock()
        mock_backend.insert = AsyncMock(side_effect=RuntimeError("db down"))

        with patch.object(VectorStoreService, "_load_backend", return_value=mock_backend):
            svc = VectorStoreService()
            with pytest.raises(RuntimeError, match="db down"):
                await svc.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=["result1"])

        with patch.object(VectorStoreService, "_load_backend", return_value=mock_backend):
            svc = VectorStoreService()
            results = await svc.search([0.1, 0.2], top_k=1)
            assert results == ["result1"]

    @pytest.mark.asyncio
    async def test_search_propagates_backend_exception(self):
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(side_effect=RuntimeError("search failed"))

        with patch.object(VectorStoreService, "_load_backend", return_value=mock_backend):
            svc = VectorStoreService()
            with pytest.raises(RuntimeError, match="search failed"):
                await svc.search([0.1, 0.2])


# ---------------------------------------------------------------------------
# PGVectorStore
# ---------------------------------------------------------------------------

class TestPGVectorStore:

    @pytest.mark.asyncio
    async def test_insert_happy_path(self, pg_store):
        conn, cur = _make_pg_conn_mock()
        docs = [_make_doc()]

        with patch("rag.vector_store.pgvector.psycopg.AsyncConnection.connect", new=AsyncMock(return_value=conn)), \
             patch("rag.vector_store.pgvector.register_vector_async", new=AsyncMock()):
            await pg_store.insert(docs)

        cur.execute.assert_awaited_once()
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_insert_embeds_all_docs(self, pg_store):
        docs = [_make_doc("A"), _make_doc("B")]
        pg_store.embed_model.embed_documents = AsyncMock(return_value=[[0.1], [0.2]])
        conn, _ = _make_pg_conn_mock()

        with patch("rag.vector_store.pgvector.psycopg.AsyncConnection.connect", new=AsyncMock(return_value=conn)), \
             patch("rag.vector_store.pgvector.register_vector_async", new=AsyncMock()):
            await pg_store.insert(docs)

        pg_store.embed_model.embed_documents.assert_awaited_once_with(docs=docs)

    @pytest.mark.asyncio
    async def test_insert_raises_on_embedding_failure(self, pg_store):
        pg_store.embed_model.embed_documents = AsyncMock(side_effect=RuntimeError("embed down"))

        with pytest.raises(RuntimeError, match="Failed to generate embeddings before PGVector insert"):
            await pg_store.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_insert_raises_on_connection_error(self, pg_store):
        with patch(
            "rag.vector_store.pgvector.psycopg.AsyncConnection.connect",
            new=AsyncMock(side_effect=psycopg.OperationalError("conn refused")),
        ):
            with pytest.raises(RuntimeError, match="Could not connect to PostgreSQL"):
                await pg_store.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_insert_raises_on_database_error(self, pg_store):
        conn, cur = _make_pg_conn_mock()
        cur.execute = AsyncMock(side_effect=psycopg.DatabaseError("constraint violation"))

        with patch("rag.vector_store.pgvector.psycopg.AsyncConnection.connect", new=AsyncMock(return_value=conn)), \
             patch("rag.vector_store.pgvector.register_vector_async", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="Failed to insert documents into PostgreSQL"):
                await pg_store.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_search_happy_path(self, pg_store):
        conn, cur = _make_pg_conn_mock()

        with patch("rag.vector_store.pgvector.psycopg.AsyncConnection.connect", new=AsyncMock(return_value=conn)), \
             patch("rag.vector_store.pgvector.register_vector_async", new=AsyncMock()):
            results = await pg_store.search([0.1, 0.2, 0.3], top_k=1)

        assert results == [("content", "test.pdf", 1, {}, 0.1)]
        cur.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_raises_on_connection_error(self, pg_store):
        with patch(
            "rag.vector_store.pgvector.psycopg.AsyncConnection.connect",
            new=AsyncMock(side_effect=psycopg.OperationalError("timeout")),
        ):
            with pytest.raises(RuntimeError, match="Could not connect to PostgreSQL"):
                await pg_store.search([0.1])

    @pytest.mark.asyncio
    async def test_search_raises_on_database_error(self, pg_store):
        conn, cur = _make_pg_conn_mock()
        cur.execute = AsyncMock(side_effect=psycopg.DatabaseError("bad query"))

        with patch("rag.vector_store.pgvector.psycopg.AsyncConnection.connect", new=AsyncMock(return_value=conn)), \
             patch("rag.vector_store.pgvector.register_vector_async", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="Failed to search documents in PostgreSQL"):
                await pg_store.search([0.1])


# ---------------------------------------------------------------------------
# QdrantVectorStore
# ---------------------------------------------------------------------------

class TestQdrantVectorStore:

    @pytest.mark.asyncio
    async def test_insert_happy_path(self, qdrant_store):
        client = _make_qdrant_client_mock()
        docs = [_make_doc()]

        with patch.object(qdrant_store, "_client", return_value=client):
            await qdrant_store.insert(docs)

        client.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_insert_point_ids_are_deterministic(self, qdrant_store):
        """Same content always maps to the same Qdrant point id."""
        client = _make_qdrant_client_mock()
        docs = [_make_doc("Deterministic content.")]

        with patch.object(qdrant_store, "_client", return_value=client):
            await qdrant_store.insert(docs)
            first_points = client.upsert.call_args.kwargs["points"]

        client.upsert.reset_mock()

        with patch.object(qdrant_store, "_client", return_value=client):
            await qdrant_store.insert(docs)
            second_points = client.upsert.call_args.kwargs["points"]

        assert first_points[0].id == second_points[0].id

    @pytest.mark.asyncio
    async def test_insert_raises_on_embedding_failure(self, qdrant_store):
        qdrant_store.embed_model.embed_documents = AsyncMock(side_effect=RuntimeError("api error"))

        with pytest.raises(RuntimeError, match="Failed to generate embeddings before Qdrant insert"):
            await qdrant_store.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_insert_raises_on_unexpected_response(self, qdrant_store):
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = _make_qdrant_client_mock()
        client.upsert = AsyncMock(side_effect=UnexpectedResponse(
            status_code=404, reason_phrase=b"Not Found", content=b"collection missing", headers={}
        ))

        with patch.object(qdrant_store, "_client", return_value=client):
            with pytest.raises(RuntimeError, match="Qdrant returned an unexpected response during insert"):
                await qdrant_store.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_insert_raises_on_generic_exception(self, qdrant_store):
        client = _make_qdrant_client_mock()
        client.upsert = AsyncMock(side_effect=ConnectionError("refused"))

        with patch.object(qdrant_store, "_client", return_value=client):
            with pytest.raises(RuntimeError, match="Failed to insert documents into Qdrant"):
                await qdrant_store.insert([_make_doc()])

    @pytest.mark.asyncio
    async def test_search_happy_path(self, qdrant_store):
        client = _make_qdrant_client_mock()

        with patch.object(qdrant_store, "_client", return_value=client):
            results = await qdrant_store.search([0.1, 0.2, 0.3], top_k=3)

        assert len(results) == 1
        client.search.assert_awaited_once()
        _, kwargs = client.search.call_args
        assert kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_search_passes_collection_name(self, qdrant_store):
        client = _make_qdrant_client_mock()

        with patch("rag.vector_store.qdrant.settings") as mock_settings, \
             patch.object(qdrant_store, "_client", return_value=client):
            mock_settings.vector_store.collection_name = "my_collection"
            await qdrant_store.search([0.1])

        _, kwargs = client.search.call_args
        assert kwargs["collection_name"] == "my_collection"

    @pytest.mark.asyncio
    async def test_search_raises_on_unexpected_response(self, qdrant_store):
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = _make_qdrant_client_mock()
        client.search = AsyncMock(side_effect=UnexpectedResponse(
            status_code=500, reason_phrase=b"Internal Server Error", content=b"error", headers={}
        ))

        with patch.object(qdrant_store, "_client", return_value=client):
            with pytest.raises(RuntimeError, match="Qdrant returned an unexpected response during search"):
                await qdrant_store.search([0.1])

    @pytest.mark.asyncio
    async def test_search_raises_on_generic_exception(self, qdrant_store):
        client = _make_qdrant_client_mock()
        client.search = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch.object(qdrant_store, "_client", return_value=client):
            with pytest.raises(RuntimeError, match="Failed to search documents in Qdrant"):
                await qdrant_store.search([0.1])
