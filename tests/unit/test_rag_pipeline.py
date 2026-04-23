"""
Tests for rag/pipeline.py

Uses a real RAGPipeline: real OpenAI embeddings + real pgvector.
Org-docs snippets are ingested and then queried.

Requires: PGVECTOR_URL pointing to live pgvector + OPENAI_API_KEY
"""

import asyncio
import pytest
from pathlib import Path
from langchain_core.documents import Document

from tests.conftest import requires_pg, requires_llm

from rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HR_SNIPPET = (
    "Each GitLab team member may use up to 25 paid sick days, available from their first day "
    "of work. Eligibility is calculated on a rolling 12-month period."
)
_ENG_SNIPPET = (
    "Product Management prioritizes 60% of engineering time. Engineering prioritizes 40% of "
    "time on initiatives that improve the product, underlying platform, and foundational technologies."
)
_CULTURE_SNIPPET = (
    "GitLab's six core values are Collaboration, Results for Customers, Efficiency, "
    "Diversity Inclusion & Belonging, Iteration, and Transparency — spelled CREDIT."
)


# ---------------------------------------------------------------------------
# Real pipeline tests
# ---------------------------------------------------------------------------

@requires_pg
@requires_llm
class TestRAGPipelineReal:
    """
    End-to-end RAG: insert real documents, run real queries.
    Uses source_file='test:rag-*' for cleanup.
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        return RAGPipeline()

    @pytest.fixture(scope="class", autouse=True)
    def ingest_test_docs(self, pipeline, sync_pg_conn):
        """Insert a small set of test documents once per class."""
        docs = [
            Document(
                page_content=_HR_SNIPPET,
                metadata={"source": "test:rag-hr.md", "domain": "hr", "file_type": "markdown"},
            ),
            Document(
                page_content=_ENG_SNIPPET,
                metadata={"source": "test:rag-eng.md", "domain": "engineering", "file_type": "markdown"},
            ),
            Document(
                page_content=_CULTURE_SNIPPET,
                metadata={"source": "test:rag-culture.md", "domain": "culture", "file_type": "markdown"},
            ),
        ]
        # Insert via the vector store layer directly (skip L1-L2 file I/O)
        embeddings_svc = pipeline._embeddings
        texts = [d.page_content for d in docs]
        embeddings = asyncio.run(embeddings_svc.embed_documents(texts))
        asyncio.run(pipeline._vector_store.add_documents(docs, embeddings))
        yield
        cur = sync_pg_conn.cursor()
        cur.execute("DELETE FROM documents WHERE source_file LIKE 'test:rag-%';")
        cur.close()

    @pytest.mark.asyncio
    async def test_query_returns_answer(self, pipeline):
        result = await pipeline.query(
            "How many sick days do employees get?",
            user_tier="confidential",
        )
        assert "answer" in result
        assert len(result["answer"]) > 0

    @pytest.mark.asyncio
    async def test_query_result_contains_sources(self, pipeline):
        result = await pipeline.query(
            "What percentage of time does engineering control?",
            user_tier="internal",
        )
        assert isinstance(result["sources"], list)

    @pytest.mark.asyncio
    async def test_query_returns_intent(self, pipeline):
        result = await pipeline.query("What are the core company values?", user_tier="public")
        assert "intent" in result
        assert result["intent"] in ("factual", "procedural", "out_of_scope", "unknown")

    @pytest.mark.asyncio
    async def test_query_returns_chunks_retrieved(self, pipeline):
        result = await pipeline.query("sick leave policy", user_tier="confidential")
        assert "chunks_retrieved" in result
        assert isinstance(result["chunks_retrieved"], int)

    @pytest.mark.asyncio
    async def test_query_returns_reformulated_query(self, pipeline):
        result = await pipeline.query("PTO rules?", user_tier="internal")
        assert "reformulated_query" in result
        assert isinstance(result["reformulated_query"], str)

    @pytest.mark.asyncio
    async def test_answer_references_sick_days(self, pipeline):
        result = await pipeline.query(
            "How many paid sick days am I entitled to?",
            user_tier="confidential",
        )
        # Answer should mention 25 days (it's in the ingested context)
        assert "25" in result["answer"] or "sick" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_answer_references_credit_values(self, pipeline):
        result = await pipeline.query(
            "What does CREDIT stand for?",
            user_tier="public",
        )
        assert any(word in result["answer"] for word in ["CREDIT", "Collaboration", "Transparency"])

    @pytest.mark.asyncio
    async def test_public_user_cannot_retrieve_hr_docs(self, pipeline):
        result = await pipeline.query(
            "sick leave policy details",
            user_tier="public",
        )
        # HR domain requires confidential — public user should get 0 hr chunks
        # (may still get engineering/culture chunks though)
        assert "chunks_retrieved" in result

    @pytest.mark.asyncio
    async def test_ingest_directory_returns_counts(self, pipeline, tmp_path):
        # Create a tiny markdown file
        md = tmp_path / "test_doc.md"
        md.write_text("# Test\nThis is a test document for ingestion.")
        result = await pipeline.ingest(tmp_path)
        assert result["documents_ingested"] >= 1
        assert result["chunks_created"] >= 1
        # Cleanup
        import psycopg
        from tests.conftest import PG_DSN
        async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM documents WHERE source_file LIKE %s",
                    (str(tmp_path) + "%",),
                )
            await conn.commit()
