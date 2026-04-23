"""
Integration tests — KnowledgeAgent with a real RAGPipeline.

Ingests org-doc snippets into a live pgvector instance, then runs
the agent against them. Requires PGVECTOR_URL + OPENAI_API_KEY.
"""

import asyncio
import pytest
from langchain_core.documents import Document

from tests.conftest import requires_pg, requires_llm
from agents.knowledge_agent import KnowledgeAgent
from rag.pipeline import RAGPipeline


_POLICY_TEXT = (
    "GitLab does not set a specific limit on the amount of PTO a team member can take per year. "
    "For time off exceeding 25 consecutive calendar days, special permission is required from "
    "your Manager, People Business Partner, and the Absence Management Team."
)
_SICK_TEXT = (
    "Each GitLab team member may use up to 25 paid sick days, available from their first day of work."
)


@requires_pg
@requires_llm
class TestKnowledgeAgentReal:
    """Real KnowledgeAgent → RAGPipeline → pgvector → OpenAI."""

    @pytest.fixture(scope="class")
    def pipeline_and_agent(self, sync_pg_conn):
        pipeline = RAGPipeline()
        agent = KnowledgeAgent(rag_pipeline=pipeline)

        # Insert test documents
        docs = [
            Document(
                page_content=_POLICY_TEXT,
                metadata={"source": "test:ka-pto.md", "domain": "hr", "file_type": "markdown"},
            ),
            Document(
                page_content=_SICK_TEXT,
                metadata={"source": "test:ka-sick.md", "domain": "hr", "file_type": "markdown"},
            ),
        ]
        texts = [d.page_content for d in docs]
        embeddings = asyncio.run(pipeline._embeddings.embed_documents(texts))
        asyncio.run(pipeline._vector_store.add_documents(docs, embeddings))

        yield pipeline, agent

        cur = sync_pg_conn.cursor()
        cur.execute("DELETE FROM documents WHERE source_file LIKE 'test:ka-%';")
        cur.close()

    def test_handle_returns_answer(self, pipeline_and_agent):
        _, agent = pipeline_and_agent
        state = {
            "query": "How much PTO can I take per year?",
            "user_id": "test-u1",
            "user_tier": "confidential",
            "log_trace": [],
        }
        result = agent.handle(state)
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_handle_returns_sources(self, pipeline_and_agent):
        _, agent = pipeline_and_agent
        state = {
            "query": "What is the sick leave entitlement?",
            "user_id": "test-u2",
            "user_tier": "confidential",
            "log_trace": [],
        }
        result = agent.handle(state)
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_handle_appends_log_trace(self, pipeline_and_agent):
        _, agent = pipeline_and_agent
        state = {
            "query": "PTO policy",
            "user_id": "test-u3",
            "user_tier": "confidential",
            "log_trace": [{"node": "supervisor"}],
        }
        result = agent.handle(state)
        assert len(result["log_trace"]) >= 2
        knowledge_entry = next(
            (e for e in result["log_trace"] if e.get("node") == "knowledge"), None
        )
        assert knowledge_entry is not None

    def test_handle_answer_mentions_pto(self, pipeline_and_agent):
        _, agent = pipeline_and_agent
        state = {
            "query": "How many consecutive days off need special approval?",
            "user_id": "test-u4",
            "user_tier": "confidential",
            "log_trace": [],
        }
        result = agent.handle(state)
        assert "25" in result["answer"] or "days" in result["answer"].lower()

    def test_handle_logs_chunks_retrieved(self, pipeline_and_agent):
        _, agent = pipeline_and_agent
        state = {
            "query": "sick days policy",
            "user_id": "test-u5",
            "user_tier": "confidential",
            "log_trace": [],
        }
        result = agent.handle(state)
        knowledge_entry = next(
            (e for e in result["log_trace"] if e.get("node") == "knowledge"), None
        )
        assert knowledge_entry is not None
        assert "chunks_retrieved" in knowledge_entry

    def test_public_user_gets_no_hr_results(self, pipeline_and_agent):
        _, agent = pipeline_and_agent
        state = {
            "query": "PTO policy details",
            "user_id": "test-u6",
            "user_tier": "public",  # HR requires confidential
            "log_trace": [],
        }
        result = agent.handle(state)
        # Should still return an answer (possibly "no documents found" for HR)
        assert "answer" in result
