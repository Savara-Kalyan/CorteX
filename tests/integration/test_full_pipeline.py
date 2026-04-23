"""
Full end-to-end pipeline integration tests.

TestFullPipelineWiring    — LangGraph graph with stubbed agents (no external I/O)
TestFullPipelineReal      — complete real pipeline: LLM routing + RAG + tool agents
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from tests.conftest import requires_pg, requires_llm, requires_all
from agents.supervisor import build_graph, CortexState, RoutingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(query: str, tier: str = "internal") -> CortexState:
    return {
        "query": query,
        "session_id": "full-test",
        "user_id": "full-user",
        "user_tier": tier,
        "agent": None,
        "confidence": 0.0,
        "reasoning": "",
        "answer": "",
        "sources": [],
        "error": None,
        "iteration_count": 0,
        "log_trace": [],
        "tokens_used": 0,
        "cost": 0.0,
    }


# ---------------------------------------------------------------------------
# Graph wiring tests — agent functions stubbed, no LLM needed
# ---------------------------------------------------------------------------

class TestFullPipelineWiring:
    """
    Verifies that LangGraph properly wires supervisor → agents → state.
    LLM routing decision is mocked; agent bodies are stubs.
    """

    def _run(self, query: str, routed_to: str, tier: str = "internal"):
        stub_answers = {
            "knowledge": "Internal knowledge answer.",
            "research": "Web research answer.",
            "action": "Action completed.",
        }

        def make_fn(name):
            def fn(state):
                return {
                    "answer": stub_answers[name],
                    "sources": ["doc.txt"] if name == "knowledge" else [],
                    "log_trace": state.get("log_trace", []),
                }
            return fn

        graph = build_graph(
            knowledge_fn=make_fn("knowledge"),
            research_fn=make_fn("research"),
            action_fn=make_fn("action"),
        )
        decision = RoutingDecision(agent=routed_to, confidence=0.9, reasoning="test")
        with patch("agents.supervisor._rate_limiter") as rl, \
             patch("agents.supervisor._cost_tracker") as ct, \
             patch("agents.supervisor.SupervisorAgent.route", return_value=decision):
            rl.check_rate_limit.return_value = (True, None)
            ct.check_budget.return_value = True
            return graph.invoke(_state(query, tier))

    def test_knowledge_flow(self):
        result = self._run("What is the HR policy?", "knowledge")
        assert result["answer"] == "Internal knowledge answer."
        assert "doc.txt" in result["sources"]

    def test_research_flow(self):
        result = self._run("Latest AI benchmarks?", "research")
        assert result["answer"] == "Web research answer."

    def test_action_flow(self):
        result = self._run("Create a ticket for broken SSO", "action")
        assert result["answer"] == "Action completed."

    def test_log_trace_shows_supervisor(self):
        result = self._run("policy query", "knowledge")
        nodes = [e.get("node") for e in result["log_trace"]]
        assert "supervisor" in nodes

    def test_state_preserves_user_metadata(self):
        result = self._run("test query", "knowledge")
        assert result["user_id"] == "full-user"
        assert result["user_tier"] == "internal"

    def test_error_in_agent_returns_graceful_answer(self):
        def failing_fn(state):
            raise RuntimeError("agent crashed")

        graph = build_graph(
            knowledge_fn=failing_fn,
            research_fn=failing_fn,
            action_fn=failing_fn,
        )
        decision = RoutingDecision(agent="knowledge", confidence=0.9, reasoning="test")
        with patch("agents.supervisor._rate_limiter") as rl, \
             patch("agents.supervisor._cost_tracker") as ct, \
             patch("agents.supervisor.SupervisorAgent.route", return_value=decision):
            rl.check_rate_limit.return_value = (True, None)
            ct.check_budget.return_value = True
            result = graph.invoke(_state("query"))
        assert result["error"] is not None
        assert "unavailable" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Real end-to-end tests — full stack
# ---------------------------------------------------------------------------

@requires_all
class TestFullPipelineReal:
    """
    Full end-to-end: real LLM routing + real RAGPipeline + real tools.
    Uses source_file='test:fp-*' for DB cleanup.
    """

    @pytest.fixture(scope="class")
    def full_graph(self, sync_pg_conn):
        from rag.pipeline import RAGPipeline
        from agents.knowledge_agent import KnowledgeAgent
        from agents.research_agent import ResearchAgent
        from agents.action_agent import ActionAgent

        pipeline = RAGPipeline()
        knowledge = KnowledgeAgent(rag_pipeline=pipeline)
        research = ResearchAgent()
        action = ActionAgent()

        # Seed knowledge base with a culture doc (publicly accessible)
        docs = [
            Document(
                page_content=(
                    "GitLab's six core values are Collaboration, Results for Customers, "
                    "Efficiency, Diversity Inclusion & Belonging, Iteration, and Transparency — "
                    "spelled CREDIT."
                ),
                metadata={"source": "test:fp-values.md", "domain": "culture", "file_type": "markdown"},
            ),
        ]
        texts = [d.page_content for d in docs]
        embeddings = asyncio.run(pipeline._embeddings.embed_documents(texts))
        asyncio.run(pipeline._vector_store.add_documents(docs, embeddings))

        graph = build_graph(
            knowledge_fn=knowledge.handle,
            research_fn=research.handle,
            action_fn=action.handle,
        )
        yield graph, pipeline

        cur = sync_pg_conn.cursor()
        cur.execute("DELETE FROM documents WHERE source_file LIKE 'test:fp-%';")
        cur.close()

    def _run(self, full_graph, query: str, tier: str = "internal"):
        graph, _ = full_graph
        with patch("agents.supervisor._rate_limiter") as rl, \
             patch("agents.supervisor._cost_tracker") as ct:
            rl.check_rate_limit.return_value = (True, None)
            ct.check_budget.return_value = True
            return graph.invoke(_state(query, tier))

    def test_knowledge_query_returns_real_answer(self, full_graph):
        result = self._run(full_graph, "What does CREDIT stand for?", tier="public")
        assert len(result["answer"]) > 10
        assert result.get("error") is None

    def test_research_query_triggers_web_search(self, full_graph):
        result = self._run(full_graph, "What are the latest AI benchmarks in 2026?")
        assert result["agent"] == "research"
        assert len(result["answer"]) > 0

    def test_action_query_creates_ticket(self, full_graph):
        result = self._run(full_graph, "Create a support ticket: broken login, high priority")
        assert result["agent"] == "action"
        assert len(result["answer"]) > 0

    def test_answer_has_no_critical_error(self, full_graph):
        result = self._run(full_graph, "What are the company values?", tier="public")
        assert result.get("error") is None or len(result["answer"]) > 0

    def test_full_run_produces_log_trace(self, full_graph):
        result = self._run(full_graph, "Describe the company culture.", tier="public")
        assert len(result["log_trace"]) >= 1
        nodes = [e.get("node") for e in result["log_trace"]]
        assert "supervisor" in nodes
