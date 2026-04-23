"""
Integration tests — SupervisorAgent routing.

TestSupervisorGraphWiring  — LangGraph state machine + mocked agents (no LLM)
TestSupervisorRealRouting  — real LLM routing decisions (requires OPENAI_API_KEY)
"""

import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import requires_llm
from agents.supervisor import build_graph, CortexState, RoutingDecision, SupervisorAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_fn(agent_name: str):
    def fn(state: dict) -> dict:
        return {
            "answer": f"Answer from {agent_name}",
            "sources": [],
            "log_trace": state.get("log_trace", []) + [{"node": agent_name}],
        }
    return fn


def _initial_state(query: str, tier: str = "internal") -> CortexState:
    return {
        "query": query,
        "session_id": "test-session",
        "user_id": "test-user",
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


def _graph():
    return build_graph(
        knowledge_fn=_make_agent_fn("knowledge"),
        research_fn=_make_agent_fn("research"),
        action_fn=_make_agent_fn("action"),
    )


# ---------------------------------------------------------------------------
# Graph wiring tests — routing decision mocked, rate/budget logic tested
# ---------------------------------------------------------------------------

class TestSupervisorGraphWiring:
    """Tests the LangGraph structure and guard rails without calling the LLM."""

    def _invoke(self, agent: str, tier: str = "internal", confidence: float = 0.9):
        decision = RoutingDecision(agent=agent, confidence=confidence, reasoning="test")
        with patch("agents.supervisor._rate_limiter") as mock_rl, \
             patch("agents.supervisor._cost_tracker") as mock_ct, \
             patch("agents.supervisor.SupervisorAgent.route", return_value=decision):
            mock_rl.check_rate_limit.return_value = (True, None)
            mock_ct.check_budget.return_value = True
            return _graph().invoke(_initial_state("test query", tier))

    def test_routes_to_knowledge(self):
        result = self._invoke("knowledge")
        assert result["answer"] == "Answer from knowledge"

    def test_routes_to_research(self):
        result = self._invoke("research")
        assert result["answer"] == "Answer from research"

    def test_routes_to_action(self):
        result = self._invoke("action")
        assert result["answer"] == "Answer from action"

    def test_log_trace_contains_supervisor_entry(self):
        result = self._invoke("knowledge")
        supervisor_entries = [e for e in result["log_trace"] if e.get("node") == "supervisor"]
        assert len(supervisor_entries) == 1
        assert supervisor_entries[0]["agent"] == "knowledge"

    def test_state_preserves_user_metadata(self):
        result = self._invoke("knowledge")
        assert result["user_id"] == "test-user"
        assert result["user_tier"] == "internal"

    def test_rate_limited_sets_error(self):
        with patch("agents.supervisor._rate_limiter") as mock_rl, \
             patch("agents.supervisor._cost_tracker") as mock_ct:
            mock_rl.check_rate_limit.return_value = (False, 30)
            mock_ct.check_budget.return_value = True
            result = _graph().invoke(_initial_state("query"))
        assert result["error"] is not None
        assert "rate limit" in result["error"].lower()

    def test_budget_exceeded_sets_error(self):
        with patch("agents.supervisor._rate_limiter") as mock_rl, \
             patch("agents.supervisor._cost_tracker") as mock_ct:
            mock_rl.check_rate_limit.return_value = (True, None)
            mock_ct.check_budget.return_value = False
            result = _graph().invoke(_initial_state("query"))
        assert result["error"] is not None
        assert "budget" in result["error"].lower()

    def test_agent_crash_returns_error_answer(self):
        def crashing_fn(state):
            raise RuntimeError("agent crashed")

        graph = build_graph(
            knowledge_fn=crashing_fn,
            research_fn=crashing_fn,
            action_fn=crashing_fn,
        )
        decision = RoutingDecision(agent="knowledge", confidence=0.9, reasoning="test")
        with patch("agents.supervisor._rate_limiter") as mock_rl, \
             patch("agents.supervisor._cost_tracker") as mock_ct, \
             patch("agents.supervisor.SupervisorAgent.route", return_value=decision):
            mock_rl.check_rate_limit.return_value = (True, None)
            mock_ct.check_budget.return_value = True
            result = graph.invoke(_initial_state("query"))
        assert result["error"] is not None
        assert "unavailable" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Real LLM routing tests
# ---------------------------------------------------------------------------

@requires_llm
class TestSupervisorRealRouting:
    """
    Calls the real SupervisorAgent.route() to verify routing decisions.
    Agent nodes are still stubbed so we don't need the full pipeline.
    """

    @pytest.fixture(scope="class")
    def supervisor(self):
        return SupervisorAgent()

    def _invoke_with_real_routing(self, query: str, tier: str = "internal"):
        with patch("agents.supervisor._rate_limiter") as mock_rl, \
             patch("agents.supervisor._cost_tracker") as mock_ct:
            mock_rl.check_rate_limit.return_value = (True, None)
            mock_ct.check_budget.return_value = True
            return _graph().invoke(_initial_state(query, tier))

    def test_hr_query_routes_to_knowledge(self):
        result = self._invoke_with_real_routing("What is the PTO policy?")
        assert result["agent"] == "knowledge"

    def test_web_search_query_routes_to_research(self):
        result = self._invoke_with_real_routing("What are the latest AI model benchmarks in 2026?")
        assert result["agent"] == "research"

    def test_ticket_creation_routes_to_action(self):
        result = self._invoke_with_real_routing("Create a support ticket for broken SSO login")
        assert result["agent"] == "action"

    def test_calendar_query_routes_to_action(self):
        result = self._invoke_with_real_routing("Check when the engineering team is free this week")
        assert result["agent"] in ("action", "knowledge")

    def test_routing_sets_confidence(self):
        result = self._invoke_with_real_routing("What are the company values?")
        assert 0.0 < result["confidence"] <= 1.0

    def test_routing_sets_reasoning(self):
        result = self._invoke_with_real_routing("How do I request a workplace accommodation?")
        assert len(result["reasoning"]) > 0
