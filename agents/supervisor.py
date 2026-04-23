from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Literal, Optional, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from reliability.fallback import CircuitBreaker, CircuitBreakerOpen
from reliability.rate_limiter import RateLimiter, retry_with_backoff
from reliability.cost_tracker import CostTracker
from observability.logger import get_logger

logger = get_logger(__name__)


class RoutingDecision(BaseModel):
    agent: Literal["knowledge", "research", "action"] = Field(
        description="Target specialist agent"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Routing confidence")
    reasoning: str = Field(description="Why this agent was chosen")


class CortexState(TypedDict):
    query: str
    session_id: str
    user_id: str
    user_tier: Literal["public", "internal", "confidential", "restricted"]
    agent: Optional[str]
    confidence: float
    reasoning: str
    answer: str
    sources: list
    error: Optional[str]
    iteration_count: int
    log_trace: list
    tokens_used: int
    cost: float


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(agent_name: str, version: str = "v1.0.0") -> dict:
    path = _PROMPTS_DIR / agent_name / f"{version}.yaml"
    if not path.exists():
        path = _PROMPTS_DIR / agent_name / "v1.0.0.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compile_supervisor_prompt(prompt_data: dict) -> str:
    role = prompt_data.get("role", {})
    security = prompt_data.get("security", {})
    context = prompt_data.get("context", {})
    examples = prompt_data.get("examples", [])

    agents_desc = "\n".join(
        f"  - {name}: {info.get('description', '')} | triggers: {info.get('triggers', [])}"
        for name, info in context.get("agents", {}).items()
    )
    examples_text = "\n".join(
        f"  Query: {e['user']}\n  → agent={e['correct_response']['agent']} "
        f"confidence={e['correct_response']['confidence']}"
        for e in examples
    )

    return (
        f"{security.get('top_guard', '')}\n\n"
        f"ROLE: {role.get('identity', '')}\n"
        f"EXPERTISE: {role.get('expertise', '')}\n\n"
        f"AVAILABLE AGENTS:\n{agents_desc}\n\n"
        f"EXAMPLES:\n{examples_text}\n\n"
        f"{security.get('bottom_guard', '')}"
    )


_rate_limiter = RateLimiter()
_cost_tracker = CostTracker()
_circuit_breaker = CircuitBreaker(max_failures=3, timeout=60)


class SupervisorAgent:
    def __init__(self, prompt_version: str = "v1.0.0"):
        self._llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3).with_structured_output(
            RoutingDecision
        )
        self._prompt_version = prompt_version

    @retry_with_backoff(max_retries=2, initial_delay=1.0, backoff_factor=2.0)
    def route(self, query: str, user_id: str) -> RoutingDecision:
        prompt_data = _load_prompt("supervisor", self._prompt_version)
        system_prompt = _compile_supervisor_prompt(prompt_data)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        return _circuit_breaker.call(lambda: self._llm.invoke(messages))


def build_graph(
    knowledge_fn,
    research_fn,
    action_fn,
    prompt_version: str = "v1.0.0",
):
    supervisor = SupervisorAgent(prompt_version=prompt_version)
    workflow = StateGraph(CortexState)

    async def supervisor_node(state: CortexState) -> dict:
        logger.info("supervisor routing query", query=state["query"][:80], user_id=state["user_id"])
        t0 = time.time()

        allowed, retry_after = await _rate_limiter.check_rate_limit(
            state["user_id"], max_requests=20, window_seconds=60
        )
        if not allowed:
            return {"error": f"Rate limit exceeded. Retry in {retry_after}s", "agent": "knowledge"}

        if not await _cost_tracker.check_budget(state["user_id"], daily_limit=2.0):
            return {"error": "Daily budget exceeded", "agent": "knowledge"}

        try:
            decision = supervisor.route(state["query"], state["user_id"])
            latency = (time.time() - t0) * 1000
            logger.info(
                "routing decision",
                agent=decision.agent,
                confidence=decision.confidence,
                latency_ms=latency,
            )
            return {
                "agent": decision.agent,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": state.get("log_trace", []) + [{
                    "node": "supervisor",
                    "agent": decision.agent,
                    "confidence": decision.confidence,
                    "latency_ms": latency,
                }],
            }
        except CircuitBreakerOpen as exc:
            logger.error("supervisor circuit open", error=str(exc))
            return {"error": str(exc), "agent": "knowledge"}
        except Exception as exc:
            logger.error("supervisor error", error=str(exc))
            return {"error": str(exc), "agent": "knowledge"}

    async def knowledge_node(state: CortexState) -> dict:
        try:
            return await knowledge_fn(state)
        except Exception as exc:
            logger.error("knowledge agent failed", error=str(exc))
            return {"error": str(exc), "answer": "Knowledge base is temporarily unavailable."}

    async def research_node(state: CortexState) -> dict:
        try:
            return await research_fn(state)
        except Exception as exc:
            logger.error("research agent failed", error=str(exc))
            return {"error": str(exc), "answer": "Web research is temporarily unavailable."}

    async def action_node(state: CortexState) -> dict:
        try:
            return await action_fn(state)
        except Exception as exc:
            logger.error("action agent failed", error=str(exc))
            return {"error": str(exc), "answer": "Action tools are temporarily unavailable."}

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("knowledge", knowledge_node)
    workflow.add_node("research", research_node)
    workflow.add_node("action", action_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s.get("agent", "knowledge"),
        {"knowledge": "knowledge", "research": "research", "action": "action"},
    )

    workflow.add_edge("knowledge", END)
    workflow.add_edge("research", END)
    workflow.add_edge("action", END)

    return workflow.compile()
