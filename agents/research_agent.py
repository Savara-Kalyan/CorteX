from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools.web_search import web_search
from reliability.rate_limiter import RateLimiter, retry_with_backoff
from observability.logger import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "agents" / "research_agent"
_rate_limiter = RateLimiter()


class ResearchAgent:
    def __init__(self, prompt_version: str = "v1.0.0"):
        self._llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        self._prompt = self._load_prompt(prompt_version)

    def _load_prompt(self, version: str) -> dict:
        path = _PROMPTS_DIR / f"{version}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _compile_system_prompt(self) -> str:
        if not self._prompt:
            return "You are a research assistant. Answer using only the search results provided."
        role = self._prompt.get("role", {})
        constraints = self._prompt.get("constraints", {})
        security = self._prompt.get("security", {})

        constraints_text = "\n".join(
            f"- {k}: {v}" for k, v in constraints.items()
            if k != "prohibited_actions"
        )
        prohibited = "\n".join(
            f"- {a}" for a in constraints.get("prohibited_actions", [])
        )
        return (
            f"{security.get('top_guard', '').strip()}\n\n"
            f"ROLE: {role.get('identity', '')}\n"
            f"EXPERTISE: {role.get('expertise', '')}\n\n"
            f"CONSTRAINTS:\n{constraints_text}\n"
            f"PROHIBITED:\n{prohibited}\n\n"
            f"{security.get('bottom_guard', '').strip()}"
        )

    @retry_with_backoff(max_retries=2, initial_delay=1.0, backoff_factor=2.0)
    async def handle(self, state: dict) -> dict:
        query = state.get("query", "")
        user_id = state.get("user_id", "anonymous")

        allowed, retry_after = await _rate_limiter.check_rate_limit(
            f"research:{user_id}", max_requests=10, window_seconds=60
        )
        if not allowed:
            return {
                "answer": f"Web search rate limit reached. Retry in {retry_after}s.",
                "sources": [],
                "log_trace": state.get("log_trace", []) + [{"node": "research", "status": "rate_limited"}],
            }

        logger.info("research agent searching", query=query[:80], user_id=user_id)

        search_result = web_search.invoke({"query": query, "max_results": 5})

        if not search_result.get("success"):
            return {
                "answer": f"Web search unavailable: {search_result.get('message', 'unknown error')}",
                "sources": [],
                "log_trace": state.get("log_trace", []) + [
                    {"node": "research", "status": "search_failed", "error": search_result.get("message")}
                ],
            }

        results = search_result.get("results", [])
        context = "\n\n".join(
            f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in results
        )
        sources = [r["url"] for r in results if r.get("url")]

        system = self._compile_system_prompt()
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Search results:\n{context}\n\nQuestion: {query}"),
        ]
        response = self._llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": sources,
            "log_trace": state.get("log_trace", []) + [{
                "node": "research",
                "results_found": len(results),
                "source": search_result.get("source", "unknown"),
            }],
        }
