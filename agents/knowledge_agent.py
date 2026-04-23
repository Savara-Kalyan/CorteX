from __future__ import annotations

import logging
from pathlib import Path

import yaml

from rag.pipeline import RAGPipeline
from observability.logger import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "agents" / "knowledge_agent"


class KnowledgeAgent:
    def __init__(self, rag_pipeline: RAGPipeline | None = None, prompt_version: str = "v1.0.0"):
        self._pipeline = rag_pipeline or RAGPipeline()
        self._prompt = self._load_prompt(prompt_version)

    def _load_prompt(self, version: str) -> dict:
        path = _PROMPTS_DIR / f"{version}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _compile_system_prompt(self) -> str | None:
        if not self._prompt:
            return None
        role = self._prompt.get("role", {})
        constraints = self._prompt.get("constraints", {})
        fmt = self._prompt.get("context", {}).get("answer_format", [])
        security = self._prompt.get("security", {})

        constraints_text = "\n".join(
            f"- {k}: {v}" for k, v in constraints.items()
            if k != "prohibited_actions"
        )
        prohibited = "\n".join(
            f"- {a}" for a in constraints.get("prohibited_actions", [])
        )
        format_text = "\n".join(f"- {f}" for f in fmt)

        return (
            f"{security.get('top_guard', '').strip()}\n\n"
            f"ROLE: {role.get('identity', '')}\n\n"
            f"CONSTRAINTS:\n{constraints_text}\n"
            f"PROHIBITED:\n{prohibited}\n\n"
            f"ANSWER FORMAT:\n{format_text}\n\n"
            f"{security.get('bottom_guard', '').strip()}"
        )

    async def handle(self, state: dict) -> dict:
        query = state.get("query", "")
        user_id = state.get("user_id", "anonymous")
        user_tier = state.get("user_tier", "public")

        logger.info("knowledge agent handling", query=query[:80], user_id=user_id, tier=user_tier)

        result = await self._pipeline.query(
            query=query,
            user_id=user_id,
            user_tier=user_tier,
            top_k=5,
            system_prompt=self._compile_system_prompt(),
            domain=state.get("domain"),
        )

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "log_trace": state.get("log_trace", []) + [{
                "node": "knowledge",
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "intent": result.get("intent", "unknown"),
                "reformulated_query": result.get("reformulated_query", query),
            }],
        }
