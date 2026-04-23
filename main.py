from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# psycopg async is incompatible with Windows ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv

load_dotenv()

from observability.logger import get_logging_config
from observability.logger import configure, get_logger
from observability.metrics import MetricsDashboard

configure(get_logging_config())
logger = get_logger(__name__)

from agents.knowledge_agent import KnowledgeAgent
from agents.research_agent import ResearchAgent
from agents.action_agent import ActionAgent
from agents.supervisor import build_graph, CortexState
from memory.session_memory import RedisSessionManager
from memory.entity_store import LongTermMemoryStore
from rag.pipeline import RAGPipeline

dashboard = MetricsDashboard()

_rag_pipeline = RAGPipeline()
_knowledge_agent = KnowledgeAgent(rag_pipeline=_rag_pipeline)
_research_agent = ResearchAgent()
_action_agent = ActionAgent()
_session_memory = RedisSessionManager()
_entity_store = LongTermMemoryStore()

graph = build_graph(
    knowledge_fn=_knowledge_agent.handle,
    research_fn=_research_agent.handle,
    action_fn=_action_agent.handle,
)


async def run_query(
    query: str,
    user_id: str = "demo-user",
    user_tier: str = "internal",
    session_id: str | None = None,
) -> str:
    session_id = session_id or f"session-{user_id}"

    initial_state: CortexState = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "user_tier": user_tier,
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

    result = await graph.ainvoke(initial_state)

    if result.get("agent"):
        dashboard.record_agent_call(result["agent"])

    await _session_memory.append_turn(session_id, query, result.get("answer", ""))

    answer = result.get("answer", "No answer generated.")
    if result.get("error"):
        answer += f"\n\n[System note: {result['error']}]"
    return answer


async def ingest_demo_docs() -> None:
    kb_path = Path(__file__).parent / "org-docs"
    logger.info("Ingesting sample knowledge base", path=str(kb_path))
    result = await _rag_pipeline.ingest(kb_path)
    print(f"Ingested {result['documents_ingested']} documents → {result['chunks_created']} chunks")


DEMO_QUERIES = [
    ("What is the parental leave policy?", "internal"),
    ("How do I set up the VPN on my laptop?", "internal"),
    ("What are the expense claim limits for travel?", "confidential"),
    ("Create a support ticket: broken SSO login, high priority", "internal"),
    ("When is the engineering team free this week?", "internal"),
    ("What are the latest AI benchmarks for 2026?", "public"),
]


async def run_demo() -> None:
    print("\n" + "=" * 60)
    print("  CORTEX DEMO — running canned queries")
    print("=" * 60)
    for i, (query, tier) in enumerate(DEMO_QUERIES, 1):
        print(f"\n[{i}/{len(DEMO_QUERIES)}] [{tier.upper()}] {query}")
        print("-" * 60)
        try:
            answer = await run_query(query, user_id="demo-user", user_tier=tier)
            print(answer[:400])
        except Exception as exc:
            print(f"ERROR: {exc}")
    await dashboard.print_summary(user_id="demo-user")


async def run_repl() -> None:
    print("\nCortex AI Assistant — type 'exit' to quit, 'metrics' for dashboard")
    user_id = os.getenv("CORTEX_USER_ID", "repl-user")
    user_tier = os.getenv("CORTEX_USER_TIER", "internal")
    print(f"User: {user_id}  Tier: {user_tier}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "metrics":
            await dashboard.print_summary(user_id=user_id)
            continue
        answer = await run_query(query, user_id=user_id, user_tier=user_tier)
        print(f"Cortex: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cortex AI Assistant")
    parser.add_argument("--demo", action="store_true", help="Run canned demo queries")
    parser.add_argument("--ingest", action="store_true", help="Ingest sample knowledge base")
    args = parser.parse_args()

    if args.ingest:
        asyncio.run(ingest_demo_docs())
    elif args.demo:
        asyncio.run(run_demo())
    else:
        asyncio.run(run_repl())


if __name__ == "__main__":
    main()
