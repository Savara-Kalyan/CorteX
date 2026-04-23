from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools.ticketing import create_support_ticket
from tools.calendar import get_team_calendar
from tools.report_generator import generate_report
from reliability.cost_tracker import CostTracker
from observability.logger import get_logger

logger = get_logger(__name__)

_TOOLS = [create_support_ticket, get_team_calendar, generate_report]
_cost_tracker = CostTracker()

_SYSTEM_PROMPT = """You are the Cortex Action Agent. You have access to three tools:

1. create_support_ticket  — open a ticket for issues needing tracking or escalation
2. get_team_calendar      — check team or person availability
3. generate_report        — format collected data into a structured report

Instructions:
- Pick the SINGLE best tool for the user's request.
- Pass all required arguments; use defaults for optional ones.
- If the user's request does not match any tool, reply with a plain text explanation.
- Always report the result clearly (ticket ID, time slots, or report content).
"""


class ActionAgent:
    def __init__(self):
        self._llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2).bind_tools(_TOOLS)

    async def handle(self, state: dict) -> dict:
        query = state.get("query", "")
        user_id = state.get("user_id", "anonymous")

        if not await _cost_tracker.check_budget(user_id, daily_limit=2.0):
            return {
                "answer": "Daily budget exceeded — action tools are disabled until tomorrow.",
                "sources": [],
                "log_trace": state.get("log_trace", []) + [{"node": "action", "status": "budget_exceeded"}],
            }

        logger.info("action agent handling", query=query[:80], user_id=user_id)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = self._llm.invoke(messages)

        tool_results: list[dict] = []
        if response.tool_calls:
            tool_map = {t.name: t for t in _TOOLS}
            for tc in response.tool_calls:
                tool = tool_map.get(tc["name"])
                if tool:
                    result = tool.invoke(tc["args"])
                    tool_results.append({"tool": tc["name"], "result": result})
                    logger.info("action tool executed", tool=tc["name"], success=result.get("success"))

        answer = self._format_answer(query, response, tool_results)
        return {
            "answer": answer,
            "sources": [],
            "log_trace": state.get("log_trace", []) + [{
                "node": "action",
                "tools_called": [r["tool"] for r in tool_results],
                "tool_results": tool_results,
            }],
        }

    def _format_answer(self, query: str, llm_response: Any, tool_results: list[dict]) -> str:
        if not tool_results:
            return llm_response.content or "No action was taken."

        parts: list[str] = []
        for tr in tool_results:
            result = tr["result"]
            tool_name = tr["tool"]

            if tool_name == "create_support_ticket" and result.get("success"):
                parts.append(
                    f"Ticket created: **{result['ticket_id']}** (status: {result['status']}, "
                    f"estimated response: {result['estimated_response_hours']}h)"
                )
            elif tool_name == "get_team_calendar" and result.get("success"):
                free = ", ".join(s["time"] for s in result.get("free_slots", []))
                parts.append(
                    f"**{result['team']}** availability on {result['date']}: {free or 'No free slots'}"
                )
            elif tool_name == "generate_report" and result.get("success"):
                note = f" *(summary — full report ~{result.get('full_token_estimate')} tokens)*" if result.get("budget_exceeded") else ""
                parts.append(f"{result['report']}{note}")
            elif not result.get("success"):
                parts.append(f"{tool_name} failed: {result.get('message', 'unknown error')}")
            else:
                parts.append(str(result))

        return "\n\n".join(parts)
