"""
Support ticket tool with input validation and 3-tier fallback.

Tier 1: live JSONPlaceholder POST (free, no key — demonstrates real HTTP)
Tier 2: in-memory queue (when API is down)
Tier 3: return queue ID from in-memory store

W4 pattern: proper schema + input validation + fallback.
"""

from __future__ import annotations

import logging
import random
import string
from typing import Any, Literal

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_TICKET_API_URL = "https://jsonplaceholder.typicode.com/posts"
_HTTP = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0))

# In-memory fallback queue: list[dict]
_pending_queue: list[dict] = []

_HOURS_BY_PRIORITY = {"low": 48, "normal": 24, "high": 4, "critical": 1}


def _ticket_id() -> str:
    suffix = "".join(random.choices(string.digits, k=5))
    return f"TKT-{suffix}"


# ---------------------------------------------------------------------------
# Tier 1 — live API
# ---------------------------------------------------------------------------


def _create_via_api(title: str, category: str, priority: str, description: str) -> dict:
    payload = {
        "title": title,
        "body": description or f"category={category} priority={priority}",
        "userId": 1,
    }
    try:
        r = _HTTP.post(_TICKET_API_URL, json=payload)
        r.raise_for_status()
        resp = r.json()
        return {
            "success": True,
            "ticket_id": f"TKT-{resp.get('id', 0):05d}",
            "status": "created",
            "estimated_response_hours": _HOURS_BY_PRIORITY.get(priority, 24),
            "source": "api",
        }
    except httpx.TimeoutException:
        return {"success": False, "error_type": "timeout", "message": "Ticketing API timed out"}
    except httpx.HTTPStatusError as exc:
        return {"success": False, "error_type": "http_error", "message": str(exc)}
    except Exception as exc:
        return {"success": False, "error_type": "request_error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tier 2 — in-memory queue
# ---------------------------------------------------------------------------


def _create_in_queue(title: str, category: str, priority: str, description: str) -> dict:
    tid = _ticket_id()
    _pending_queue.append({"ticket_id": tid, "title": title, "category": category, "priority": priority})
    return {
        "success": True,
        "ticket_id": tid,
        "status": "queued",
        "estimated_response_hours": _HOURS_BY_PRIORITY.get(priority, 24) + 2,
        "note": "API unavailable — ticket queued for sync",
        "source": "queue",
    }


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


class TicketInput(BaseModel):
    title: str = Field(description="Short issue summary (3-100 characters)")
    category: Literal["billing", "technical", "hr", "it", "general"] = Field(
        description="Issue category — billing, technical, hr, it, or general"
    )
    priority: Literal["low", "normal", "high", "critical"] = Field(
        default="normal",
        description="Use 'critical' only for: data loss, security breach, payment failure",
    )
    description: str = Field(default="", description="Detailed description (optional)")
    user_id: str = Field(default="unknown", description="User ID of the requester")


@tool(args_schema=TicketInput)
def create_support_ticket(
    title: str,
    category: Literal["billing", "technical", "hr", "it", "general"],
    priority: Literal["low", "normal", "high", "critical"] = "normal",
    description: str = "",
    user_id: str = "unknown",
) -> dict[str, Any]:
    """
    Create a support ticket in the ticketing system.

    Use when:
    - User reports an issue that needs tracking or follow-up
    - An issue must be escalated to a human agent
    - User asks to 'open a ticket' or 'raise an issue'

    Do NOT use for: answering questions, providing information, calendar lookups.

    Returns:
        dict with ticket_id (str), status ('created'|'queued'),
        estimated_response_hours (int).  On validation error: error_type + message.
    """
    if not title or len(title.strip()) < 3:
        return {"success": False, "error_type": "validation_error", "message": "Title too short — minimum 3 characters"}

    title = title.strip()[:100]

    result = _create_via_api(title, category, priority, description)
    if result.get("success"):
        logger.info("ticket created via API: %s", result["ticket_id"])
        return result

    result = _create_in_queue(title, category, priority, description)
    logger.warning("ticket queued (API down): %s", result["ticket_id"])
    return result
