"""
Team calendar tool — mock data with realistic schedules and auth pattern.

W4 pattern: auth token check + graceful failure on missing credentials.

Returns:
    dict with success (bool), availability (list[dict]), or error on failure.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock schedule data
# ---------------------------------------------------------------------------

_TEAM_SCHEDULES: dict[str, list[dict]] = {
    "engineering": [
        {"time": "09:00–10:00", "status": "busy", "event": "Sprint standup"},
        {"time": "10:00–12:00", "status": "free"},
        {"time": "14:00–15:00", "status": "busy", "event": "Architecture review"},
        {"time": "15:00–17:00", "status": "free"},
    ],
    "hr": [
        {"time": "09:00–10:30", "status": "busy", "event": "Interviews"},
        {"time": "11:00–12:00", "status": "free"},
        {"time": "14:00–16:00", "status": "busy", "event": "Performance reviews"},
    ],
    "finance": [
        {"time": "09:00–09:30", "status": "free"},
        {"time": "09:30–11:00", "status": "busy", "event": "Budget reconciliation"},
        {"time": "11:00–17:00", "status": "free"},
    ],
    "default": [
        {"time": "09:00–12:00", "status": "free"},
        {"time": "13:00–14:00", "status": "busy", "event": "Team sync"},
        {"time": "14:00–17:00", "status": "free"},
    ],
}


def _get_auth_token() -> str | None:
    """Retrieve calendar auth token from environment."""
    return os.getenv("CALENDAR_AUTH_TOKEN") or os.getenv("CALENDAR_API_KEY")


def _next_working_day() -> str:
    """Return the next working day date string."""
    d = datetime.utcnow() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


class CalendarInput(BaseModel):
    team_or_person: str = Field(
        description="Team name (engineering, hr, finance) or person's name"
    )
    date: str = Field(
        default="",
        description="Date in YYYY-MM-DD format (defaults to next working day)",
    )


@tool(args_schema=CalendarInput)
def get_team_calendar(team_or_person: str, date: str = "") -> dict[str, Any]:
    """
    Get availability for a team or person for a given date.

    Use when:
    - User asks about scheduling a meeting or checking availability
    - User wants to know when a team is free
    - User asks 'when can I meet with the HR team?'

    Do NOT use for: creating meetings, sending invites, or fetching personal data.

    Returns:
        dict with team (str), date (str), availability (list of time slots).
        On auth failure: error_type='auth_error' with retry guidance.
    """
    token = _get_auth_token()
    if not token:
        logger.warning("calendar auth token missing — returning graceful failure")
        return {
            "success": False,
            "error_type": "auth_error",
            "message": "Calendar unavailable — authentication not configured. Try again in 5 minutes or contact IT.",
            "retry_after_seconds": 300,
        }

    target_date = date.strip() if date.strip() else _next_working_day()
    schedule_key = team_or_person.lower().strip()
    slots = _TEAM_SCHEDULES.get(schedule_key, _TEAM_SCHEDULES["default"])

    free_slots = [s for s in slots if s["status"] == "free"]
    busy_slots = [s for s in slots if s["status"] == "busy"]

    return {
        "success": True,
        "team": team_or_person,
        "date": target_date,
        "availability": slots,
        "free_slots": free_slots,
        "busy_slots": busy_slots,
        "next_free": free_slots[0]["time"] if free_slots else "No free slots",
    }
