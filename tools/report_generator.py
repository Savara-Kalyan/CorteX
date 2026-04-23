"""
Report generator tool — token-budget-aware output formatting.

W4 pattern: if estimated token cost > budget, return summary instead of full report.

Returns:
    dict with success (bool), report (str), report_type ('full'|'summary'),
    token_estimate (int).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Approximate tokens per character for English text
_TOKENS_PER_CHAR = 0.25
_FULL_REPORT_TOKEN_BUDGET = 1_500


def _estimate_tokens(text: str) -> int:
    return int(len(text) * _TOKENS_PER_CHAR)


def _format_full_report(title: str, sections: dict[str, str], metadata: dict) -> str:
    lines = [f"# {title}", ""]
    if metadata:
        lines += ["## Metadata"]
        for k, v in metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    for section_title, content in sections.items():
        lines += [f"## {section_title}", content, ""]
    return "\n".join(lines).strip()


def _format_summary(title: str, sections: dict[str, str]) -> str:
    summary_parts = [f"**{title}** — Summary"]
    for section_title, content in sections.items():
        first_sentence = content.split(".")[0].strip()
        summary_parts.append(f"• {section_title}: {first_sentence}.")
    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


class ReportInput(BaseModel):
    title: str = Field(description="Report title (e.g., 'Q3 IT Incident Summary')")
    sections: dict[str, str] = Field(
        description="Ordered dict of section_title → content strings"
    )
    format: Literal["markdown", "plain"] = Field(
        default="markdown", description="Output format"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata (author, date, classification)",
    )
    token_budget: int = Field(
        default=_FULL_REPORT_TOKEN_BUDGET,
        ge=100,
        description="Max tokens for the report body; triggers summary if exceeded",
    )


@tool(args_schema=ReportInput)
def generate_report(
    title: str,
    sections: dict[str, str],
    format: Literal["markdown", "plain"] = "markdown",
    metadata: dict[str, str] | None = None,
    token_budget: int = _FULL_REPORT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """
    Format collected information into a structured report.

    Use when:
    - User asks for a summary report or analysis document
    - Multiple data points need to be organized into sections
    - User says 'generate a report' or 'put this in a document'

    Do NOT use for: searching data, answering questions, creating tickets.

    Returns:
        dict with report (str), report_type ('full'|'summary'), token_estimate (int).
        Returns summary if estimated tokens exceed token_budget.
    """
    if not title or not sections:
        return {"success": False, "error_type": "validation_error", "message": "title and sections are required"}

    metadata = metadata or {}
    full_report = _format_full_report(title, sections, metadata)
    estimated = _estimate_tokens(full_report)

    if estimated <= token_budget:
        logger.debug("report '%s' within budget (%d tokens)", title, estimated)
        return {
            "success": True,
            "report": full_report,
            "report_type": "full",
            "token_estimate": estimated,
            "format": format,
        }

    # Budget exceeded — return summary only
    summary = _format_summary(title, sections)
    summary_tokens = _estimate_tokens(summary)
    logger.info(
        "report '%s' exceeded budget (%d > %d tokens) — returning summary",
        title, estimated, token_budget,
    )
    return {
        "success": True,
        "report": summary,
        "report_type": "summary",
        "token_estimate": summary_tokens,
        "full_token_estimate": estimated,
        "budget_exceeded": True,
        "format": format,
        "note": f"Full report (~{estimated} tokens) exceeded budget ({token_budget}). Returning summary.",
    }
