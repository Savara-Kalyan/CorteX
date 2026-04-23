"""
Web search tool — DuckDuckGo primary, SerpAPI fallback, cached last result.

W4 pattern: rate limiting + 3-tier fallback chain.

Returns:
    dict with keys: success (bool), results (list[dict]), source (str),
                    error_type / message on failure.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# HTTP client with conservative timeouts
_HTTP = httpx.Client(
    timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
    follow_redirects=True,
)

_DDG_URL = "https://api.duckduckgo.com/"
_SERPAPI_URL = "https://serpapi.com/search"

# Simple in-process cache: stores last successful result per query
_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Tier 1 — DuckDuckGo (free, no key)
# ---------------------------------------------------------------------------


def _search_duckduckgo(query: str, max_results: int) -> dict:
    try:
        r = _HTTP.get(
            _DDG_URL,
            params={"q": query, "format": "json", "no_html": "1", "no_redirect": "1"},
        )
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        return {"success": False, "error_type": "timeout", "message": "DuckDuckGo timed out"}
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            return {"success": False, "error_type": "rate_limit", "retry_after_seconds": 60, "message": "DuckDuckGo rate limited"}
        return {"success": False, "error_type": "http_error", "message": str(exc)}
    except Exception as exc:
        return {"success": False, "error_type": "request_error", "message": str(exc)}

    results = [
        {"title": t.get("Text", ""), "url": t.get("FirstURL", ""), "snippet": t.get("Text", "")}
        for t in data.get("RelatedTopics", [])
        if isinstance(t, dict) and t.get("FirstURL")
    ][:max_results]

    if not results and data.get("AbstractText"):
        results = [{"title": data.get("Heading", query), "url": data.get("AbstractURL", ""), "snippet": data.get("AbstractText", "")}]

    if not results:
        return {"success": False, "error_type": "no_results", "message": "DuckDuckGo returned no results"}

    return {"success": True, "results": results, "source": "duckduckgo"}


# ---------------------------------------------------------------------------
# Tier 2 — SerpAPI (requires SERPAPI_API_KEY)
# ---------------------------------------------------------------------------


def _search_serpapi(query: str, max_results: int) -> dict:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {"success": False, "error_type": "config_error", "message": "SERPAPI_API_KEY not set"}
    try:
        r = _HTTP.get(
            _SERPAPI_URL,
            params={"engine": "google", "q": query, "api_key": api_key, "num": min(max_results, 10)},
        )
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        return {"success": False, "error_type": "timeout", "message": "SerpAPI timed out"}
    except httpx.HTTPStatusError as exc:
        return {"success": False, "error_type": "http_error", "message": str(exc)}
    except Exception as exc:
        return {"success": False, "error_type": "request_error", "message": str(exc)}

    organic = data.get("organic_results") or []
    results = [
        {"title": o.get("title", ""), "url": o.get("link", ""), "snippet": o.get("snippet", "")}
        for o in organic[:max_results]
    ]
    if not results:
        return {"success": False, "error_type": "no_results", "message": "SerpAPI returned no results"}
    return {"success": True, "results": results, "source": "serpapi"}


# ---------------------------------------------------------------------------
# Tier 3 — cached last result
# ---------------------------------------------------------------------------


def _return_cached(query: str, max_results: int) -> dict:
    if query in _cache:
        cached = _cache[query].copy()
        cached["source"] = "cache"
        cached["cached"] = True
        return cached
    return {"success": False, "error_type": "cache_miss", "message": "No cached result available"}


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string (be specific for better results)")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of results to return (1-10)")


@tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search the web for recent information on a topic.

    Use when:
    - User asks about current events, news, or external information
    - Internal knowledge base has no relevant documents
    - User needs up-to-date data (prices, news, regulations)

    Do NOT use for: internal company documents, HR policies, engineering specs.

    Returns:
        dict with success (bool), results (list of {title, url, snippet}), source (str).
        On failure: error_type and message.
    """
    if not query or not query.strip():
        return {"success": False, "error_type": "validation_error", "message": "Query cannot be empty"}

    query = query.strip()

    for tier_fn in (_search_duckduckgo, _search_serpapi, _return_cached):
        result = tier_fn(query, max_results)
        if result.get("success"):
            _cache[query] = result
            return result

    return {"success": False, "error_type": "all_tiers_failed", "message": "All search tiers exhausted"}
