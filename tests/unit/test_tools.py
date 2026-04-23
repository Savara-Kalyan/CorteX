"""Unit tests for tools/ — schema validation + error handling."""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_empty_query_returns_validation_error(self):
        from tools.web_search import web_search
        result = web_search.invoke({"query": "   ", "max_results": 5})
        assert result["success"] is False
        assert result["error_type"] == "validation_error"

    def test_successful_search_returns_results(self):
        from tools.web_search import _search_duckduckgo
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "RelatedTopics": [
                {"Text": "Test result", "FirstURL": "https://example.com/1"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("tools.web_search._HTTP") as mock_http:
            mock_http.get.return_value = mock_response
            result = _search_duckduckgo("test query", 5)
        assert result["success"] is True
        assert len(result["results"]) >= 1
        assert result["source"] == "duckduckgo"

    def test_ddg_timeout_returns_error(self):
        import httpx
        from tools.web_search import _search_duckduckgo
        with patch("tools.web_search._HTTP") as mock_http:
            mock_http.get.side_effect = httpx.TimeoutException("timeout")
            result = _search_duckduckgo("query", 5)
        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_serpapi_missing_key_returns_config_error(self):
        from tools.web_search import _search_serpapi
        with patch.dict("os.environ", {}, clear=True):
            # Remove SERPAPI_API_KEY if set
            import os
            os.environ.pop("SERPAPI_API_KEY", None)
            result = _search_serpapi("query", 5)
        assert result["success"] is False
        assert result["error_type"] == "config_error"

    def test_cache_returns_last_result_on_miss(self):
        from tools.web_search import _return_cached, _cache
        result = _return_cached("uncached-query", 5)
        assert result["success"] is False
        assert result["error_type"] == "cache_miss"


# ---------------------------------------------------------------------------
# create_support_ticket
# ---------------------------------------------------------------------------

class TestCreateSupportTicket:
    def test_short_title_returns_validation_error(self):
        from tools.ticketing import create_support_ticket
        result = create_support_ticket.invoke({
            "title": "Ab",
            "category": "technical",
        })
        assert result["success"] is False
        assert result["error_type"] == "validation_error"

    def test_valid_ticket_returns_ticket_id(self):
        from tools.ticketing import _create_in_queue
        result = _create_in_queue("Broken SSO login", "technical", "high", "")
        assert result["success"] is True
        assert result["ticket_id"].startswith("TKT-")
        assert result["status"] == "queued"

    def test_priority_maps_to_response_hours(self):
        from tools.ticketing import _create_in_queue
        critical = _create_in_queue("critical issue", "technical", "critical", "")
        low = _create_in_queue("low issue", "technical", "low", "")
        assert critical["estimated_response_hours"] < low["estimated_response_hours"]


# ---------------------------------------------------------------------------
# get_team_calendar
# ---------------------------------------------------------------------------

class TestGetTeamCalendar:
    def test_missing_auth_returns_auth_error(self):
        from tools.calendar import get_team_calendar
        import os
        os.environ.pop("CALENDAR_AUTH_TOKEN", None)
        os.environ.pop("CALENDAR_API_KEY", None)
        result = get_team_calendar.invoke({"team_or_person": "engineering"})
        assert result["success"] is False
        assert result["error_type"] == "auth_error"

    def test_with_auth_token_returns_slots(self):
        from tools.calendar import get_team_calendar
        with patch.dict("os.environ", {"CALENDAR_AUTH_TOKEN": "test-token"}):
            result = get_team_calendar.invoke({"team_or_person": "engineering"})
        assert result["success"] is True
        assert "availability" in result
        assert len(result["availability"]) > 0

    def test_unknown_team_uses_default_schedule(self):
        from tools.calendar import get_team_calendar
        with patch.dict("os.environ", {"CALENDAR_AUTH_TOKEN": "test-token"}):
            result = get_team_calendar.invoke({"team_or_person": "unknown-team-xyz"})
        assert result["success"] is True


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_missing_title_returns_error(self):
        from tools.report_generator import generate_report
        result = generate_report.invoke({"title": "", "sections": {"s": "content"}})
        assert result["success"] is False

    def test_within_budget_returns_full_report(self):
        from tools.report_generator import generate_report
        result = generate_report.invoke({
            "title": "Test Report",
            "sections": {"Summary": "Short content."},
            "token_budget": 5000,
        })
        assert result["success"] is True
        assert result["report_type"] == "full"
        assert "Test Report" in result["report"]

    def test_over_budget_returns_summary(self):
        from tools.report_generator import generate_report
        large_content = "x " * 5000  # ~10k tokens estimated
        result = generate_report.invoke({
            "title": "Large Report",
            "sections": {"Section": large_content},
            "token_budget": 100,
        })
        assert result["success"] is True
        assert result["report_type"] == "summary"
        assert result.get("budget_exceeded") is True

    def test_report_contains_all_sections(self):
        from tools.report_generator import generate_report
        result = generate_report.invoke({
            "title": "Multi-Section",
            "sections": {"Intro": "First.", "Details": "Second.", "Conclusion": "Third."},
            "token_budget": 5000,
        })
        assert "Intro" in result["report"]
        assert "Details" in result["report"]
        assert "Conclusion" in result["report"]
