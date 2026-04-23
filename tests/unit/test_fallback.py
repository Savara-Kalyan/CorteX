"""Unit tests for reliability/fallback.py"""
import pytest
from reliability.fallback import FallbackChain, RetryExecutor, CircuitBreaker, CircuitBreakerOpen


class TestFallbackChain:
    def test_first_tool_succeeds(self):
        chain = FallbackChain("test", [
            lambda **kw: {"success": True, "value": "primary"},
            lambda **kw: {"success": True, "value": "secondary"},
        ])
        result = chain.execute()
        assert result["success"] is True
        assert result["value"] == "primary"
        assert result["fallback_used"] is False
        assert result["fallback_index"] == 0

    def test_falls_back_to_second_on_failure(self):
        chain = FallbackChain("test", [
            lambda **kw: {"success": False, "error_type": "timeout"},
            lambda **kw: {"success": True, "value": "secondary"},
        ])
        result = chain.execute()
        assert result["success"] is True
        assert result["value"] == "secondary"
        assert result["fallback_used"] is True
        assert result["fallback_index"] == 1

    def test_all_tools_fail_returns_error(self):
        chain = FallbackChain("test", [
            lambda **kw: {"success": False, "error_type": "timeout"},
            lambda **kw: {"success": False, "error_type": "http_error"},
        ])
        result = chain.execute()
        assert result["success"] is False
        assert result["error_type"] == "all_tools_failed"
        assert result["chain"] == "test"

    def test_exception_in_tool_continues_chain(self):
        chain = FallbackChain("test", [
            lambda **kw: (_ for _ in ()).throw(RuntimeError("crash")),
            lambda **kw: {"success": True, "value": "recovered"},
        ])
        result = chain.execute()
        assert result["success"] is True

    def test_kwargs_passed_to_tools(self):
        received = {}

        def tool(**kw):
            received.update(kw)
            return {"success": True}

        chain = FallbackChain("test", [tool])
        chain.execute(query="hello", top_k=5)
        assert received["query"] == "hello"
        assert received["top_k"] == 5


class TestRetryExecutor:
    def test_success_on_first_attempt(self):
        def tool(**kw):
            return {"success": True, "data": "ok"}

        executor = RetryExecutor(tool, max_retries=2)
        result = executor.execute()
        assert result["success"] is True
        assert result["attempts"] == 1

    def test_retries_on_timeout(self):
        calls = []

        def tool(**kw):
            calls.append(1)
            if len(calls) < 3:
                return {"success": False, "error_type": "timeout"}
            return {"success": True}

        executor = RetryExecutor(tool, max_retries=3)
        result = executor.execute()
        assert result["success"] is True
        assert len(calls) == 3

    def test_no_retry_on_non_timeout_error(self):
        calls = []

        def tool(**kw):
            calls.append(1)
            return {"success": False, "error_type": "validation_error"}

        executor = RetryExecutor(tool, max_retries=3)
        result = executor.execute()
        assert result["success"] is False
        assert len(calls) == 1

    def test_exhausted_retries(self):
        def tool(**kw):
            return {"success": False, "error_type": "timeout"}

        executor = RetryExecutor(tool, max_retries=2)
        result = executor.execute()
        assert result["error_type"] == "retry_exhausted"
        assert result["attempts"] == 3


class TestCircuitBreaker:
    def test_closed_state_allows_calls(self):
        cb = CircuitBreaker(max_failures=3, timeout=60)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.failures == 0

    def test_opens_after_max_failures(self):
        cb = CircuitBreaker(max_failures=3, timeout=60)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
            except RuntimeError:
                pass
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "never called")

    def test_reset_closes_circuit(self):
        cb = CircuitBreaker(max_failures=2, timeout=60)
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
            except RuntimeError:
                pass
        cb.reset()
        result = cb.call(lambda: "works")
        assert result == "works"

    def test_get_state_returns_dict(self):
        cb = CircuitBreaker(max_failures=3, timeout=60)
        state = cb.get_state()
        assert "state" in state
        assert "failures" in state
        assert state["failures"] == 0
