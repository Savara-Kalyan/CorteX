from __future__ import annotations

import time
import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class FallbackChain:
    def __init__(self, name: str, tools: list[Callable[..., dict]]):
        self.name = name
        self.tools = tools

    def execute(self, **kwargs: Any) -> dict:
        last_error: dict | None = None
        for idx, tool in enumerate(self.tools):
            try:
                result = tool(**kwargs)
            except Exception as exc:
                last_error = {"success": False, "error_type": "exception", "message": str(exc)}
                logger.warning("%s: tool[%d] raised %s", self.name, idx, exc)
                continue
            if result.get("success") is True:
                result["fallback_used"] = idx > 0
                result["fallback_index"] = idx
                return result
            last_error = result
            logger.warning("%s: tool[%d] returned failure: %s", self.name, idx, result)
        return {
            "success": False,
            "error_type": "all_tools_failed",
            "chain": self.name,
            "last_error": last_error,
        }


class RetryExecutor:
    def __init__(self, tool_fn: Callable[..., dict], max_retries: int = 2):
        self.tool_fn = tool_fn
        self.max_retries = max_retries

    def execute(self, **kwargs: Any) -> dict:
        attempts = 0
        last_result: dict | None = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = self.tool_fn(**kwargs)
            except Exception as exc:
                result = {"success": False, "error_type": "exception", "message": str(exc)}
            last_result = result
            if result.get("success") is True:
                result["attempts"] = attempts
                return result
            if result.get("error_type") != "timeout":
                result["attempts"] = attempts
                return result
        return {
            "success": False,
            "error_type": "retry_exhausted",
            "attempts": attempts,
            "last_result": last_result,
        }


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit is open and the timeout has not elapsed."""


class CircuitBreaker:
    def __init__(self, max_failures: int = 3, timeout: int = 60):
        self.max_failures = max_failures
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable) -> Any:
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time if self.last_failure_time else 0
            if elapsed >= self.timeout:
                logger.info("CircuitBreaker → HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
            else:
                wait = self.timeout - elapsed
                raise CircuitBreakerOpen(
                    f"Circuit open after {self.failures} failures. Retry in {wait:.0f}s"
                )

        try:
            result = func()
            if self.state == CircuitState.HALF_OPEN:
                logger.info("CircuitBreaker probe succeeded → CLOSED")
                self.state = CircuitState.CLOSED
            self.failures = 0
            return result
        except Exception as exc:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.max_failures or self.state == CircuitState.HALF_OPEN:
                logger.warning("CircuitBreaker → OPEN after %d failures", self.failures)
                self.state = CircuitState.OPEN
            raise exc

    def reset(self) -> None:
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def get_state(self) -> dict:
        return {
            "state": self.state.value,
            "failures": self.failures,
            "max_failures": self.max_failures,
            "timeout": self.timeout,
            "time_since_last_failure": (
                time.time() - self.last_failure_time if self.last_failure_time else None
            ),
        }
