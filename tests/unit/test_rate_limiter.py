import asyncio
import time
import pytest
from tests.conftest import requires_redis
from reliability.rate_limiter import RateLimiter, retry_with_backoff


@requires_redis
class TestRateLimiter:
    @pytest.fixture(autouse=True)
    async def limiter(self, redis_client):
        inst = RateLimiter(host="localhost", port=6379)
        keys = redis_client.keys(f"{RateLimiter.PREFIX}test-rl-*")
        if keys:
            redis_client.delete(*keys)
        self._limiter = inst
        self._redis = redis_client
        yield inst
        keys = redis_client.keys(f"{RateLimiter.PREFIX}test-rl-*")
        if keys:
            redis_client.delete(*keys)

    async def test_first_request_allowed(self):
        allowed, retry_after = await self._limiter.check_rate_limit(
            "test-rl-user-1", max_requests=5, window_seconds=60
        )
        assert allowed is True
        assert retry_after is None

    async def test_requests_within_limit_all_allowed(self):
        for _ in range(5):
            allowed, _ = await self._limiter.check_rate_limit(
                "test-rl-user-2", max_requests=5, window_seconds=60
            )
            assert allowed is True

    async def test_request_exceeding_limit_blocked(self):
        for _ in range(5):
            await self._limiter.check_rate_limit(
                "test-rl-user-3", max_requests=5, window_seconds=60
            )
        allowed, retry_after = await self._limiter.check_rate_limit(
            "test-rl-user-3", max_requests=5, window_seconds=60
        )
        assert allowed is False
        assert retry_after is not None
        assert retry_after >= 1

    async def test_different_users_are_independent(self):
        for _ in range(5):
            await self._limiter.check_rate_limit(
                "test-rl-user-a", max_requests=5, window_seconds=60
            )
        allowed, _ = await self._limiter.check_rate_limit(
            "test-rl-user-b", max_requests=5, window_seconds=60
        )
        assert allowed is True

    async def test_window_expires(self):
        limiter = RateLimiter(host="localhost", port=6379)
        for _ in range(3):
            await limiter.check_rate_limit("test-rl-user-exp", max_requests=3, window_seconds=1)
        allowed, _ = await limiter.check_rate_limit(
            "test-rl-user-exp", max_requests=3, window_seconds=1
        )
        assert allowed is False
        await asyncio.sleep(1.1)
        allowed, _ = await limiter.check_rate_limit(
            "test-rl-user-exp", max_requests=3, window_seconds=1
        )
        assert allowed is True

    async def test_retry_after_is_positive_when_blocked(self):
        for _ in range(3):
            await self._limiter.check_rate_limit(
                "test-rl-user-ra", max_requests=3, window_seconds=30
            )
        allowed, retry_after = await self._limiter.check_rate_limit(
            "test-rl-user-ra", max_requests=3, window_seconds=30
        )
        assert not allowed
        assert 1 <= retry_after <= 30


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        calls = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=2.0)
        def fn():
            calls.append(1)
            return "ok"

        result = fn()
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_exception(self):
        calls = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=1.0)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "ok"

        result = fn()
        assert result == "ok"
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self):
        @retry_with_backoff(max_retries=2, initial_delay=0.01, backoff_factor=1.0)
        def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            fn()

    def test_only_retries_specified_exceptions(self):
        calls = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01, exceptions=(ValueError,))
        def fn():
            calls.append(1)
            raise TypeError("different exception")

        with pytest.raises(TypeError):
            fn()
        assert len(calls) == 1

    async def test_async_succeeds_on_first_try(self):
        calls = []

        @retry_with_backoff(max_retries=2, initial_delay=0.01, backoff_factor=1.0)
        async def fn():
            calls.append(1)
            return "ok"

        result = await fn()
        assert result == "ok"
        assert len(calls) == 1

    async def test_async_retries_on_exception(self):
        calls = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=1.0)
        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("transient")
            return "ok"

        result = await fn()
        assert result == "ok"
        assert len(calls) == 2
