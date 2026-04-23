from __future__ import annotations

import asyncio
import functools
import time
from typing import Callable, Optional, Tuple

import redis.asyncio as aioredis


class RateLimiter:
    PREFIX = "rate_limit:"

    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self._redis = aioredis.Redis(
            host=host, port=port, db=db, decode_responses=True,
            socket_connect_timeout=2,
        )
        self._store: dict[str, list[int]] = {}

    async def check_rate_limit(
        self,
        user_id: str,
        max_requests: int = 10,
        window_seconds: int = 60,
    ) -> Tuple[bool, Optional[int]]:
        key = f"{self.PREFIX}{user_id}"
        now = int(time.time())
        window_start = now - window_seconds

        try:
            await self._redis.zremrangebyscore(key, 0, window_start)
            count = await self._redis.zcard(key)
            if count >= max_requests:
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                retry_after = int(oldest[0][1]) + window_seconds - now if oldest else window_seconds
                return False, max(1, retry_after)
            await self._redis.zadd(key, {str(now): now})
            await self._redis.expire(key, window_seconds)
            return True, None
        except Exception:
            timestamps = self._store.setdefault(key, [])
            self._store[key] = [t for t in timestamps if t > window_start]
            if len(self._store[key]) >= max_requests:
                oldest = min(self._store[key])
                retry_after = oldest + window_seconds - now
                return False, max(1, retry_after)
            self._store[key].append(now)
            return True, None


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = initial_delay
                last_exc: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                            delay *= backoff_factor
                raise last_exc  # type: ignore[misc]
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                last_exc: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt < max_retries:
                            time.sleep(delay)
                            delay *= backoff_factor
                raise last_exc  # type: ignore[misc]
            return wrapper
    return decorator
