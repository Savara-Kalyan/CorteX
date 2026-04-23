import redis.asyncio as aioredis
from datetime import datetime
from typing import Dict


class CostTracker:
    PRICES = {
        "gpt-5.4-nano-2026-03-17": {
            "input": 0.15 / 1_000_000,
            "output": 0.60 / 1_000_000,
        },
        "text-embedding-3-small": {
            "input": 0.02 / 1_000_000,
            "output": 0.0,
        },
    }

    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self._client = aioredis.Redis(host=host, port=port, db=db, decode_responses=True)

    async def track_llm_call(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, float]:
        if model not in self.PRICES:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.PRICES.keys())}")
        call_cost = (
            input_tokens * self.PRICES[model]["input"]
            + output_tokens * self.PRICES[model]["output"]
        )
        daily_total = await self._increment(user_id, call_cost)
        return {
            "call_cost": call_cost,
            "daily_total": daily_total,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def track_embedding_call(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
    ) -> Dict[str, float]:
        if model not in self.PRICES:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.PRICES.keys())}")
        call_cost = input_tokens * self.PRICES[model]["input"]
        daily_total = await self._increment(user_id, call_cost)
        return {
            "call_cost": call_cost,
            "daily_total": daily_total,
            "input_tokens": input_tokens,
        }

    async def check_budget(self, user_id: str, daily_limit: float = 1.0) -> bool:
        return await self.get_daily_total(user_id) < daily_limit

    async def get_daily_total(self, user_id: str) -> float:
        total_cents = await self._client.get(self._key(user_id)) or 0
        return int(total_cents) / 100  # type: ignore

    def _key(self, user_id: str) -> str:
        today = datetime.utcnow().date().isoformat()
        return f"cost:{user_id}:{today}"

    async def _increment(self, user_id: str, amount: float) -> float:
        key = self._key(user_id)
        cost_cents = int(amount * 100)
        new_total_cents = await self._client.incrby(key, cost_cents)
        await self._client.expire(key, 60 * 60 * 24 * 30)
        return new_total_cents / 100  # type: ignore
