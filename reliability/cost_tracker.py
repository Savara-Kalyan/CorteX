"""
Cost Tracker - Tracks LLM and embedding usage and enforces budget limits.
Prevents runaway costs in production.

Storage: Valkey (Redis-compatible), running as the 'valkey' service in docker-compose.
"""

import redis
from datetime import datetime
from typing import Dict


class CostTracker:
    """
    Tracks LLM and embedding costs and enforces daily budgets.
    Backed by Valkey for persistence across restarts.
    """

    # Pricing per token for models used in this project
    PRICES = {
        # LLM models
        "gpt-5.4-nano-2026-03-17": {
            "input": 0.15 / 1_000_000,   # $0.15 per 1M input tokens
            "output": 0.60 / 1_000_000,  # $0.60 per 1M output tokens
        },
        # Embedding models (no output tokens)
        "text-embedding-3-small": {
            "input": 0.02 / 1_000_000,   # $0.02 per 1M tokens
            "output": 0.0,
        },
    }

    def __init__(self, host: str = "valkey", port: int = 6379, db: int = 0):
        """
        Initialize cost tracker backed by Valkey.

        Args:
            host: Valkey hostname (matches docker-compose service name)
            port: Valkey port
            db: Valkey database index
        """
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def track_llm_call(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, float]:
        """
        Track cost of an LLM call and update the user's daily total.

        Args:
            user_id: User identifier
            model: Model name (e.g., "gpt-5.4-nano-2026-03-17")
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated

        Returns:
            Dictionary with call_cost, daily_total, input_tokens, output_tokens
        """
        if model not in self.PRICES:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(self.PRICES.keys())}"
            )

        call_cost = (
            input_tokens * self.PRICES[model]["input"]
            + output_tokens * self.PRICES[model]["output"]
        )
        daily_total = self._increment(user_id, call_cost)

        return {
            "call_cost": call_cost,
            "daily_total": daily_total,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def track_embedding_call(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
    ) -> Dict[str, float]:
        """
        Track cost of an embedding call and update the user's daily total.

        Args:
            user_id: User identifier
            model: Embedding model name (e.g., "text-embedding-3-small")
            input_tokens: Number of tokens sent to the embedding model

        Returns:
            Dictionary with call_cost, daily_total, and input_tokens
        """
        if model not in self.PRICES:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(self.PRICES.keys())}"
            )

        call_cost = input_tokens * self.PRICES[model]["input"]
        daily_total = self._increment(user_id, call_cost)

        return {
            "call_cost": call_cost,
            "daily_total": daily_total,
            "input_tokens": input_tokens,
        }

    def check_budget(self, user_id: str, daily_limit: float = 1.0) -> bool:
        """
        Check if the user has budget remaining for today.

        Args:
            user_id: User identifier
            daily_limit: Maximum daily spend in dollars (default: $1.00)

        Returns:
            True if within budget, False if limit exceeded
        """
        return self.get_daily_total(user_id) < daily_limit

    def get_daily_total(self, user_id: str) -> float:
        """
        Get the user's total spend for today.

        Args:
            user_id: User identifier

        Returns:
            Total cost in dollars for today
        """
        total_cents = self._client.get(self._key(user_id)) or 0
        return int(total_cents) / 100 # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, user_id: str) -> str:
        today = datetime.utcnow().date().isoformat()
        return f"cost:{user_id}:{today}"

    def _increment(self, user_id: str, amount: float) -> float:
        """Increment the user's daily total in Valkey and return the new value."""
        key = self._key(user_id)
        cost_cents = int(amount * 100)
        new_total_cents = self._client.incrby(key, cost_cents)
        self._client.expire(key, 60 * 60 * 24 * 30)  # 30-day TTL
        return new_total_cents / 100 # type: ignore
