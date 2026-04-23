import pytest
from datetime import datetime
from tests.conftest import requires_redis
from reliability.cost_tracker import CostTracker


@requires_redis
class TestCostTracker:
    @pytest.fixture(autouse=True)
    async def tracker(self, redis_client):
        inst = CostTracker(host="localhost", port=6379)
        self._tracker = inst
        self._redis = redis_client
        yield inst
        today = datetime.utcnow().date().isoformat()
        keys = self._redis.keys(f"cost:test-cost-*:{today}")
        if keys:
            self._redis.delete(*keys)

    async def test_track_llm_call_known_model(self):
        result = await self._tracker.track_llm_call(
            user_id="test-cost-u1",
            model="gpt-5.4-nano-2026-03-17",
            input_tokens=1000,
            output_tokens=500,
        )
        assert "call_cost" in result
        assert result["call_cost"] > 0
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500

    async def test_track_llm_call_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            await self._tracker.track_llm_call("test-cost-u2", "gpt-unknown-xyz", 100, 100)

    async def test_track_embedding_call(self):
        result = await self._tracker.track_embedding_call(
            user_id="test-cost-u3",
            model="text-embedding-3-small",
            input_tokens=2000,
        )
        assert result["call_cost"] > 0
        assert result["input_tokens"] == 2000

    async def test_check_budget_within_limit(self):
        assert await self._tracker.check_budget("test-cost-u4", daily_limit=100.0) is True

    async def test_check_budget_exceeded_after_large_spend(self):
        await self._tracker.track_llm_call(
            "test-cost-u5", "gpt-5.4-nano-2026-03-17", 1_000_000, 0
        )
        assert await self._tracker.check_budget("test-cost-u5", daily_limit=0.10) is False

    async def test_daily_total_accumulates_correctly(self):
        await self._tracker.track_llm_call(
            "test-cost-u6", "gpt-5.4-nano-2026-03-17", 1_000_000, 0
        )
        total = await self._tracker.get_daily_total("test-cost-u6")
        assert abs(total - 0.15) < 0.01

    async def test_multiple_calls_accumulate(self):
        for _ in range(3):
            await self._tracker.track_llm_call(
                "test-cost-u7", "gpt-5.4-nano-2026-03-17", 100_000, 0
            )
        total = await self._tracker.get_daily_total("test-cost-u7")
        assert abs(total - 0.045) < 0.005

    async def test_key_format_includes_user_and_date(self):
        today = datetime.utcnow().date().isoformat()
        key = self._tracker._key("test-cost-abc")
        assert "test-cost-abc" in key
        assert today in key

    async def test_daily_total_persists_in_redis(self):
        await self._tracker.track_llm_call(
            "test-cost-u8", "gpt-5.4-nano-2026-03-17", 500_000, 0
        )
        tracker2 = CostTracker(host="localhost", port=6379)
        total = await tracker2.get_daily_total("test-cost-u8")
        assert total > 0
