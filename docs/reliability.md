# Reliability

**Location:** [reliability/](../reliability/)

Production reliability utilities: per-user daily cost tracking backed by Valkey, and a fallback handler.

---

## File Structure

```
reliability/
├── cost_tracker.py   # CostTracker — LLM and embedding cost tracking
└── fallback.py
```

---

## Cost Tracker — [cost_tracker.py](../reliability/cost_tracker.py)

Tracks LLM and embedding API costs per user per day. Backed by **Valkey** (Redis-compatible), running as the `valkey` service in `docker-compose`. Keys expire after 30 days.

### Supported Models & Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| `gpt-5.4-nano-2026-03-17` | $0.15 | $0.60 |
| `text-embedding-3-small` | $0.02 | — |

### `CostTracker`

```python
from reliability.cost_tracker import CostTracker

tracker = CostTracker()  # defaults: host="valkey", port=6379, db=0
```

#### `track_llm_call(user_id, model, input_tokens, output_tokens) -> dict`

Records the cost of an LLM API call and increments the user's daily total.

Returns: `{"call_cost": float, "daily_total": float, "input_tokens": int, "output_tokens": int}`

#### `track_embedding_call(user_id, model, input_tokens) -> dict`

Records the cost of an embedding API call.

Returns: `{"call_cost": float, "daily_total": float, "input_tokens": int}`

#### `check_budget(user_id, daily_limit=1.0) -> bool`

Returns `True` if the user's daily spend is below `daily_limit` (dollars). Default limit: $1.00.

#### `get_daily_total(user_id) -> float`

Returns the user's total spend for today in dollars.

### Usage

```python
from reliability.cost_tracker import CostTracker

tracker = CostTracker()

# Before making an LLM call
if not tracker.check_budget("user-123"):
    raise Exception("Daily budget exceeded")

# After an LLM call
result = tracker.track_llm_call(
    user_id="user-123",
    model="gpt-5.4-nano-2026-03-17",
    input_tokens=500,
    output_tokens=200,
)
print(f"Call cost: ${result['call_cost']:.6f}")
print(f"Daily total: ${result['daily_total']:.4f}")

# After an embedding call
tracker.track_embedding_call(
    user_id="user-123",
    model="text-embedding-3-small",
    input_tokens=300,
)
```

### Storage

Costs are stored in Valkey at keys of the form:

```
cost:{user_id}:{YYYY-MM-DD}
```

Values are stored as integer cents (cost × 100) to avoid floating-point rounding. Keys are set with a 30-day TTL on each write.
