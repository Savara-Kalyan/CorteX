# Query Understanding

**Location:** [rag/query_understanding/](../rag/query_understanding/)

Pre-processing layer that transforms a raw user query into a structured object before it reaches the search layer. Runs reformulation, expansion, and intent classification in sequence.

---

## File Structure

```
rag/query_understanding/
├── __init__.py
├── query_classification.py   # QueryIntentClassifier — intent + answerability
├── query_expansion.py        # QueryExpander — generates 3 alternative queries
├── query_reformulation.py    # QueryReformulator — cleans and rewrites the query
└── service.py                # QueryUnderstanding — orchestration facade
```

---

## Architecture

The `QueryUnderstanding` service runs three steps in sequence:

1. **Reformulation** — rewrites the raw query for clarity and precision.
2. **Expansion** — generates up to 3 semantically equivalent alternative queries.
3. **Classification** — classifies the reformulated query into a domain and determines answerability.

All three components use an LLM configured via `settings.query_understanding`.

---

## Configuration

In `config.yaml`:

```yaml
QUERY_UNDERSTANDING:
  MODEL: gpt-4o-mini
  TEMPERATURE: 0.0
```

---

## Key Class

### `QueryUnderstanding` — [service.py](../rag/query_understanding/service.py)

```python
from rag.query_understanding.service import QueryUnderstanding
```

#### `process(query: str) -> dict`

Runs the full pipeline and returns a dict with:

| Key | Type | Description |
|---|---|---|
| `query` | `str` | Original user query |
| `reformulated` | `str` | Cleaned/rewritten query |
| `all_queries` | `list[str]` | `[reformulated] + up to 2 expanded variants` |
| `intent` | `str` | Classified domain: `hr`, `engineering`, `culture`, `out_of_scope` |
| `answerable` | `bool` | Whether the query is within scope |
| `confidence` | `float` | Classification confidence (0.0–1.0) |

Raises the underlying exception if any step fails.

---

## Components

### `QueryReformulator` — [query_reformulation.py](../rag/query_understanding/query_reformulation.py)

Rewrites the input query to be more precise and search-friendly using an LLM chain.

### `QueryExpander` — [query_expansion.py](../rag/query_understanding/query_expansion.py)

Generates up to 3 alternative phrasings of the reformulated query, one per line. Falls back to an empty list on failure.

### `QueryIntentClassifier` — [query_classification.py](../rag/query_understanding/query_classification.py)

Classifies the query into one of four intents and returns a JSON response:

```json
{"intent": "hr", "confidence": 0.95, "answerable": true}
```

Falls back to `{"intent": "out_of_scope", "confidence": 0.0, "answerable": false}` on failure.

**Intent domains:**

| Intent | Description |
|---|---|
| `hr` | Leave, payroll, onboarding, benefits, performance reviews |
| `engineering` | Tech stack, architecture, deployments, tooling, workflows |
| `culture` | Company values, org structure, mission, team norms, events |
| `out_of_scope` | Does not belong to any known domain |

---

## Usage

```python
from rag.query_understanding.service import QueryUnderstanding

qu = QueryUnderstanding()
result = await qu.process("how do I request parental leave?")

if result["answerable"]:
    print(result["all_queries"])   # queries to send to search
    print(result["intent"])        # "hr"
else:
    # Reject the query — out of scope
    pass
```
