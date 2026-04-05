# Evaluation Framework

**Location:** [rag/evaluation/](../rag/evaluation/)

End-to-end RAG evaluation framework. Measures retrieval quality (Precision@K, Recall@K, MRR), optionally measures generation quality (faithfulness, answer relevancy, context precision/recall), computes adversarial rejection rate, and produces a pass/fail verdict against configurable thresholds.

---

## File Structure

```
rag/evaluation/
├── __init__.py
├── dataset.py               # GoldenDataset — loads and filters golden items
├── generation_evaluator.py  # GenerationEvaluator — LLM-judge scoring
├── metrics.py               # Aggregation helpers
├── models.py                # Pydantic models for all evaluation data
├── retrieval_evaluator.py   # RetrievalEvaluator — P@K, R@K, MRR
└── service.py               # EvaluationService — orchestration facade
```

---

## Architecture

```
GoldenDataset
      │
      ▼
EvaluationService.run()
      │
      ├──▶ RetrievalEvaluator     →  RetrievalMetrics (P@K, R@K, MRR) per item
      │
      ├──▶ GenerationEvaluator    →  GenerationMetrics per item   [optional]
      │       (LLM judge)
      │
      ├──▶ Adversarial rejection rate
      │
      └──▶ Aggregate + per-category metrics
                │
                ▼
          EvaluationReport
            ├── overall: AggregateMetrics
            ├── per_category: list[CategoryReport]
            ├── adversarial_rejection_rate: float | None
            ├── item_results: list[ItemEvaluationResult]
            ├── decision: PassFailDecision ("READY_TO_SHIP" | "DO_NOT_SHIP")
            └── k: int
```

---

## Configuration

### `EvaluationConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | `5` | Rank cut-off for retrieval metrics |
| `run_generation_eval` | `bool` | `False` | Enable LLM-judge generation evaluation |
| `thresholds` | `dict[str, float]` | see below | Pass/fail score thresholds |
| `retrieval_concurrency` | `int` | `5` | Max concurrent search calls |
| `generation_concurrency` | `int` | `4` | Max concurrent LLM judge calls |

**Default thresholds:**

| Metric | Threshold |
|---|---|
| `precision_at_k` | 0.75 |
| `recall_at_k` | 0.70 |
| `mrr` | 0.70 |
| `faithfulness` | 0.80 |
| `answer_relevancy` | 0.80 |
| `context_precision` | 0.70 |
| `context_recall` | 0.70 |

---

## Key Classes

### `EvaluationService` — [service.py](../rag/evaluation/service.py)

```python
from rag.evaluation.service import EvaluationService, EvaluationConfig
```

**Constructor**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `search_fn` | `async (query: str, top_k: int) -> list[str]` | Yes | Returns ordered document IDs |
| `generation_fn` | `async (question: str, context: str) -> str` | Only if `run_generation_eval=True` | Returns the RAG system's answer |
| `llm_caller` | `async (prompt: str) -> str \| None` | No | Override for the LLM judge |
| `config` | `EvaluationConfig \| None` | No | Defaults used when not supplied |

**`run(dataset: GoldenDataset) -> EvaluationReport`**

Runs the full evaluation pipeline. Raises `EvaluationServiceError` if the dataset is empty or a critical failure occurs.

---

### `GoldenDataset` — [dataset.py](../rag/evaluation/dataset.py)

Loads golden items from a JSONL file. Each item has a question, ground-truth answer, relevant document IDs, and a category.

```python
from rag.evaluation.dataset import GoldenDataset

dataset = GoldenDataset.from_jsonl("golden.jsonl")
answerable  = dataset.answerable_items()    # non-adversarial items
adversarial = dataset.adversarial_items()   # out-of-scope / adversarial items
```

---

### `PassFailDecision`

| Field | Type | Description |
|---|---|---|
| `passed` | `bool` | True if all metrics meet thresholds |
| `verdict` | `str` | `"READY_TO_SHIP"` or `"DO_NOT_SHIP"` |
| `failing_metrics` | `list[str]` | Human-readable list of failing metrics with scores |

---

## Usage

```python
from rag.evaluation.service import EvaluationService, EvaluationConfig
from rag.evaluation.dataset import GoldenDataset
from rag.hybrid_search.service import HybridSearchService
from rag.hybrid_search.models import HybridSearchRequest

# Wire up your search function
search_svc = HybridSearchService()

async def search_fn(query: str, top_k: int) -> list[str]:
    response = await search_svc.search(HybridSearchRequest(query=query, top_k=top_k))
    return [r.id for r in response.results]

dataset = GoldenDataset.from_jsonl("tests/golden.jsonl")
config  = EvaluationConfig(k=5, run_generation_eval=False)
service = EvaluationService(search_fn=search_fn, config=config)

report = await service.run(dataset)
print(report.decision.verdict)          # "READY_TO_SHIP" or "DO_NOT_SHIP"
print(f"P@5={report.overall.precision_at_k:.3f}")
print(f"MRR={report.overall.mrr:.3f}")

if not report.decision.passed:
    for m in report.decision.failing_metrics:
        print(f"  FAIL: {m}")
```

### With Generation Evaluation

```python
async def generation_fn(question: str, context: str) -> str:
    # Call your RAG answer generator
    ...

config  = EvaluationConfig(k=5, run_generation_eval=True)
service = EvaluationService(
    search_fn=search_fn,
    generation_fn=generation_fn,
    config=config,
)
report = await service.run(dataset)
print(f"Faithfulness={report.overall.faithfulness:.3f}")
```
