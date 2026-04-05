# Hybrid Search

**Location:** [rag/hybrid_search/](../rag/hybrid_search/)

Combines dense vector search and sparse BM25 keyword search, merges results with Reciprocal Rank Fusion (RRF), and returns a single ranked list. Both search paths run concurrently; if one fails, the other's results are still fused and returned (degraded mode).

---

## File Structure

```
rag/hybrid_search/
├── __init__.py
├── bm25.py       # KeywordSearcher — BM25 search against the document store
├── models.py     # HybridSearchRequest, HybridSearchResponse, SearchResult
├── reranker.py   # RRFReranker — Reciprocal Rank Fusion
├── service.py    # HybridSearchService — orchestration facade
└── vector.py     # VectorSearcher — dense vector search
```

---

## Architecture

```
HybridSearchRequest
        │
        ├──(async)──▶ VectorSearcher.search()    # dense retrieval
        └──(async)──▶ KeywordSearcher.search()   # BM25 retrieval
                │
                ▼
          RRFReranker.fuse()
                │
                ▼
        HybridSearchResponse
```

Both paths run via `asyncio.gather`. A failed path is logged at ERROR and treated as an empty list — only when **both** paths fail is `HybridSearchError` raised.

---

## Key Classes

### `HybridSearchService` — [service.py](../rag/hybrid_search/service.py)

```python
from rag.hybrid_search.service import HybridSearchService
from rag.hybrid_search.models import HybridSearchRequest
```

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_service` | `EmbeddingService \| None` | `None` | Pre-built embedding service (useful for testing) |
| `rrf_k` | `int` | `60` | RRF smoothing constant |

**`search(request: HybridSearchRequest) -> HybridSearchResponse`**

Runs hybrid search. Raises `HybridSearchError` only when both paths fail.

---

### `HybridSearchRequest` — [models.py](../rag/hybrid_search/models.py)

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Natural-language query |
| `top_k` | `int` | `5` | Results to return after fusion |
| `rrf_k` | `int` | `60` | RRF constant — higher = less rank-sensitive |
| `fetch_k` | `int` | `20` | Candidates fetched from each searcher before fusion |

### `HybridSearchResponse` — [models.py](../rag/hybrid_search/models.py)

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original query |
| `results` | `list[SearchResult]` | RRF-ranked results |
| `total_vector_candidates` | `int` | Candidates returned by vector search |
| `total_bm25_candidates` | `int` | Candidates returned by BM25 search |

### `SearchResult` — [models.py](../rag/hybrid_search/models.py)

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Document ID |
| `content` | `str` | Document text |
| `source_file` | `str \| None` | Source file path |
| `page_number` | `int \| None` | Page number |
| `metadata` | `dict` | Arbitrary metadata |
| `rrf_score` | `float` | Fused RRF score |
| `vector_rank` | `int \| None` | Rank in vector results (1-based) |
| `bm25_rank` | `int \| None` | Rank in BM25 results (1-based) |

---

## RRF Reranker — [reranker.py](../rag/hybrid_search/reranker.py)

Implements Reciprocal Rank Fusion (Cormack et al., 2009):

```
RRF_score(d) = Σ  1 / (rank_i(d) + k)
```

Documents appearing in only one list receive a contribution from that list; appearing in both doubles their boost. `k=60` is the value validated in the original paper.

---

## Usage

```python
from rag.hybrid_search.service import HybridSearchService
from rag.hybrid_search.models import HybridSearchRequest

svc = HybridSearchService()

response = await svc.search(
    HybridSearchRequest(query="AWS EC2 pricing", top_k=5)
)

for result in response.results:
    print(f"[{result.rrf_score:.4f}] {result.content[:80]}")
```

Customise retrieval depth and fusion sensitivity:

```python
request = HybridSearchRequest(
    query="parental leave policy",
    top_k=10,
    fetch_k=40,   # fetch more candidates before fusion
    rrf_k=30,     # lower k = top ranks matter more
)
```
