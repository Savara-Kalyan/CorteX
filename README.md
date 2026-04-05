# CorteX

CorteX is an enterprise knowledge platform that ingests, processes, and makes searchable the documents and internal knowledge of an organization. It handles document loading from multiple formats, chunks them with overlap for RAG pipelines, embeds them, stores vectors for semantic search, and provides hybrid retrieval with access control, query understanding, evaluation, cost tracking, and structured observability.

---

## Project Structure

```
CorteX/
├── .devcontainer/
│   ├── Dockerfile
│   ├── devcontainer.json
│   └── docker-compose.yaml
├── .github/
│   └── workflows/
│       └── test.yaml
├── docs/
│   ├── ingestion.md             # Ingestion service docs
│   ├── chunking.md              # Chunking service docs
│   ├── embeddings.md            # Embeddings service docs
│   ├── vector_store.md          # Vector store service docs
│   ├── access_control.md        # Access control service docs
│   ├── hybrid_search.md         # Hybrid search service docs
│   ├── query_understanding.md   # Query understanding service docs
│   ├── evaluation.md            # Evaluation framework docs
│   ├── reliability.md           # Cost tracking & fallback docs
│   └── observability.md         # Structured logging docs
├── observability/
│   └── logging/
│       ├── config.py            # Logging configuration loader
│       └── logger.py            # Structured logger (get_logger)
├── rag/
│   ├── access_control/          # Domain-filtered search with RBAC
│   │   ├── __init__.py
│   │   └── service.py
│   ├── chunking/                # Splits sections into token-bounded chunks
│   │   ├── __init__.py
│   │   └── service.py
│   ├── embeddings/              # Embeds documents and queries
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── service.py
│   ├── evaluation/              # RAG evaluation framework
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── generation_evaluator.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── retrieval_evaluator.py
│   │   └── service.py
│   ├── hybrid_search/           # Vector + BM25 search with RRF fusion
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── models.py
│   │   ├── reranker.py
│   │   ├── service.py
│   │   └── vector.py
│   ├── ingestion/               # Converts files to section Documents
│   │   ├── __init__.py
│   │   └── service.py
│   ├── query_understanding/     # Query reformulation, expansion, intent classification
│   │   ├── __init__.py
│   │   ├── query_classification.py
│   │   ├── query_expansion.py
│   │   ├── query_reformulation.py
│   │   └── service.py
│   └── vector_store/            # Stores and searches embeddings
│       ├── __init__.py
│       ├── base.py
│       ├── model.py
│       ├── pgvector.py
│       ├── qdrant.py
│       └── service.py
├── reliability/
│   ├── cost_tracker.py          # Per-user daily cost tracking (Valkey-backed)
│   └── fallback.py
├── settings/
│   ├── config.py                # Settings loader (reads config.yaml)
│   ├── document-loading-config.yaml
│   └── logging_config.yaml
├── tests/
│   └── unit/
│       ├── test_access_control.py
│       ├── test_chunking.py
│       ├── test_document_loading.py
│       └── test_vector_store.py
├── config.yaml
├── pyrightconfig.json
├── pytest.ini
├── requirements.txt
├── run.sh
└── freeze.sh
```

---

## Modules

| Module | Location | Docs |
|---|---|---|
| Ingestion | [rag/ingestion/](rag/ingestion/) | [docs/ingestion.md](docs/ingestion.md) |
| Chunking | [rag/chunking/](rag/chunking/) | [docs/chunking.md](docs/chunking.md) |
| Embeddings | [rag/embeddings/](rag/embeddings/) | [docs/embeddings.md](docs/embeddings.md) |
| Vector Store | [rag/vector_store/](rag/vector_store/) | [docs/vector_store.md](docs/vector_store.md) |
| Query Understanding | [rag/query_understanding/](rag/query_understanding/) | [docs/query_understanding.md](docs/query_understanding.md) |
| Hybrid Search | [rag/hybrid_search/](rag/hybrid_search/) | [docs/hybrid_search.md](docs/hybrid_search.md) |
| Access Control | [rag/access_control/](rag/access_control/) | [docs/access_control.md](docs/access_control.md) |
| Evaluation | [rag/evaluation/](rag/evaluation/) | [docs/evaluation.md](docs/evaluation.md) |
| Reliability | [reliability/](reliability/) | [docs/reliability.md](docs/reliability.md) |
| Observability | [observability/](observability/) | [docs/observability.md](docs/observability.md) |

---

## Pipelines

### Ingestion

```
Directory
   │
   ▼
DocumentLoader.load_directory()
   │  Smart extraction: unstructured → Docling fallback
   │  Splits markdown → one Document per heading section
   ▼
DocumentChunker.chunk_documents()
   │  Recursive chunking (Chonkie Pipeline)
   │  Overlap refinement between adjacent chunks
   │  Returns (parents, children)
   ▼
VectorStoreService.insert(children)
   │  EmbeddingService embeds each chunk
   │  Vectors written to PGVector or Qdrant
```

### Query (Retrieval)

```
User Query
   │
   ▼
QueryUnderstanding.process()
   │  Reformulate → Expand (3 variants) → Classify intent
   │  Returns: reformulated query, all_queries, intent, answerable, confidence
   ▼
 [if answerable]
   │
   ├──▶ HybridSearchService.search()          ──▶ RRF-fused results
   │       Vector search + BM25 in parallel
   │       Reciprocal Rank Fusion (k=60)
   │
   └──▶ AccessControlService.search_accessible()  ──▶ RBAC-filtered results
           Domain policy enforcement
           Parallel domain fan-out
           Merged + re-ranked by cosine similarity
```

---

## Configuration

Edit `config.yaml` to switch providers:

```yaml
LLM:
  PROVIDER: openai
  MODEL: gpt-4o

EMBEDDINGS:
  PROVIDER: openai          # or "anthropic"
  MODEL: text-embedding-3-small

VECTOR_STORE:
  PROVIDER: pgvector        # or "qdrant"
  CONNECTION_STRING: "postgresql://user:pass@host:5432/db"

QUERY_UNDERSTANDING:
  MODEL: gpt-4o-mini
  TEMPERATURE: 0.0
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start the app
bash run.sh
```
