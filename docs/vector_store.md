# Vector Store Service

**Location:** [rag/vector_store/](../rag/vector_store/)

Singleton service for inserting and searching vector embeddings. Supports PGVector (PostgreSQL) and Qdrant backends, selected via `config.yaml`.

---

## File Structure

```
rag/vector_store/
├── __init__.py       # Exports VectorStoreService
├── base.py           # BaseVectorStore — abstract interface
├── model.py          # DocumentInsert — Pydantic model for inserts
├── pgvector.py       # PGVectorStore — PostgreSQL + pgvector
├── qdrant.py         # QdrantVectorStore — Qdrant
└── service.py        # VectorStoreService — singleton facade
```

---

## Architecture

**Singleton.** `VectorStoreService` is a singleton — the backend is loaded once on first instantiation.

**Provider selection.** The active provider is read from `config.yaml` under `VECTOR_STORE.PROVIDER`. Supported values: `pgvector`, `qdrant`. The matching backend is imported lazily.

**Pluggable backend.** All backends implement `BaseVectorStore` (`insert`, `search`).

**Embedding integration.** Each backend holds a reference to `EmbeddingService` and calls `embed_documents` internally during `insert`.

---

## Configuration

In `config.yaml`:

```yaml
VECTOR_STORE:
  PROVIDER: pgvector
  CONNECTION_STRING: "postgresql://user:pass@host:5432/db"
```

For Qdrant:

```yaml
VECTOR_STORE:
  PROVIDER: qdrant
  URL: "http://localhost:6333"
  API_KEY: ""               # optional, for Qdrant Cloud
  COLLECTION_NAME: "documents"
```

---

## Key Class

### `VectorStoreService` — [service.py](../rag/vector_store/service.py)

```python
from rag.vector_store import VectorStoreService
```

**Methods**

| Method | Signature | Description |
|---|---|---|
| `insert` | `(docs: list) -> None` | Embeds and inserts a list of Documents into the vector store. |
| `search` | `(query_embedding: list[float], top_k: int = 5) -> list` | Nearest-neighbour search. Returns up to `top_k` results. |

---

## Backends

### `PGVectorStore` — [pgvector.py](../rag/vector_store/pgvector.py)

Uses `psycopg` (async) + `pgvector`. Inserts rows into a `documents` table and queries using the `<->` distance operator.

**Schema fields written per insert:**

| Column | Source |
|---|---|
| `content` | `doc.page_content` |
| `embedding` | from `EmbeddingService` |
| `source_file` | `doc.metadata["source"]` |
| `page_number` | `doc.metadata["page"]` |
| `chunk_index` | position in batch |
| `total_chunks` | batch size |
| `doc_hash` | from `DocumentInsert` model |
| `doc_type` | `doc.metadata["file_type"]` |
| `chunk_length` | `len(doc.page_content)` |
| `metadata` | full metadata dict (JSONB) |

### `QdrantVectorStore` — [qdrant.py](../rag/vector_store/qdrant.py)

Uses `qdrant-client` (async). Inserts `PointStruct` objects with a deterministic UUID (derived from `doc_hash` via `uuid.uuid5`) and upserts them into the configured collection. Searches via `client.search`.

---

## DocumentInsert Model — [model.py](../rag/vector_store/model.py)

Pydantic model that normalises document fields before insertion. Used by both backends.

---

## Usage

```python
import asyncio
from rag.vector_store import VectorStoreService

async def main():
    store = VectorStoreService()

    # Insert chunked documents (embedding happens internally)
    await store.insert(children)

    # Search (pass a pre-computed query embedding)
    from rag.embeddings import EmbeddingService
    query_vec = await EmbeddingService().embed_query("what is RAG?")
    results = await store.search(query_vec, top_k=5)
    for row in results:
        print(row)

asyncio.run(main())
```

Full pipeline from ingestion to storage:

```python
from rag.ingestion import DocumentLoader
from rag.chunking import DocumentChunker
from rag.vector_store import VectorStoreService

loader = DocumentLoader()
chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
store = VectorStoreService()

sections = await loader.load_directory("./documents")
parents, children = await chunker.chunk_documents(sections)
await store.insert(children)
```
