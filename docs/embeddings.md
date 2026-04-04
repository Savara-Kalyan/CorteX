# Embeddings Service

**Location:** [rag/embeddings/](../rag/embeddings/)

Singleton service that converts LangChain `Document`s or raw strings into vector embeddings. Supports OpenAI and Anthropic backends, selected via `config.yaml`.

---

## File Structure

```
rag/embeddings/
├── __init__.py       # Exports EmbeddingService
├── base.py           # BaseEmbeddingModel — abstract interface
├── openai.py         # OpenAIEmbeddingModel
├── anthropic.py      # AnthropicEmbeddingModel
└── service.py        # EmbeddingService — singleton facade
```

---

## Architecture

**Singleton.** `EmbeddingService` is a singleton — instantiating it multiple times returns the same instance. The backend is loaded once on first instantiation.

**Provider selection.** The active provider is read from `config.yaml` under `EMBEDDINGS.PROVIDER`. Supported values: `openai`, `anthropic`. The matching backend is imported lazily.

**Pluggable backend.** All backends implement `BaseEmbeddingModel`. A custom backend can be injected via the constructor for testing or alternative providers.

---

## Configuration

In `config.yaml`:

```yaml
EMBEDDINGS:
  PROVIDER: openai           # or "anthropic"
  MODEL: text-embedding-3-small
```

For Anthropic:

```yaml
EMBEDDINGS:
  PROVIDER: anthropic
  MODEL: voyage-3
```

---

## Key Class

### `EmbeddingService` — [service.py](../rag/embeddings/service.py)

```python
from rag.embeddings import EmbeddingService
```

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `BaseEmbeddingModel \| None` | `None` | Override the backend. If `None`, loaded from `config.yaml`. |

**Methods**

| Method | Signature | Description |
|---|---|---|
| `embed_documents` | `(docs: list) -> list[list[float]]` | Embeds a list of Documents or strings. Returns one vector per input. |
| `embed_query` | `(query: str) -> list[float]` | Embeds a single query string. Returns one vector. |

---

## Backends

### `OpenAIEmbeddingModel` — [openai.py](../rag/embeddings/openai.py)

Uses `langchain_openai.OpenAIEmbeddings`. Requires `OPENAI_API_KEY` in the environment.

### `AnthropicEmbeddingModel` — [anthropic.py](../rag/embeddings/anthropic.py)

Uses `anthropic.AsyncAnthropic`. Requires `ANTHROPIC_API_KEY` in the environment. Calls `client.embeddings.create` directly.

---

## Usage

```python
import asyncio
from rag.embeddings import EmbeddingService

async def main():
    svc = EmbeddingService()

    vectors = await svc.embed_documents(["hello world", "enterprise RAG"])
    print(len(vectors), "vectors of dim", len(vectors[0]))

    query_vec = await svc.embed_query("what is RAG?")
    print("query dim:", len(query_vec))

asyncio.run(main())
```

Injecting a custom backend for tests:

```python
from rag.embeddings.base import BaseEmbeddingModel
from rag.embeddings import EmbeddingService

class StubEmbedding(BaseEmbeddingModel):
    async def embed_documents(self, docs):
        return [[0.0] * 1536 for _ in docs]
    async def embed_query(self, query):
        return [0.0] * 1536

svc = EmbeddingService(backend=StubEmbedding())
```
