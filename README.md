# CorteX

CorteX is an enterprise knowledge platform that ingests, processes, and makes searchable the documents and internal knowledge of an organization. It handles document loading from multiple formats, chunks them with overlap for RAG pipelines, embeds them, and stores the vectors for semantic search.

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
│   ├── ingestion.md           # Ingestion service docs
│   ├── chunking.md            # Chunking service docs
│   ├── embeddings.md          # Embeddings service docs
│   └── vector_store.md        # Vector store service docs
├── logging/
│   ├── config.py              # Logging configuration loader
│   └── logger.py              # Structured logger (get_logger)
├── rag/
│   ├── chunking/              # Splits sections into token-bounded chunks
│   │   ├── __init__.py
│   │   └── service.py
│   ├── embeddings/            # Embeds documents and queries
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── service.py
│   ├── ingestion/             # Converts files to section Documents
│   │   ├── __init__.py
│   │   └── service.py
│   └── vector_store/          # Stores and searches embeddings
│       ├── __init__.py
│       ├── base.py
│       ├── model.py
│       ├── pgvector.py
│       ├── qdrant.py
│       └── service.py
├── settings/
│   ├── config.py              # Settings loader (reads config.yaml)
│   ├── document-loading-config.yaml
│   └── logging_config.yaml
├── tests/
│   └── unit/
│       ├── test_chunking.py
│       └── test_document_loading.py
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

### Pipeline

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
   ▼
VectorStoreService.search(query_embedding)
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
