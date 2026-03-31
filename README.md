# CorteX

CorteX is an enterprise knowledge platform that ingests, processes, and makes searchable the documents and internal knowledge of an organization. It handles document loading from multiple formats, converts them to structured sections via Docling, chunks them with overlap for RAG pipelines, and prepares the output for embedding and vector store ingestion.

---

## Project Structure

```
CorteX/
├── app/
│   ├── logging/
│   │   ├── config.py              # Logging configuration loader
│   │   └── logger.py              # Structured logger (get_logger)
│   └── services/
│       ├── document_loading/      # Converts files to section Documents
│       │   ├── __init__.py
│       │   └── service.py
│       └── chunking/              # Splits sections into token-bounded chunks
│           ├── __init__.py
│           └── service.py
├── docs/
│   └── services/
│       ├── document_loading.md    # Document loading service docs
│       └── chunking.md            # Chunking service docs
├── scripts/
│   └── download_docs.py           # Scrapes GitLab handbook pages to markdown
├── settings/
│   ├── document-loading-config.yaml
│   └── logging_config.yaml
├── tests/
│   └── unit/
│       ├── test_chunking.py
│       └── test_document_loading.py
├── config.yaml                    # Top-level app config
├── requirements.txt
├── run.sh
└── freeze.sh
```

---

## Services

| Service | Location | Docs |
|---|---|---|
| Document Loading | [app/services/document_loading/](app/services/document_loading/) | [docs/services/document_loading.md](docs/services/document_loading.md) |
| Chunking | [app/services/chunking/](app/services/chunking/) | [docs/services/chunking.md](docs/services/chunking.md) |

### Pipeline

```
Directory
   │
   ▼
DocumentLoader.load_directory()
   │  Converts files → markdown (Docling)
   │  Splits markdown → one Document per heading section
   ▼
DocumentChunker.chunk_documents()
   │  Recursive chunking (Chonkie Pipeline)
   │  Overlap refinement between adjacent chunks
   ▼
(parents, children)
   │
   ▼
Embedding + Vector Store
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
