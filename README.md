# CorteX

CorteX is an enterprise knowledge platform that ingests, processes, and makes searchable the documents and internal knowledge of an organization. It handles document loading from multiple formats, extracts text using a tiered fallback strategy, deduplicates content, and prepares documents for downstream RAG pipelines.

---

## Project Structure

```
CorteX/
├── app/
│   ├── logging/
│   │   ├── config.py              # Logging configuration loader
│   │   └── logger.py              # Structured logger (get_logger)
│   └── services/
│       └── document_loading/      # Document loading service (see below)
│           ├── __init__.py
│           ├── analyzers.py       # Content quality analysis
│           ├── config.py          # Service configuration dataclasses
│           ├── deduplication.py   # Content/file hash deduplication
│           ├── exceptions.py      # Service-specific exceptions
│           ├── extractors.py      # Tiered extraction orchestration
│           ├── loader.py          # File type detection and loaders
│           ├── models.py          # Dataclasses and enums
│           ├── service.py         # DocumentLoadingService (main entry)
│           └── validators.py      # File and directory validation
├── scripts/
│   └── download_docs.py           # Scrapes GitLab handbook pages to markdown
├── settings/
│   ├── document-loading-config.yaml  # Document loading settings
│   └── logging_config.yaml           # Logging settings
├── tests/
│   └── unit/
│       └── services/
│           └── document_loading/  # Unit tests for document loading
├── config.yaml                    # Top-level app config (LLM provider, etc.)
├── requirements.txt
├── run.sh
└── freeze.sh
```

---

## Services

| Service | Location | Docs |
|---|---|---|
| Document Loading | [app/services/document_loading/](app/services/document_loading/) | [docs/services/document_loading.md](docs/services/document_loading.md) |

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
