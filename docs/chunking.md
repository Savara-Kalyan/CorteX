# Chunking Service

**Location:** [rag/chunking/](../rag/chunking/)

Takes section-level `Document`s from the ingestion stage and splits them into overlapping token-bounded chunks using Chonkie's recursive pipeline. Returns a `(parents, children)` pair for parent-child retrieval architectures.

---

## File Structure

```
rag/chunking/
├── __init__.py     # Exports DocumentChunker, ChunkingException
└── service.py      # DocumentChunker — main entry point
```

---

## Architecture

**Chonkie Pipeline.** Chunking is done via a two-step `Pipeline`:
1. `chunk_with("recursive", chunk_size=N)` — splits text recursively on paragraph, sentence, and word boundaries using `RecursiveChunker`.
2. `refine_with("overlap", context_size=N)` — applies `OverlapRefinery` to add a sliding context window between adjacent chunks.

**Async and parallel.** `chunk_documents` processes all input documents concurrently via `asyncio.gather`. Each `_chunk_single` call awaits `pipeline.arun(text)` without blocking the event loop.

**Parent-child structure.** For every input `Document`, one parent is produced (carrying the full section text and a breadcrumb of its heading path) alongside N children (the actual token-bounded chunks). This enables parent-document retrieval: embed and search children, then fetch the parent for full context.

**Resilient.** If chunking fails for a section, the error is logged and that section is skipped. The rest of the batch continues unaffected.

---

## Key Class

### `DocumentChunker` — [service.py](../rag/chunking/service.py)

```python
from rag.chunking import DocumentChunker
```

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `512` | Maximum tokens per chunk. |
| `chunk_overlap` | `int` | `50` | Number of overlap characters added between adjacent chunks by `OverlapRefinery`. |

**Methods**

| Method | Signature | Description |
|---|---|---|
| `chunk_documents` | `(docs: List[Document]) -> Tuple[List[Document], List[Document]]` | Chunks a batch of section Documents. Returns `(parents, children)`. |
| `_chunk_single` | `(doc: Document) -> Tuple[List[Document], List[Document]]` | Internal. Chunks a single section and builds parent + child Documents. |

---

## Output Format

### Parents

Each parent is the original section `Document` enriched with:

| Field | Type | Description |
|---|---|---|
| `parent_id` | `str` | UUID assigned to this parent. Referenced by its children. |
| `breadcrumb` | `str` | Heading path, e.g. `report.pdf > Chapter 1 > Introduction`. |

All other metadata from the input `Document` is preserved as-is.

### Children

Each child carries all parent metadata plus:

| Field | Type | Description |
|---|---|---|
| `is_child` | `bool` | Always `True`. |
| `token_count` | `int` | Token count of this chunk as reported by Chonkie. |
| `chunk_start` | `int` | Start character index within the parent text. |
| `chunk_end` | `int` | End character index within the parent text. |

Empty or whitespace-only chunks are filtered out.

---

## Exceptions

| Exception | When raised |
|---|---|
| `ChunkingException` | Chonkie pipeline raises an error for a specific section. Carries `file_name`, `section_index`, and `reason`. |

Sections that raise `ChunkingException` are logged as errors and skipped; they do not abort the batch.

---

## Usage

```python
import asyncio
from rag.chunking import DocumentChunker
from langchain_core.documents import Document

async def main():
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)

    docs = [
        Document(
            page_content="Enterprise RAG systems require careful chunking...",
            metadata={"file_name": "rag_guide.pdf", "section_index": 0, "h1": "Overview"},
        )
    ]

    parents, children = await chunker.chunk_documents(docs)

    print(f"{len(parents)} parents, {len(children)} children")
    for child in children:
        print(child.metadata["breadcrumb"], "|", child.metadata["token_count"], "tokens")
        print(child.page_content[:100])

asyncio.run(main())
```

Used together with the ingestion service:

```python
from rag.ingestion import DocumentLoader
from rag.chunking import DocumentChunker

loader = DocumentLoader()
chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)

sections = await loader.load_directory("./documents")
parents, children = await chunker.chunk_documents(sections)

# children are ready for embedding and vector store ingestion
```
