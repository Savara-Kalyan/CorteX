# Ingestion Service

**Location:** [rag/ingestion/](../rag/ingestion/)

Discovers files in a directory, extracts text using a smart two-stage strategy (unstructured → Docling fallback), and splits the content into one LangChain `Document` per heading section — ready for downstream chunking.

---

## File Structure

```
rag/ingestion/
├── __init__.py     # Exports DocumentLoader, FileNotFoundException, ExtractionException
└── service.py      # DocumentLoader — main entry point
```

---

## Supported File Types

| Format      | Extension(s)                |
|-------------|-----------------------------|
| PDF         | `.pdf`                      |
| Word        | `.docx`                     |
| PowerPoint  | `.pptx`                     |
| Excel       | `.xlsx`                     |
| HTML        | `.html`                     |
| Markdown    | `.md`, `.ascii`             |

---

## Architecture

**Smart extraction.** Each file is processed with a two-stage strategy:
1. **unstructured** is tried first on a sample of the first 2 pages (PDFs) or the whole file.
2. If the extracted content is sparse (< 500 chars), **Docling** is used as a fallback (OCR-capable, layout-aware markdown export).

The `extractor_type` metadata field records which backend succeeded (`"unstructured"` or `"docling"`).

**Async and parallel.** `load_directory` processes all files concurrently via `asyncio.gather`. CPU/IO-bound extraction is offloaded to threads via `asyncio.to_thread`.

**Section splitting.** The resulting markdown is split on heading boundaries (`#` through `######`) using LangChain's `MarkdownHeaderTextSplitter` with `strip_headers=False`. Each heading block becomes one `Document` with structured metadata.

---

## Key Class

### `DocumentLoader` — [service.py](../rag/ingestion/service.py)

```python
from rag.ingestion import DocumentLoader
```

| Method | Signature | Description |
|---|---|---|
| `load_directory` | `(dir_path: str) -> List[Document]` | Scans a directory recursively and processes all supported files in parallel. |
| `process_single_file` | `(file_path: Path) -> List[Document]` | Extracts a single file and splits it into section `Document`s. |

---

## Output Format

Each `Document` has the following metadata:

| Field | Type | Description |
|---|---|---|
| `source` | `str` | Absolute path to the source file. |
| `file_name` | `str` | Filename with extension (e.g. `report.pdf`). |
| `file_type` | `str` | Extension without leading dot (e.g. `pdf`). |
| `section_index` | `int` | Zero-based position of this section within the file. |
| `section_heading` | `str` | Text of the deepest heading present. Empty for preamble sections. |
| `section_hash` | `str` | SHA-256 of the section text — useful for deduplication. |
| `ingestion_timestamp` | `str` | ISO-8601 timestamp of when the file was processed. |
| `text_len_chars` | `int` | Character count of the section text. |
| `extractor_type` | `str` | `"unstructured"` or `"docling"` — which backend extracted the file. |
| `h1`–`h6` | `str` | Header breadcrumb values from the splitter (only levels present above the section). |

---

## Exceptions

All exceptions extend `DocumentLoadingException`.

| Exception | When raised |
|---|---|
| `FileNotFoundException` | `load_directory` is called with a path that does not exist or is not a directory. |
| `ExtractionException` | Both unstructured and Docling fail, or Docling returns no content. |

---

## Usage

```python
import asyncio
from rag.ingestion import DocumentLoader

async def main():
    loader = DocumentLoader()

    docs = await loader.load_directory("./documents")
    print(f"Loaded {len(docs)} sections")

    for doc in docs:
        print(doc.metadata["file_name"], "|", doc.metadata["section_heading"])
        print(doc.page_content[:200])

asyncio.run(main())
```

Processing a single file:

```python
from pathlib import Path

docs = await loader.process_single_file(Path("./documents/report.pdf"))
# Returns one Document per heading section found in the file
```

Passing output directly to the chunking service:

```python
from rag.chunking import DocumentChunker

loader = DocumentLoader()
chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)

sections = await loader.load_directory("./documents")
parents, children = await chunker.chunk_documents(sections)
```
