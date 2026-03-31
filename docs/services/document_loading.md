# Document Loading Service

**Location:** [app/services/document_loading/](../../app/services/document_loading/)

Discovers files in a directory, converts them to markdown via Docling, and splits the content into one LangChain `Document` per heading section — ready for downstream chunking.

---

## File Structure

```
app/services/document_loading/
├── __init__.py     # Exports DocumentLoader, FileNotFoundException, ExtractionException
└── service.py      # DocumentLoader — main entry point
```

---

## Supported File Types

| Format      | Extension(s)                         |
|-------------|--------------------------------------|
| PDF         | `.pdf`                               |
| Word        | `.docx`                              |
| PowerPoint  | `.pptx`                              |
| Excel       | `.xlsx`                              |
| HTML        | `.html`                              |
| Markdown    | `.md`, `.ascii`                      |

---

## Architecture

**Async and parallel.** `load_directory` processes all files concurrently via `asyncio.gather`. Each `process_single_file` call offloads the CPU/IO-bound Docling conversion and markdown splitting to a thread via `asyncio.to_thread`.

**Docling conversion.** Every file is converted to markdown using `DoclingLoader` (from `langchain-docling`). This handles layout-aware extraction across PDFs, Word, PowerPoint, and other formats.

**Section splitting.** The resulting markdown is split on heading boundaries (`#` through `######`) using LangChain's `MarkdownHeaderTextSplitter` with `strip_headers=False`. Each heading block becomes one `Document` with structured metadata.

---

## Key Class

### `DocumentLoader` — [service.py](../../app/services/document_loading/service.py)

```python
from app.services.document_loading import DocumentLoader
```

| Method | Signature | Description |
|---|---|---|
| `load_directory` | `(dir_path: str) -> List[Document]` | Scans a directory recursively and processes all supported files in parallel. |
| `process_single_file` | `(file_path: Path) -> List[Document]` | Converts a single file to markdown, then splits it into section `Document`s. |

---

## Output Format

Each `Document` returned by `process_single_file` (and aggregated by `load_directory`) has the following metadata:

| Field | Type | Description |
|---|---|---|
| `source` | `str` | Absolute path to the source file. |
| `file_name` | `str` | Filename with extension (e.g. `report.pdf`). |
| `file_type` | `str` | Extension without leading dot (e.g. `pdf`). |
| `section_index` | `int` | Zero-based position of this section within the file. |
| `section_heading` | `str` | Text of the deepest heading present (e.g. `Introduction`). Empty string for preamble sections. |
| `section_hash` | `str` | SHA-256 of the section text — useful for deduplication. |
| `ingestion_timestamp` | `str` | ISO-8601 timestamp of when the file was processed. |
| `text_len_chars` | `int` | Character count of the section text. |
| `h1`–`h6` | `str` | Header breadcrumb values from the splitter (present only for the levels that appear above the section). |

---

## Exceptions

All exceptions extend `DocumentLoadingException`.

| Exception | When raised |
|---|---|
| `FileNotFoundException` | `load_directory` is called with a path that does not exist or is not a directory. |
| `ExtractionException` | Docling returns no content for a file, or an unexpected error occurs during conversion. |

---

## Usage

```python
import asyncio
from app.services.document_loading import DocumentLoader

async def main():
    loader = DocumentLoader()

    # Load an entire directory
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

Passing the output directly to the chunking service:

```python
from app.services.chunking import DocumentChunker

loader = DocumentLoader()
chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)

sections = await loader.load_directory("./documents")
parents, children = await chunker.chunk_documents(sections)
```
