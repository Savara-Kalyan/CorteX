# Document Loading Service

**Location:** [app/services/document_loading/](../../app/services/document_loading/)

The document loading service discovers files in a directory, extracts text using a tiered fallback strategy, deduplicates results, and returns structured output for downstream use.

---

## File Structure

```
app/services/document_loading/
├── service.py          # DocumentLoadingService — main entry point
├── extractors.py       # DocumentExtractor — per-file extraction with fallback logic
├── loader.py           # File type detection and individual loaders
├── analyzers.py        # ContentQualityAnalyzer — decides when to trigger fallback
├── deduplication.py    # DeduplicationManager — SHA-256 hash-based dedup
├── validators.py       # DocumentLoadingValidator — directory and file validation
├── models.py           # Dataclasses and enums
├── config.py           # Configuration dataclasses and env loading
└── exceptions.py       # Service-specific exceptions
```

---

## Supported File Types

| Format     | Extension(s)            |
|------------|-------------------------|
| PDF        | `.pdf`                  |
| Markdown   | `.md`                   |
| Word       | `.docx`                 |
| Plain text | `.txt`                  |
| Images     | `.png`, `.jpg`, `.jpeg` |

---

## Extraction Strategy

Each file goes through a tiered pipeline. The service tries strategies in order and stops at the first one that produces content above `char_count_threshold`.

```
Standard  -->  Docling  -->  OCR  -->  VLM
```

| Step | Strategy | Library | When used |
|---|---|---|---|
| 1 | Standard | PDFPlumber, UnstructuredMarkdown, Docx2txt, TextLoader | Always tried first |
| 2 | Docling | `docling.DocumentConverter` | PDFs where Standard yields sparse content |
| 3 | OCR | `UnstructuredPDFLoader` / `UnstructuredImageLoader` | PDFs and images where Standard and Docling both fall short |
| 4 | VLM | GPT-4 Vision via LangChain | Last resort; disabled by default |

If no strategy improves over Standard, the Standard result is returned as-is. If Standard produces nothing and all fallbacks fail, the file is marked with `ExtractionStrategy.FAILED`.

The strategy used is recorded in `ExtractionMetrics.extraction_strategy` on every result.

---

## Architecture

**Async and parallel.** `load_directory` and `load_with_task` are `async` methods. All files in a batch are processed with `asyncio.gather`. Blocking loader calls run in a `ThreadPoolExecutor` via `loop.run_in_executor`.

**Concurrency control.** An `asyncio.Semaphore` caps how many files are extracted at once. This is controlled by `max_concurrent_files` in config.

**Validation.** Before extraction begins, each file is checked: must exist, have a supported extension, and fall within the configured size bounds. Invalid files are skipped and logged; they do not raise exceptions.

**Deduplication.** After extraction, each result is checked against a SHA-256 hash registry (`seen_hashes`). A duplicate is any file whose extracted content (or raw bytes, depending on `method`) hashes to a value already seen in the current session. Duplicates are dropped from the output. The registry is cleared on `service.shutdown()`.

**Latency tracking.** Every task logs: time to first result, per-strategy timing for each file, total task duration, and throughput in files/sec.

---

## Key Classes

### `DocumentLoadingService` — [service.py](../../app/services/document_loading/service.py)

Main entry point. Owns the `ThreadPoolExecutor`, `DocumentExtractor`, `DocumentLoadingValidator`, and `DeduplicationManager`. Use as a context manager to ensure clean shutdown.

| Method | Description |
|---|---|
| `load_directory(directory_path, pattern, file_types, max_concurrent)` | Discover and load all supported files from a directory. |
| `load_with_task(task, max_concurrent)` | Same as above but accepts a `LoadingTask` for more control. |
| `shutdown()` | Shuts down the thread pool and resets deduplication state. |

---

### `DocumentExtractor` — [extractors.py](../../app/services/document_loading/extractors.py)

Handles per-file extraction. Calls `ContentQualityAnalyzer` after each strategy to decide whether to proceed to the next fallback.

| Method | Description |
|---|---|
| `extract_file(file_path, executor)` | Extract a single file. Returns `ExtractionResult`. |

---

### `FileTypeDetector` — [loader.py](../../app/services/document_loading/loader.py)

Maps file extensions to `FileType` enum values.

| Method | Description |
|---|---|
| `detect(file_path)` | Returns the `FileType` for a given path. |
| `is_supported(file_path)` | Returns `True` if the extension maps to a known type. |

---

### Loaders — [loader.py](../../app/services/document_loading/loader.py)

Each loader is a standalone class with `async` static/class methods. All blocking library calls are wrapped with `loop.run_in_executor`.

| Class | Strategy | Notes |
|---|---|---|
| `StandardLoader` | Standard | Routes to the correct loader by `FileType`. |
| `DoclingLoader` | Docling | PDF only. Falls back silently if `docling` is not installed. |
| `OCRLoader` | OCR | PDF and image files. Uses Unstructured under the hood. |
| `VLMLoader` | VLM | Requires `langchain-openai`. Initialized only when `enable_vlm: true`. |

---

### `DeduplicationManager` — [deduplication.py](../../app/services/document_loading/deduplication.py)

Maintains an in-memory `dict[hash -> Path]` for the lifetime of a service instance.

| Method | Description |
|---|---|
| `is_duplicate(result)` | Returns `(True, original_path)` if the result is a duplicate, else `(False, None)`. |
| `reset()` | Clears the hash registry. |

Two hash methods are supported, set via `deduplication.method` in config:
- `content_hash` — SHA-256 of the concatenated extracted text.
- `file_hash` — SHA-256 of the raw file bytes.

---

### `DocumentLoadingValidator` — [validators.py](../../app/services/document_loading/validators.py)

| Method | Description |
|---|---|
| `validate_directory(path)` | Checks the path exists and is a directory. Returns `(bool, error_msg)`. |
| `validate_file(path)` | Checks existence, supported type, and size bounds. Returns `(bool, error_msg)`. |
| `validate_files(paths)` | Batch validation. Returns `(valid_files, error_list)`. |

---

### `ContentQualityAnalyzer` — [analyzers.py](../../app/services/document_loading/analyzers.py)

Compares the character count of extracted documents against `char_count_threshold`. Used by `DocumentExtractor` after each strategy attempt.

---

## Data Models — [models.py](../../app/services/document_loading/models.py)

### `ExtractionResult`

Output of processing one file.

| Field | Type | Description |
|---|---|---|
| `file_path` | `Path` | Path to the source file. |
| `documents` | `List[Document]` | Extracted LangChain `Document` objects. |
| `metrics` | `ExtractionMetrics` | Per-file stats. |
| `used_fallback` | `bool` | Whether a fallback strategy was used. |

---

### `ExtractionMetrics`

| Field | Type | Description |
|---|---|---|
| `file_path` | `Path` | |
| `file_type` | `FileType` | |
| `file_size_bytes` | `int` | |
| `extraction_strategy` | `ExtractionStrategy` | Which strategy succeeded. |
| `char_count` | `int` | Total characters extracted. |
| `page_count` | `int` | Number of documents (roughly pages). |
| `extraction_time_seconds` | `float` | Wall time for the full extraction. |
| `error_message` | `Optional[str]` | Set if strategy is `FAILED`. |
| `document_count` | `int` | Same as `page_count`. |

---

### `LoadingTask`

Input descriptor for a loading run.

| Field | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | auto UUID | |
| `directory_path` | `Path` | `./documents` | Root directory to scan. |
| `file_pattern` | `Optional[str]` | `None` | Glob pattern (e.g. `**/*.pdf`). If `None`, all supported files are included. |
| `file_types` | `Optional[List[FileType]]` | `None` | Restrict to specific types. |
| `status` | `LoadingStatus` | `PENDING` | Updated throughout the run. |

---

### `LoadingSummary`

Aggregate stats produced at the end of a task.

| Field | Type | Description |
|---|---|---|
| `total_files` | `int` | Files discovered. |
| `successfully_extracted` | `int` | Files that produced at least one document. |
| `failed_files` | `int` | Files that produced no documents. |
| `total_documents` | `int` | Sum of all documents across all files. |
| `total_chars` | `int` | Sum of all character counts. |
| `total_time_seconds` | `float` | Sum of individual extraction times. |
| `files_using_fallback` | `int` | Files where a fallback strategy was used. |
| `extraction_strategies` | `Dict[str, int]` | Count per strategy (e.g. `{"standard": 18, "docling": 2}`). |
| `success_rate` | `float` | `successfully_extracted / total_files`. |
| `avg_extraction_time` | `float` | `total_time_seconds / total_files`. |
| `avg_chars_per_file` | `float` | `total_chars / total_files`. |

---

### Enums

**`FileType`**
`PDF`, `MARKDOWN`, `DOCX`, `TXT`, `PNG`, `JPG`, `JPEG`, `UNKNOWN`

**`ExtractionStrategy`**
`STANDARD`, `DOCLING`, `OCR`, `VLM`, `FAILED`

**`LoadingStatus`**
`PENDING`, `VALIDATING`, `PROCESSING`, `COMPLETED`, `FAILED`

---

## Configuration — [config.py](../../app/services/document_loading/config.py)

Config is loaded from environment variables via `DocumentLoadingConfig.from_env()`. A YAML file at [settings/document-loading-config.yaml](../../settings/document-loading-config.yaml) shows all available options.

```yaml
environment: production

extraction:
  char_count_threshold: 500       # Min chars before triggering fallback
  max_concurrent_files: 5         # Max files processed simultaneously
  max_workers: 10                 # ThreadPoolExecutor worker count
  extraction_timeout_seconds: 300.0
  enable_ocr: true
  enable_vlm: false
  vlm_model: gpt-4-vision-preview
  max_file_size_mb: 500
  min_file_size_bytes: 100

deduplication:
  enabled: true
  method: content_hash            # content_hash or file_hash
  threshold: 0.95

documents_root: ./documents
temp_dir: /tmp/cortex-doc-loading

enable_deduplication: true
enable_validation: true
```

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `CORTEX_ENV` | `production` | Runtime environment |
| `DL_CHAR_THRESHOLD` | `500` | Fallback trigger threshold |
| `DL_MAX_CONCURRENT` | `5` | Max concurrent files |
| `DL_ENABLE_OCR` | `true` | Enable OCR fallback |
| `DL_ENABLE_VLM` | `false` | Enable VLM fallback |

---

## Exceptions — [exceptions.py](../../app/services/document_loading/exceptions.py)

All exceptions extend `DocumentLoadingException`, which carries a `message`, `error_code`, and optional `details` dict.

| Exception | Error Code | When raised |
|---|---|---|
| `ValidationException` | `VALIDATION_ERROR` | Directory or file fails validation |
| `ExtractionException` | `EXTRACTION_ERROR` | Standard extraction raises an unhandled error |
| `FileDiscoveryException` | `FILE_DISCOVERY_ERROR` | File discovery step fails |

---

## Usage

```python
import asyncio
from pathlib import Path
from app.services.document_loading.service import DocumentLoadingService
from app.services.document_loading.models import FileType

async def main():
    with DocumentLoadingService() as service:
        results, summary = await service.load_directory(
            directory_path=Path("./documents"),
            file_types=[FileType.PDF, FileType.MARKDOWN],
        )

        print(f"Processed {summary.total_files} files")
        print(f"Success rate: {summary.success_rate:.0%}")
        print(f"Strategies used: {summary.extraction_strategies}")

        for result in results:
            for doc in result.documents:
                print(doc.page_content[:200])

asyncio.run(main())
```

Using `LoadingTask` directly:

```python
from app.services.document_loading.models import LoadingTask

task = LoadingTask(
    directory_path=Path("./documents"),
    file_pattern="**/*.pdf",
)

results, summary = await service.load_with_task(task, max_concurrent=10)
```
