# Observability

**Location:** [observability/](../observability/)

Structured logging with dual output: JSON to rotating files and a human-readable format to the console. Configured via `settings/logging_config.yaml`.

---

## File Structure

```
observability/
└── logging/
    ├── __init__.py
    ├── config.py    # LoggingConfig — loads logging_config.yaml
    └── logger.py    # Logger, get_logger, configure
```

---

## Architecture

- **`JSONFormatter`** — serialises log records as JSON (timestamp, level, logger name, message, module, function, line, optional exception and extra fields).
- **`ConsoleFormatter`** — compact `[LEVEL] message` format for terminal output.
- **`Logger`** — wraps Python's `logging.Logger`; supports structured context fields via `set_context`.
- **`get_logger(name)`** — module-level factory; initialises the logging system on first call if `configure()` has not been called.

---

## Configuration

In `settings/logging_config.yaml`:

```yaml
level: INFO
console_enabled: true
file_enabled: true
log_file: logs/cortex.log
max_file_size_mb: 10
backup_count: 5
```

---

## Key Functions

### `get_logger(name: str) -> Logger`

Returns a `Logger` instance for the given name. Lazy-initialises configuration from `logging_config.yaml` on first call.

```python
from observability.logging.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing document", doc_id="abc123")
```

### `configure(config: LoggingConfig)`

Explicitly sets the logging configuration before any `get_logger` call. Call once at application startup.

```python
from observability.logging.logger import configure
from observability.logging.config import get_logging_config

configure(get_logging_config())
```

---

## `Logger` Methods

| Method | Description |
|---|---|
| `debug(message, **kwargs)` | Debug-level log |
| `info(message, **kwargs)` | Info-level log |
| `warning(message, **kwargs)` | Warning-level log |
| `error(message, **kwargs)` | Error-level log |
| `critical(message, **kwargs)` | Critical-level log |
| `set_context(**kwargs)` | Attach key-value pairs to all subsequent log records |
| `clear_context()` | Remove all stored context fields |

Extra `**kwargs` and context fields are merged into the `extra_data` field of the JSON log output.

---

## Output Formats

**Console:**
```
[INFO] Processing document
[ERROR] Embedding failed: connection timeout
```

**JSON (log file):**
```json
{
  "timestamp": "2026-04-05T14:23:01.123456",
  "level": "INFO",
  "logger": "rag.ingestion.service",
  "message": "Processing document",
  "module": "service",
  "function": "load_file",
  "line": 42,
  "doc_id": "abc123"
}
```

---

## Usage

```python
from observability.logging.logger import get_logger

logger = get_logger(__name__)

# Simple log
logger.info("Starting ingestion")

# Structured fields
logger.info("Chunk inserted", chunk_id="c-001", tokens=128)

# Request-scoped context
logger.set_context(request_id="req-xyz", user_id="user-123")
logger.info("Search started")      # includes request_id and user_id
logger.info("Search complete")     # includes request_id and user_id
logger.clear_context()
```
