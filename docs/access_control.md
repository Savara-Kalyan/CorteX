# Access Control

**Location:** [rag/access_control/](../rag/access_control/)

Domain-filtered semantic search with role-based access control (RBAC). Enforces which users can query which knowledge domains, creates the required database indexes, and runs parallel domain fan-outs with cosine-similarity re-ranking.

---

## File Structure

```
rag/access_control/
├── __init__.py
└── service.py   # AccessPolicy, IndexManager, AccessControlService
```

---

## Architecture

Three components work together:

- **`AccessPolicy`** — maps domains to required access levels; determines which domains a given user level can read.
- **`IndexManager`** — idempotently creates all PostgreSQL indexes needed for fast domain-filtered vector search.
- **`AccessControlService`** — async facade that embeds queries, enforces policy, and runs filtered searches.

### Access Level Hierarchy (least → most privileged)

```
public < internal < confidential < restricted
```

### Default Domain → Minimum Access Level

| Domain | Required Level |
|---|---|
| `culture` | `public` |
| `engineering` | `internal` |
| `general` | `internal` |
| `hr` | `confidential` |

---

## Database Indexes

`IndexManager` creates four indexes (all `IF NOT EXISTS`):

| Index | Type | Purpose |
|---|---|---|
| `documents_hnsw_idx` | HNSW (cosine) | Fast approximate nearest-neighbour search |
| `documents_metadata_gin` | GIN | Fast JSONB key/value lookups |
| `documents_domain_expr_idx` | B-tree | Fast domain equality filter |
| `documents_access_level_idx` | B-tree | Fast access level filter |

---

## Key Class

### `AccessControlService` — [service.py](../rag/access_control/service.py)

```python
from rag.access_control.service import AccessControlService
```

All methods are async. Safe to instantiate once and reuse.

#### `ensure_indexes() -> None`

Creates all required database indexes (idempotent).

#### `search_by_domain(query, domain, user_access_level, top_k) -> list[dict]`

Semantic search within a single domain after enforcing access control.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Natural-language query |
| `domain` | `str` | — | Target domain (`hr`, `engineering`, `culture`, `general`) |
| `user_access_level` | `str` | `"internal"` | Caller's access level |
| `top_k` | `int` | `5` | Maximum results |

Returns a list of dicts with keys: `id`, `content`, `access_level`, `source_file`, `domain`, `similarity`.

Raises `PermissionDeniedError` if the user's level is insufficient.

#### `search_multi_domain(query, domains, user_access_level, top_k_per_domain) -> dict[str, list[dict]]`

Runs the same query against multiple domains concurrently. Inaccessible domains are silently skipped (logged at WARNING). Individual domain failures are logged at ERROR and return an empty list — they do not abort the full search.

Returns a mapping of `domain → result list`.

#### `search_accessible(query, user_access_level, top_k) -> list[dict]`

Searches all domains accessible to the user, then merges and re-ranks results by cosine similarity.

---

## Exceptions

| Exception | Raised when |
|---|---|
| `PermissionDeniedError` | User's access level is below the domain's required level |
| `SearchError` | Embedding generation or query execution fails |
| `DatabaseConnectionError` | PostgreSQL connection fails |
| `IndexCreationError` | An index DDL statement fails |

---

## Usage

```python
from rag.access_control.service import AccessControlService

svc = AccessControlService()

# One-time index setup (idempotent)
await svc.ensure_indexes()

# Single-domain search
results = await svc.search_by_domain(
    query="parental leave policy",
    domain="hr",
    user_access_level="confidential",
)

# All accessible domains merged and re-ranked
results = await svc.search_accessible(
    query="career growth",
    user_access_level="internal",
)

for r in results:
    print(f"[{r['similarity']:.3f}] [{r['domain']}] {r['content'][:80]}")
```

### Custom Policy

```python
from rag.access_control.service import AccessPolicy, AccessControlService

policy = AccessPolicy(
    domain_access_map={"finance": "restricted", "public-docs": "public"},
    access_levels=["public", "internal", "restricted"],
)
svc = AccessControlService(policy=policy)
```
