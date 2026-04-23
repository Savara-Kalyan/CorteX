# CorteX — Multi-Agent RAG System

A multi-agent system for querying internal documents. LangGraph supervisor routes queries to three specialist agents. The knowledge agent runs a 7-layer retrieval pipeline. All I/O is async (Redis via `redis.asyncio`, PostgreSQL via `psycopg`).

---

<!-- eval-results-start -->
## Measured Results
*Last evaluated: 2026-04-20 03:54 IST — golden dataset n=20*

### Retrieval Quality (word-overlap scoring)

| Metric | Target | Measured |
|---|---|---|
| **P@5** | >= 0.25 | **0.4300** |
| **MRR** | >= 0.50 | **0.8000** |
| **NDCG@5** | >= 0.50 | **0.6596** |

### RAGAS Scores (LLM-as-judge)

| Metric | Target | Score |
|---|---|---|
| **Faithfulness** | >= 0.60 | **0.7500** |
| **Answer Relevancy** | >= 0.60 | **0.5757** |
| **Context Precision** | >= 0.50 | **0.8717** |
| **Answer Correctness** | >= 0.50 | **0.5473** |
<!-- eval-results-end -->

---

## Success Metrics

### Retrieval Quality

| Metric | Target | Description |
|---|---|---|
| **P@5** | **>= 0.70** | Minimum pass gate |
| **MRR** | **>= 0.65** | First relevant result ranking |
| **NDCG@5** | **>= 0.70** | Ranking quality |
| **P@5 (stretch)** | **0.84** | 7-layer pipeline with hybrid search |

### Reliability Targets

| Target | Value | Description |
|---|---|---|
| **Query success rate** | **>= 95%** | No unhandled exceptions |
| **Max cost per query** | **<= $0.05** | Budget enforcement via CostTracker |
| **Automated tests** | **18+** | All passing |
| **Fallback depth** | **3-tier** | For all external calls |

### Memory Correctness

| Test Scenario | Expected Result |
|---|---|
| Server restart mid-session | Entity memory restored from PostgreSQL |
| Turn 8: "What was my order number?" (told in turn 1) | Correct entity retrieved |
| New session, same user | Long-term entities loaded from PostgreSQL |
| Standard employee queries financial report | Access denied — RBAC filters doc |

---

## Architecture

```
                            ┌─────────────────────────────────────────┐
                            │         Cortex Supervisor Agent          │
                            │   LangGraph state machine (W1 + W2)     │
                            │  Rate limit → Budget check → Route       │
                            └──────────────┬──────────────────────────┘
                                           │
               ┌───────────────────────────┼──────────────────────────┐
               ▼                           ▼                          ▼
    ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
    │  Knowledge Agent │       │  Research Agent  │       │  Action Agent    │
    │  (W3 — 7 layers) │       │  (W4 — web)      │       │  (W4 — tools)    │
    └────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
             │                          │                           │
             ▼                          ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
    │  RAG Pipeline    │       │  web_search      │       │  create_ticket   │
    │  L1 Ingestion    │       │  DDG → SerpAPI   │       │  get_calendar    │
    │  L2 Chunking     │       │  → cache         │       │  generate_report │
    │  L3 Embeddings   │       └──────────────────┘       └──────────────────┘
    │  L4 VectorStore  │
    │  L5 QueryUnder.  │              Memory Layer
    │  L6 AccessCtrl   │       ┌──────────────────┐
    │  L7 HybridSearch │       │  Redis (session) │  ← 6-turn sliding window
    └──────────────────┘       │  Postgres (long) │  ← entity facts, persist
                               └──────────────────┘

                               Reliability Layer
                        ┌─────────────────────────────┐
                        │ RateLimiter  CostTracker     │
                        │ FallbackChain  CircuitBreaker│
                        │ RetryExecutor  (exponential) │
                        └─────────────────────────────┘
```

---

## Project Structure

```
cortex/
├── main.py                      ← Entry point: --demo, --ingest, REPL
├── docker-compose.yml           ← pgvector + redis + prometheus + grafana
├── requirements.txt
├── config.yaml                  ← Switch LLM / embedding / vector store
├── .env.example
│
├── agents/
│   ├── supervisor.py            ← LangGraph state machine
│   ├── knowledge_agent.py       ← RAG specialist
│   ├── research_agent.py        ← Web research + rate limiting
│   └── action_agent.py          ← Tools: ticket, calendar, report
│
├── rag/
│   ├── pipeline.py              ← Orchestrates all 7 layers
│   ├── ingestion/               ← L1: PDF (pdfplumber) + markdown
│   ├── chunking.py              ← L2: RecursiveCharacterTextSplitter
│   ├── embeddings.py            ← L3: OpenAI / Anthropic
│   ├── vector_store.py          ← L4: PGVector / Qdrant
│   ├── query_understanding.py   ← L5: reformulation + expansion + intent
│   ├── access_control.py        ← L6: RBAC tier filter
│   └── hybrid_search.py         ← L7: BM25 + vector + RRF
│
├── memory/
│   ├── session_memory.py        ← Redis sliding window (6 turns, async)
│   └── entity_store.py          ← PostgreSQL long-term entities (async)
│
├── tools/
│   ├── web_search.py            ← DuckDuckGo + SerpAPI + cache
│   ├── ticketing.py             ← Ticket creation + queue fallback
│   ├── calendar.py              ← Availability check + auth pattern
│   └── report_generator.py      ← Budget-enforced report tool
│
├── reliability/
│   ├── rate_limiter.py          ← Sliding window (async Redis) + retry_with_backoff
│   ├── fallback.py              ← FallbackChain, RetryExecutor, CircuitBreaker
│   └── cost_tracker.py          ← Per-query budget enforcement (async Redis)
│
├── prompts/
│   ├── supervisor/
│   │   ├── v1.0.0.yaml          ← Production version
│   │   └── v1.1.0.yaml          ← A/B test variant (confidence calibration)
│   └── agents/
│       ├── knowledge_agent/v1.0.0.yaml
│       └── research_agent/v1.0.0.yaml
│
├── observability/
│   ├── logger.py                ← Structured JSON + console logger
│   └── metrics.py               ← Prometheus metrics + start_metrics_server
│
├── monitoring/
│   ├── prometheus.yml           ← Scrape config (cortex-app:8001)
│   └── grafana/
│       ├── provisioning/        ← Auto-wired datasource + dashboard loader
│       └── dashboards/          ← cortex.json (8 panels)
│
├── data/
│   └── sample_knowledge_base/   ← 10 docs: HR, IT, Finance, Engineering, Culture
│
├── org-docs/                    ← Source documents used for ingestion
│
└── tests/
    ├── unit/                    ← tools, rate_limiter, fallback, cost_tracker, rag_pipeline
    ├── integration/             ← supervisor_routing, knowledge_agent, full_pipeline
    └── evaluation/
        ├── golden_dataset.json  ← 30 hand-authored QA pairs from org-docs
        └── test_ragas_eval.py   ← RAGAS evaluation (faithfulness, relevancy, precision)
```

---

## Quick Start

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, DATABASE_URL, etc.

# 4. Ingest documents
python main.py --ingest

# 5. Run demo queries
python main.py --demo

# 6. Interactive REPL
python main.py
```

Grafana dashboard: http://localhost:3000 (admin / cortex)
Prometheus metrics: http://localhost:9090

---

## Configuration

Edit [`config.yaml`](config.yaml) to switch providers:

```yaml
LLM:
  PROVIDER: openai
  MODEL: gpt-4.1-mini

EMBEDDINGS:
  PROVIDER: openai
  MODEL: text-embedding-3-small

VECTOR_STORE:
  PROVIDER: pgvector
  CONNECTION_STRING: "postgresql://cortex:cortex@localhost:5432/cortexdb"

QUERY_UNDERSTANDING:
  MODEL: gpt-4.1-nano
  TEMPERATURE: 0.0
```

### User Tiers (RBAC)

| Tier | Access |
|---|---|
| `public` | Culture documents only |
| `internal` | Engineering, IT documents |
| `confidential` | HR, Finance documents |
| `restricted` | Full access |

---

## Running Tests

```bash
# Unit tests (no API key needed — fully mocked)
pytest tests/unit/ -v

# Integration tests (mocked LLM + graph wiring)
pytest tests/integration/ -v

# RAGAS evaluation (requires OPENAI_API_KEY + running infrastructure)
pytest tests/evaluation/ -v

# Full suite with coverage
pytest --cov=. --cov-report=term-missing
```

---

## The 5 Tools

| Tool | Pattern | Failure Mode |
|---|---|---|
| `web_search` | 3-tier fallback (DDG → SerpAPI → cache) | Tier exhausted → error dict |
| `create_support_ticket` | Input validation + API → queue fallback | API down → in-memory queue + queue ID |
| `get_team_calendar` | Auth token pattern + graceful failure | Missing token → `auth_error` + retry guidance |
| `generate_report` | Token budget enforcement | Over budget → summary instead of full report |
| `knowledge_base_search` | Proper tool schema + error feedback to LLM | No docs found → LLM-recoverable message |

---

## Reliability Layer

| Component | Pattern |
|---|---|
| `RateLimiter` | Sliding-window (async Redis, in-memory fallback) |
| `CostTracker` | Per-user daily budget enforcement (async Redis) |
| `FallbackChain` | Tries tools in order; returns first success |
| `RetryExecutor` | Retries on timeout up to N times |
| `CircuitBreaker` | Opens after N failures; HALF_OPEN probe; auto-recovers |
| `retry_with_backoff` | Decorator with exponential backoff (sync and async) |

---

## RAG Pipeline Detail

```
User Query
    │
    ▼  L5 — Query Understanding
    │    Reformulate → Expand (3 variants) → Classify intent
    │    Returns: reformulated, all_queries, intent, answerable
    │
    ▼  L6 + L7 — Hybrid Search + Access Control
    │    Vector search  ──┐
    │    BM25 search    ──┤──▶ RRF fusion ──▶ RBAC tier filter
    │                     │
    │    (concurrent via asyncio.gather)
    │
    ▼  Answer Synthesis
         LLM with retrieved context
         Every claim cited: [Source: <document>, <section>]
```

**Ingestion path (L1–L4):**

```
Directory
    │
    ▼  L1 — Ingestion
    │    pdfplumber for PDF, Path.read_text() for markdown
    │    MarkdownHeaderTextSplitter → one Document per section
    │
    ▼  L2 — Chunking
    │    RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
    │
    ▼  L3 — Embeddings
    │    OpenAI text-embedding-3-small (configurable)
    │
    ▼  L4 — Vector Store
         PGVector (default) or Qdrant
         pgvector HNSW index for fast ANN search
```
