"""
Root conftest — real-service fixtures and skip markers.

All infrastructure-dependent tests import markers from here:

    from tests.conftest import requires_pg, requires_redis, requires_llm

Services are expected at:
  PostgreSQL  localhost:5432  (or PGVECTOR_URL env var)
  Redis       localhost:6379  (or REDIS_URL env var)
  OpenAI      OPENAI_API_KEY env var

Start infrastructure with:
    docker-compose up postgres redis -d
"""

from __future__ import annotations

import asyncio
import os

import psycopg
import pytest
import redis as redis_lib

# ---------------------------------------------------------------------------
# Connection URLs — override via env vars
# ---------------------------------------------------------------------------

PG_DSN = os.getenv("PGVECTOR_URL", "postgresql://cortex:cortex@localhost:5432/cortexdb")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Availability probes — called once at collection time
# ---------------------------------------------------------------------------

def _pg_available() -> bool:
    try:
        conn = psycopg.connect(PG_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


def _redis_available() -> bool:
    try:
        r = redis_lib.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        return True
    except Exception:
        return False


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


_PG_UP = _pg_available()
_REDIS_UP = _redis_available()
_LLM_UP = _llm_available()

# ---------------------------------------------------------------------------
# Pytest skip markers
# ---------------------------------------------------------------------------

requires_pg = pytest.mark.skipif(
    not _PG_UP,
    reason="PostgreSQL/pgvector not reachable — run: docker-compose up postgres -d",
)
requires_redis = pytest.mark.skipif(
    not _REDIS_UP,
    reason="Redis not reachable — run: docker-compose up redis -d",
)
requires_llm = pytest.mark.skipif(
    not _LLM_UP,
    reason="OPENAI_API_KEY not set",
)
requires_all = pytest.mark.skipif(
    not (_PG_UP and _REDIS_UP and _LLM_UP),
    reason="Full infrastructure required: postgres + redis + OPENAI_API_KEY",
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pg_dsn() -> str:
    return PG_DSN


@pytest.fixture(scope="session")
def redis_url() -> str:
    return REDIS_URL


@pytest.fixture(scope="session")
def sync_pg_conn():
    """Synchronous psycopg connection, session-scoped."""
    if not _PG_UP:
        pytest.skip("PostgreSQL not available")
    conn = psycopg.connect(PG_DSN)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def redis_client():
    """Real Redis client, session-scoped."""
    if not _REDIS_UP:
        pytest.skip("Redis not available")
    client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    yield client
    client.close()


@pytest.fixture(scope="session")
def llm():
    """Real ChatOpenAI client, session-scoped."""
    if not _LLM_UP:
        pytest.skip("OPENAI_API_KEY not set")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-5.4-nano-2026-03-17", temperature=0)


@pytest.fixture(scope="session")
def embed_service():
    """Real OpenAI embedding service, session-scoped."""
    if not _LLM_UP:
        pytest.skip("OPENAI_API_KEY not set")
    from rag.embeddings import EmbeddingService
    return EmbeddingService()


# ---------------------------------------------------------------------------
# DB schema setup — runs once per session if postgres is available
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_db_schema(sync_pg_conn):
    """Create the documents table and extensions if they don't exist."""
    if not _PG_UP:
        return
    cur = sync_pg_conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id                    BIGSERIAL PRIMARY KEY,
            content               TEXT        NOT NULL,
            embedding             vector(1536),
            source_file           TEXT,
            page_number           INTEGER,
            chunk_index           INTEGER     NOT NULL DEFAULT 0,
            total_chunks          INTEGER     NOT NULL DEFAULT 1,
            doc_hash              TEXT,
            access_level          TEXT        NOT NULL DEFAULT 'internal',
            created_by            TEXT,
            doc_type              TEXT        NOT NULL DEFAULT 'unknown',
            chunk_type            TEXT        NOT NULL DEFAULT 'text',
            extraction_method     TEXT        NOT NULL DEFAULT 'docling',
            extraction_confidence FLOAT       NOT NULL DEFAULT 0.95,
            chunk_length          INTEGER,
            metadata              JSONB       NOT NULL DEFAULT '{}',
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    cur.close()


# ---------------------------------------------------------------------------
# Test-data cleanup — removes rows written by tests
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_test_docs(sync_pg_conn):
    """
    Delete all test-authored documents after each test.
    Tests that insert rows should tag them with source_file='test:*'.
    """
    yield
    cur = sync_pg_conn.cursor()
    cur.execute("DELETE FROM documents WHERE source_file LIKE 'test:%';")
    cur.close()
