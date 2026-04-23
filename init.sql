-- Cortex database schema
-- Applied automatically by postgres on first container start.

CREATE EXTENSION IF NOT EXISTS vector;

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

-- Indexes for fast filtered vector search
CREATE INDEX IF NOT EXISTS documents_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_metadata_gin
    ON documents USING gin (metadata);

CREATE INDEX IF NOT EXISTS documents_domain_expr_idx
    ON documents ((metadata->>'domain'));

CREATE INDEX IF NOT EXISTS documents_access_level_idx
    ON documents (access_level);
