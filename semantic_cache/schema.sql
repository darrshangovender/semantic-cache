-- Run this once against your Postgres database before using semantic-cache.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_cache_entries (
    id              BIGSERIAL PRIMARY KEY,
    namespace       TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    embedding       vector(1536),
    response        TEXT NOT NULL,
    model           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_hit_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hit_count       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(namespace, prompt_hash)
);

CREATE INDEX IF NOT EXISTS idx_sc_ns_hash ON semantic_cache_entries (namespace, prompt_hash);
CREATE INDEX IF NOT EXISTS idx_sc_embedding ON semantic_cache_entries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_sc_created_at ON semantic_cache_entries (created_at);