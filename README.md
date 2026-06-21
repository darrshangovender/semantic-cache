<div align="center">

# semantic-cache — production cache for LLM responses, keyed by meaning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![pgvector](https://img.shields.io/badge/pgvector-0.7+-4169E1)](https://github.com/pgvector/pgvector)
[![Status](https://img.shields.io/badge/Status-Working%20code-blue)](#)

</div>

---

> A semantic cache for LLM responses. Hashes the prompt for exact-match hits, embeds for near-match hits (cosine > threshold), returns cached response or calls through. Saves 30-80% of LLM cost on chatbot-style workloads where users ask the same question in different words.

**Why this exists.** Exact-string caches catch a tiny fraction of repeat queries because users phrase things differently every time ("how do I cancel?" vs "I want to cancel my subscription" vs "cancellation process"). Embedding the prompt and looking up by similarity catches the rest. Cost savings on a real chatbot workload are typically 30-80%.

---

## How it works

```
incoming prompt
   │
   ▼
┌──────────────┐
│ Exact hash   │ ── SHA-256 lookup. Sub-millisecond. Catches verbatim repeats.
└──────┬───────┘
       │ miss
       ▼
┌──────────────┐
│ Embed prompt │ ── one embedding call per cache miss
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ pgvector kNN │ ── cosine similarity, top-1, threshold (default 0.95)
└──────┬───────┘
       │ hit              │ miss
       ▼                  ▼
   return cached        call LLM, store
```

## Usage

```python
from semantic_cache import SemanticCache
from anthropic import Anthropic

cache = SemanticCache(
    dsn="postgresql://user:pass@localhost/cache_db",
    similarity_threshold=0.95,
    ttl_hours=168,  # 1 week
)
client = Anthropic()

@cache.cached(model="claude-sonnet-4-5", namespace="support-bot")
def ask_support(question: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text

# First call: hits LLM, stores answer + embedding.
ask_support("how do I cancel my subscription?")

# Second call: exact-match hit. Sub-millisecond.
ask_support("how do I cancel my subscription?")

# Third call: paraphrase. Embedding similarity > 0.95 → near-match hit.
ask_support("how can I cancel my account?")
```

## Why a threshold of 0.95 by default

Empirically, sentence-transformer cosine similarity above 0.95 between two questions is almost always paraphrase territory; above 0.90 starts to drift into "related but different" territory. The threshold is per-namespace because support questions tend to cluster tightly while creative-prompt workloads do not.

For workloads where wrong-answer-on-near-hit is catastrophic (medical, legal, financial), raise to 0.98+. For workloads where it's annoying but recoverable (creative, brainstorming), drop to 0.90 for more hits.

## What the cache does NOT do

- **No automatic invalidation by content change** — if your source data updates and the cached answer becomes stale, you have to bust the cache (by namespace, by tag, or by truncating).
- **No request-deduplication for in-flight calls** — two simultaneous identical prompts will both cache-miss and both hit the LLM. Add a request-coalescing layer if that matters at your scale.
- **No streaming response support** — the cached unit is the full string. Streaming is a separate problem.

## Repo structure

```
.
├── semantic_cache/
│   ├── __init__.py
│   ├── cache.py        # main SemanticCache class + @cached decorator
│   ├── embeddings.py   # pluggable embedder (OpenAI default)
│   └── schema.sql      # Postgres + pgvector schema
├── tests/
│   └── test_cache.py
└── pyproject.toml
```

## Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE semantic_cache_entries (
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

CREATE INDEX ON semantic_cache_entries (namespace, prompt_hash);
CREATE INDEX ON semantic_cache_entries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Status

- [x] SemanticCache class with cached() decorator
- [x] Pluggable embedder (OpenAI default)
- [x] Exact-hash + near-match lookup
- [x] TTL enforcement on read
- [x] Per-namespace isolation
- [x] Hit-count instrumentation
- [ ] Redis backend (alternative to Postgres for low-latency cases)
- [ ] Request coalescing for in-flight identical prompts

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)