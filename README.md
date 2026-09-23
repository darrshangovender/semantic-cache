# semantic-cache — cache LLM responses by meaning, not by string

[![tests](https://github.com/darrshangovender/semantic-cache/actions/workflows/tests.yml/badge.svg)](https://github.com/darrshangovender/semantic-cache/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![pgvector](https://img.shields.io/badge/pgvector-Postgres-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)

> A two-tier cache for LLM responses backed by Postgres and pgvector. SHA-256 for verbatim repeats, embedding kNN above a cosine threshold for paraphrases, TTL enforced on read, and per-namespace isolation so unrelated apps don't poison each other.

**Why this exists.** Exact-string caches catch a tiny fraction of repeat queries, because users phrase the same question differently every time — "how do I cancel?", "I want to cancel my subscription", "cancellation process". Those are one question and three cache keys. Embedding the prompt and looking up by similarity catches the rest, and on support-shaped workloads the repeats are most of the traffic.

---

## Quick start

Create the schema first — the cache assumes the table and the pgvector extension exist:

```bash
psql "$DATABASE_URL" -f semantic_cache/schema.sql
pip install -e ".[dev]"
```

```python
from semantic_cache import SemanticCache, OpenAIEmbedder

cache = SemanticCache(
    "postgresql://user:pass@localhost/cache_db",
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    similarity_threshold=0.95,
    ttl_hours=168,
)

@cache.cached(model="claude-sonnet-4-5", namespace="support-bot")
def ask_support(question: str) -> str:
    return call_your_llm(question)

ask_support("how do I cancel my subscription?")   # miss → LLM → stored
ask_support("how do I cancel my subscription?")   # exact-hash hit
ask_support("how can I cancel my account?")       # paraphrase → similarity hit
```

Manual form, if you'd rather not decorate:

```python
hit = cache.lookup("support-bot", question)
if hit is None:
    hit = call_your_llm(question)
    cache.store("support-bot", question, hit, "claude-sonnet-4-5")

cache.invalidate("support-bot")     # returns rows deleted
```

## How it works

```mermaid
flowchart LR
    P[prompt] --> H[sha256 lookup]
    H -->|hit| RH[record hit]
    H -->|miss| E[embed prompt]
    E --> K["pgvector kNN, top-1"]
    K -->|cosine ≥ threshold| RH
    K -->|below| L[call the LLM]
    L --> S[embed again + upsert]
    RH --> R[return cached]
    S --> R
```

1. The wrapped function is called with the prompt as its first argument.
2. `lookup()` hashes the prompt and queries on `(namespace, prompt_hash)` within the TTL window.
3. A hash hit bumps `hit_count` and `last_hit_at` and returns immediately — no embedding call at all.
4. On a miss, the prompt is embedded once.
5. A pgvector kNN query takes the top-1 neighbour within the namespace and TTL window.
6. If `1 - distance` clears the threshold, that entry is a hit.
7. Otherwise the real function runs and `store()` upserts the result on `(namespace, prompt_hash)`.

## Choosing a threshold

Cosine similarity above roughly 0.95 between two questions is usually paraphrase territory; below 0.90 drifts into "related but different". The right value depends entirely on what a wrong near-hit costs you:

| Workload | Threshold | Because |
|---|---|---|
| Medical, legal, financial | 0.98+ | A confidently wrong cached answer is the whole risk |
| Customer support, FAQ | 0.95 | Questions cluster tightly; near-misses are recoverable |
| Creative, brainstorming | 0.90 | More hits, and a slightly-off answer is fine |

The threshold is an attribute of the cache instance, not of the namespace — so different thresholds means different `SemanticCache` objects.

## Design decisions

| Decision | Why |
|---|---|
| **Hash tier before embedding tier** | Verbatim repeats are common and cost nothing to catch. Embedding every lookup would pay an API call to discover something a hash already knew. |
| **TTL enforced on read, not by a sweeper** | Fewer moving parts, no background job to deploy or monitor. The cost of that choice is in the limitations below, and it is real. |
| **Namespaces, not one global keyspace** | Two apps sharing a cache will eventually return one's answer to the other's user. Isolation is cheaper than the incident. |
| **A pluggable `Embedder` protocol** | You should be able to swap in a local model and stop paying per lookup. Only the OpenAI implementation ships. |
| **`hit_count` and `last_hit_at` on every row** | Without them you cannot tell a cache that is working from a table that is merely large. |

## Limitations

- **Two embedding API calls per cache miss.** `lookup()` embeds the prompt, then `store()` embeds the identical prompt again — the vector is never threaded from one to the other. Every miss pays double the embedding cost and latency. This is the cheapest fix in the repo and has not been made.
- **A new Postgres connection per operation, no pooling.** A cache miss opens two TCP connections. Any "sub-millisecond hit" intuition is wrong for that reason — connection setup dominates the hash tier entirely.
- **TTL is enforced against `created_at`, which the upsert resets.** Re-storing an entry silently extends its lifetime, and because expiry is read-side only, expired rows are never deleted. The table grows unbounded and the ivfflat scan degrades over time. You need your own sweeper.
- **Embedding dimension is never validated against the schema.** The schema hardcodes `vector(1536)` and the `Embedder` protocol's `dimension` attribute is read nowhere in the codebase. A custom embedder of another size fails at INSERT time with a raw Postgres error rather than at construction.
- **The ANN index fights the namespace predicate.** The kNN query filters on namespace and TTL while ordering by vector distance. ivfflat probes a fixed number of lists *before* the filter applies, so with many namespaces the top-1 can be a non-match and the query returns nothing even when a valid near-hit exists in the table.
- **No request coalescing.** Two simultaneous identical prompts both miss and both hit the LLM. Add a coalescing layer if that matters at your scale.
- **No content-based invalidation.** If your source data changes and a cached answer goes stale, you have to bust it yourself — by namespace, or by truncating.
- **No streaming support.** The cached unit is a complete string.
- **The published savings figures have been removed.** This README previously claimed 30–80% LLM cost savings, and the source docstring still claims tier-specific hit rates. There is no benchmark, no results file, and no timing test in this repo — nothing here measures a hit rate.
- **Zero test coverage of the actual cache behaviour.** All three tests run without a database: threshold validation, threshold storage, and that the decorator returns a callable. Not one line of SQL in the project is executed by any test.

## Project layout

```
semantic-cache/
├── semantic_cache/
│   ├── cache.py        # SemanticCache: lookup · store · cached · invalidate
│   ├── embeddings.py   # Embedder protocol + OpenAIEmbedder
│   └── schema.sql      # table + ivfflat index (run this first)
└── tests/              # 3 tests
```

The schema is one table keyed `UNIQUE(namespace, prompt_hash)`, with a `vector(1536)` embedding column, an ivfflat cosine index, and hit instrumentation.

## Tests

```bash
pytest tests/ -q         # 3 tests, no database required
```

Honest state: see the last limitation. The first thing this repo needs is a `pytest` fixture spinning up Postgres with pgvector so `lookup`, `store` and the TTL boundary are actually exercised — every bug listed above lives in code no test runs.

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za) · Durban, South Africa
