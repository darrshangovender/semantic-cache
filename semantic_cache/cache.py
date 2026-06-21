"""SemanticCache — main API.

Design choices:
  - prompt_hash for exact-match (sub-ms, hits ~30% of repeats in support workloads)
  - pgvector kNN for near-match (cosine > threshold, hits another 30-50%)
  - TTL enforced on read, not via background sweeper — fewer moving parts.
  - per-namespace isolation so unrelated apps don't poison each other.
"""

from __future__ import annotations

import functools
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Callable

import psycopg
from psycopg.types.json import Json

from .embeddings import Embedder, OpenAIEmbedder


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _to_pgvector(v: list[float]) -> str:
    """psycopg passes vectors as strings of the form '[1.0,2.0,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class CacheMiss(Exception):
    pass


class SemanticCache:
    def __init__(
        self,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 168,
    ):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.dsn = dsn
        self.embedder = embedder or OpenAIEmbedder()
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

    # ---- lookup primitives ----

    def lookup(self, namespace: str, prompt: str) -> str | None:
        """Return cached response if hit (exact or near), else None.

        Does NOT call the LLM. The decorator does that on miss.
        """
        prompt_hash = _hash(prompt)
        cutoff = (datetime.now(timezone.utc) - self.ttl).isoformat()

        with psycopg.connect(self.dsn) as conn:
            # Exact match first — sub-ms.
            row = conn.execute(
                "SELECT response, id FROM semantic_cache_entries "
                "WHERE namespace = %s AND prompt_hash = %s AND created_at >= %s",
                (namespace, prompt_hash, cutoff),
            ).fetchone()
            if row:
                self._record_hit(conn, row[1])
                return row[0]

            # Near match — cost is one embedding call.
            emb = self.embedder.embed(prompt)
            row = conn.execute(
                "SELECT response, id, 1 - (embedding <=> %s::vector) AS sim "
                "FROM semantic_cache_entries "
                "WHERE namespace = %s AND created_at >= %s "
                "ORDER BY embedding <=> %s::vector LIMIT 1",
                (_to_pgvector(emb), namespace, cutoff, _to_pgvector(emb)),
            ).fetchone()
            if row and row[2] is not None and row[2] >= self.similarity_threshold:
                self._record_hit(conn, row[1])
                return row[0]

        return None

    def store(self, namespace: str, prompt: str, response: str, model: str) -> None:
        prompt_hash = _hash(prompt)
        emb = self.embedder.embed(prompt)
        with psycopg.connect(self.dsn) as conn:
            conn.execute(
                "INSERT INTO semantic_cache_entries (namespace, prompt_hash, prompt, embedding, response, model) "
                "VALUES (%s, %s, %s, %s::vector, %s, %s) "
                "ON CONFLICT (namespace, prompt_hash) DO UPDATE SET "
                "response = EXCLUDED.response, model = EXCLUDED.model, created_at = NOW()",
                (namespace, prompt_hash, prompt, _to_pgvector(emb), response, model),
            )

    def _record_hit(self, conn, entry_id: int) -> None:
        conn.execute(
            "UPDATE semantic_cache_entries SET hit_count = hit_count + 1, last_hit_at = NOW() WHERE id = %s",
            (entry_id,),
        )

    # ---- decorator API ----

    def cached(self, *, model: str, namespace: str = "default") -> Callable:
        """Decorate a function whose first positional arg is the prompt string."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(prompt: str, *args, **kwargs):
                hit = self.lookup(namespace, prompt)
                if hit is not None:
                    return hit
                result = fn(prompt, *args, **kwargs)
                self.store(namespace, prompt, result, model)
                return result
            return wrapper
        return decorator

    def invalidate(self, namespace: str) -> int:
        """Delete every entry in a namespace. Returns rows affected."""
        with psycopg.connect(self.dsn) as conn:
            cur = conn.execute("DELETE FROM semantic_cache_entries WHERE namespace = %s", (namespace,))
            return cur.rowcount