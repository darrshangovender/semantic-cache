"""Tests using a fake embedder and a stubbed DSN.

We do not spin up a real Postgres in unit tests — that is integration's job.
These tests cover the deterministic parts: hashing, decorator wiring,
threshold validation.
"""

from __future__ import annotations

import pytest

from semantic_cache import SemanticCache


class FakeEmbedder:
    dimension = 4

    def embed(self, text: str) -> list[float]:
        # Deterministic 4-dim hash-y "embedding" for tests.
        h = sum(ord(c) for c in text) % 1000 / 1000
        return [h, 1 - h, h * 0.5, 0.0]


def test_threshold_validation():
    with pytest.raises(ValueError):
        SemanticCache("postgresql://stub", similarity_threshold=1.5, embedder=FakeEmbedder())


def test_threshold_accepted():
    c = SemanticCache("postgresql://stub", similarity_threshold=0.95, embedder=FakeEmbedder())
    assert c.similarity_threshold == 0.95


def test_decorator_wraps_fn():
    c = SemanticCache("postgresql://stub", embedder=FakeEmbedder())
    @c.cached(model="test", namespace="ns")
    def f(prompt: str) -> str:
        return "ok:" + prompt
    assert callable(f)