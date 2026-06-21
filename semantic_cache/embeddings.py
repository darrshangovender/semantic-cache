"""Pluggable embedder. Default is OpenAI text-embedding-3-small (1536 dim)."""

from __future__ import annotations

import os
from typing import Protocol


class Embedder(Protocol):
    """Anything that turns a string into a list[float] of fixed dimension."""

    dimension: int

    def embed(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small embedder. 1536-dim, ~$0.02 / 1M tokens."""

    dimension = 1536

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding