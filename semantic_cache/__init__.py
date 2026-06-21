"""semantic-cache — production semantic cache for LLM responses."""

from .cache import SemanticCache
from .embeddings import Embedder, OpenAIEmbedder

__version__ = "0.1.0"
__all__ = ["SemanticCache", "Embedder", "OpenAIEmbedder"]