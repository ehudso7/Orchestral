"""
Semantic caching module for Orchestral.

Provides intelligent caching that matches semantically similar queries,
reducing API costs by 40-60% while maintaining response quality.
"""

from orchestral.cache.semantic import SemanticCache, CacheEntry, CacheConfig
from orchestral.cache.embeddings import EmbeddingProvider, LocalEmbeddings

__all__ = [
    "SemanticCache",
    "CacheEntry",
    "CacheConfig",
    "EmbeddingProvider",
    "LocalEmbeddings",
]
