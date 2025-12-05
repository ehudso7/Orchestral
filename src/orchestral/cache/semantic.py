"""
Semantic cache implementation for Orchestral.

Caches responses based on semantic similarity of queries,
enabling 40-60% cost reduction for similar requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field

from orchestral.cache.embeddings import (
    EmbeddingProvider,
    LocalEmbeddings,
    cosine_similarity,
)


class CacheConfig(BaseModel):
    """Configuration for semantic cache."""

    enabled: bool = Field(default=True, description="Enable caching")
    similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for cache hit (0.92 = high precision)",
    )
    max_entries: int = Field(default=10000, description="Maximum cache entries")
    ttl_seconds: int = Field(default=3600, description="Time to live in seconds")
    model_specific: bool = Field(
        default=True,
        description="Cache per model (True) or share across models (False)",
    )
    include_temperature: bool = Field(
        default=True,
        description="Include temperature in cache key",
    )


@dataclass
class CacheEntry:
    """A cached response entry."""

    query_hash: str
    query_text: str
    embedding: list[float]
    response: dict[str, Any]
    model: str
    temperature: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_cost_saved: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_text": self.query_text,
            "embedding": self.embedding,
            "response": self.response,
            "model": self.model,
            "temperature": self.temperature,
            "created_at": self.created_at.isoformat(),
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat(),
            "estimated_cost_saved": self.estimated_cost_saved,
        }


@dataclass
class CacheStats:
    """Cache statistics."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost_saved: float = 0.0
    avg_similarity_on_hit: float = 0.0
    entries_count: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_cost_saved": f"${self.total_cost_saved:.4f}",
            "avg_similarity_on_hit": f"{self.avg_similarity_on_hit:.4f}",
            "entries_count": self.entries_count,
            "evictions": self.evictions,
        }


class SemanticCache:
    """
    Intelligent semantic cache that matches similar queries.

    Features:
    - Embedding-based similarity matching
    - Configurable similarity threshold
    - LRU eviction policy
    - Per-model or shared caching
    - TTL expiration
    - Cost savings tracking

    Example:
        cache = SemanticCache()

        # Check for cached response
        result = await cache.get("What is Python?", model="gpt-4o")
        if result:
            return result.response

        # Cache new response
        await cache.set("What is Python?", response_data, model="gpt-4o")
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.config = config or CacheConfig()
        self.embeddings = embedding_provider or LocalEmbeddings()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
        self._similarity_sum = 0.0

    def _compute_key(self, query: str, model: str, temperature: float) -> str:
        """Compute a deterministic cache key."""
        key_parts = [query]
        if self.config.model_specific:
            key_parts.append(model)
        if self.config.include_temperature:
            key_parts.append(f"t={temperature:.2f}")
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
        return age > self.config.ttl_seconds

    async def _find_similar(
        self,
        query_embedding: list[float],
        model: str,
        temperature: float,
    ) -> tuple[CacheEntry | None, float]:
        """Find the most similar cached entry above threshold."""
        best_entry = None
        best_similarity = 0.0

        for entry in self._cache.values():
            # Skip if model-specific and model doesn't match
            if self.config.model_specific and entry.model != model:
                continue

            # Skip if temperature-specific and temperature doesn't match
            if self.config.include_temperature:
                if abs(entry.temperature - temperature) > 0.01:
                    continue

            # Skip expired entries
            if self._is_expired(entry):
                continue

            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_similarity >= self.config.similarity_threshold:
            return best_entry, best_similarity

        return None, best_similarity

    async def get(
        self,
        query: str,
        model: str,
        temperature: float = 0.7,
        estimated_cost: float = 0.0,
    ) -> CacheEntry | None:
        """
        Look up a cached response for a semantically similar query.

        Args:
            query: The input query text
            model: Model identifier
            temperature: Temperature setting
            estimated_cost: Estimated cost of the API call (for savings tracking)

        Returns:
            CacheEntry if found, None otherwise
        """
        if not self.config.enabled:
            return None

        async with self._lock:
            self._stats.total_queries += 1

            # Generate embedding for query
            query_embedding = await self.embeddings.embed(query)

            # Find similar entry
            entry, similarity = await self._find_similar(
                query_embedding, model, temperature
            )

            if entry is not None:
                # Cache hit
                self._stats.cache_hits += 1
                self._similarity_sum += similarity
                self._stats.avg_similarity_on_hit = (
                    self._similarity_sum / self._stats.cache_hits
                )

                # Update entry stats
                entry.hit_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                entry.estimated_cost_saved += estimated_cost
                self._stats.total_cost_saved += estimated_cost

                # Move to end (most recently used)
                self._cache.move_to_end(entry.query_hash)

                return entry

            self._stats.cache_misses += 1
            return None

    async def set(
        self,
        query: str,
        response: dict[str, Any],
        model: str,
        temperature: float = 0.7,
    ) -> CacheEntry:
        """
        Cache a response for a query.

        Args:
            query: The input query text
            response: The response data to cache
            model: Model identifier
            temperature: Temperature setting

        Returns:
            The created CacheEntry
        """
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.config.max_entries:
                # Remove oldest (least recently used)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            # Generate embedding
            query_embedding = await self.embeddings.embed(query)
            query_hash = self._compute_key(query, model, temperature)

            entry = CacheEntry(
                query_hash=query_hash,
                query_text=query,
                embedding=query_embedding,
                response=response,
                model=model,
                temperature=temperature,
            )

            self._cache[query_hash] = entry
            self._stats.entries_count = len(self._cache)

            return entry

    async def invalidate(self, model: str | None = None) -> int:
        """
        Invalidate cache entries.

        Args:
            model: If specified, only invalidate entries for this model

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            if model is None:
                count = len(self._cache)
                self._cache.clear()
                self._stats.entries_count = 0
                return count

            keys_to_remove = [
                k for k, v in self._cache.items() if v.model == model
            ]
            for key in keys_to_remove:
                del self._cache[key]

            self._stats.entries_count = len(self._cache)
            return len(keys_to_remove)

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        async with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() if self._is_expired(v)
            ]
            for key in keys_to_remove:
                del self._cache[key]

            self._stats.entries_count = len(self._cache)
            return len(keys_to_remove)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent cache entries for debugging."""
        entries = list(self._cache.values())[-limit:]
        return [e.to_dict() for e in entries]
