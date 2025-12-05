"""
Semantic caching for Orchestral.

Uses embeddings to find semantically similar cached responses,
providing cache hits even when prompts are worded differently.
This is a major differentiator from exact-match caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class SemanticCacheEntry:
    """A semantically cached response with embedding."""

    cache_id: str
    embedding: list[float]
    model: str
    prompt_text: str
    response_content: str
    response_data: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    cost_saved_usd: float = 0.0
    tenant_id: str = "global"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "cache_id": self.cache_id,
            "embedding": self.embedding,
            "model": self.model,
            "prompt_text": self.prompt_text,
            "response_content": self.response_content,
            "response_data": self.response_data,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count,
            "cost_saved_usd": self.cost_saved_usd,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticCacheEntry":
        """Create from dictionary."""
        return cls(
            cache_id=data["cache_id"],
            embedding=data["embedding"],
            model=data["model"],
            prompt_text=data["prompt_text"],
            response_content=data["response_content"],
            response_data=data["response_data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            hit_count=data.get("hit_count", 0),
            cost_saved_usd=data.get("cost_saved_usd", 0.0),
            tenant_id=data.get("tenant_id", "global"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SemanticCacheStats:
    """Statistics for semantic cache."""

    total_hits: int = 0
    total_misses: int = 0
    semantic_hits: int = 0  # Hits via semantic similarity (not exact match)
    exact_hits: int = 0  # Hits via exact hash match
    total_cost_saved_usd: float = 0.0
    cache_size: int = 0
    avg_similarity_score: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.total_hits + self.total_misses
        if total == 0:
            return 0.0
        return (self.total_hits / total) * 100

    @property
    def semantic_hit_rate(self) -> float:
        """Semantic hit rate (% of hits that were semantic vs exact)."""
        if self.total_hits == 0:
            return 0.0
        return (self.semantic_hits / self.total_hits) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "semantic_hits": self.semantic_hits,
            "exact_hits": self.exact_hits,
            "hit_rate_percent": self.hit_rate,
            "semantic_hit_rate_percent": self.semantic_hit_rate,
            "total_cost_saved_usd": self.total_cost_saved_usd,
            "cache_size": self.cache_size,
            "avg_similarity_score": self.avg_similarity_score,
        }


class EmbeddingProvider:
    """Generates embeddings for semantic similarity."""

    def __init__(self, provider: str = "openai", model: str = "text-embedding-3-small"):
        self.provider = provider
        self.model = model
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Lazy load embedding client."""
        if self._client is None:
            if self.provider == "openai":
                import openai
                self._client = openai.AsyncOpenAI()
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            if self.provider == "openai":
                client = await self._get_client()
                response = await client.embeddings.create(
                    model=self.model,
                    input=text[:8000],  # Truncate to model limit
                )
                return response.data[0].embedding
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.warning("Embedding generation failed", error=str(e))
            # Return zero vector as fallback (will force cache miss)
            return [0.0] * 1536

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if self.provider == "openai":
                client = await self._get_client()
                response = await client.embeddings.create(
                    model=self.model,
                    input=[t[:8000] for t in texts],
                )
                return [item.embedding for item in response.data]
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.warning("Batch embedding failed", error=str(e))
            return [[0.0] * 1536 for _ in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class SemanticCache:
    """
    Semantic response cache using embeddings for similarity matching.

    Key differentiator from competitors:
    - Finds cache hits even when prompts are worded differently
    - Configurable similarity threshold
    - Falls back to exact matching for high-confidence cases
    - Tenant-isolated embeddings
    """

    CACHE_PREFIX = "orch:semantic:"
    INDEX_PREFIX = "orch:semantic:idx:"
    STATS_KEY = "orch:semantic:stats"

    def __init__(
        self,
        redis_client: Any | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.92,
        default_ttl_seconds: int = 3600,
        max_entries: int = 50000,
        enabled: bool = True,
    ):
        """
        Initialize semantic cache.

        Args:
            redis_client: Redis client for persistence
            embedding_provider: Provider for generating embeddings
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
            default_ttl_seconds: Default cache TTL
            max_entries: Maximum cache entries
            enabled: Whether semantic caching is enabled
        """
        self._redis = redis_client
        self._embedder = embedding_provider or EmbeddingProvider()
        self._similarity_threshold = similarity_threshold
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._enabled = enabled

        # Local storage for non-Redis mode
        self._local_cache: dict[str, SemanticCacheEntry] = {}
        self._local_embeddings: list[tuple[str, list[float]]] = []
        self._stats = SemanticCacheStats()
        # Bounded deque to prevent memory leak - stores last 10000 similarity scores
        self._similarity_scores: deque[float] = deque(maxlen=10000)

    async def get(
        self,
        prompt: str,
        model: str,
        tenant_id: str | None = None,
        temperature: float = 0.0,
    ) -> tuple[SemanticCacheEntry | None, float]:
        """
        Find semantically similar cached response.

        Args:
            prompt: The prompt text
            model: Model ID
            tenant_id: Tenant ID for isolation
            temperature: Temperature setting (only cache low temps)

        Returns:
            Tuple of (CacheEntry or None, similarity_score)
        """
        if not self._enabled or temperature > 0.3:
            return None, 0.0

        tenant = tenant_id or "global"

        # First try exact hash match
        exact_key = self._generate_exact_key(prompt, model, tenant)
        exact_entry = await self._get_exact(exact_key)
        if exact_entry:
            self._stats.total_hits += 1
            self._stats.exact_hits += 1
            exact_entry.hit_count += 1
            await self._store_entry(exact_entry)
            logger.debug("Semantic cache exact hit", cache_id=exact_entry.cache_id)
            return exact_entry, 1.0

        # Generate embedding for semantic search
        prompt_embedding = await self._embedder.embed(prompt)

        # Search for similar entries
        best_match, best_score = await self._search_similar(
            prompt_embedding, model, tenant
        )

        if best_match and best_score >= self._similarity_threshold:
            self._stats.total_hits += 1
            self._stats.semantic_hits += 1
            self._similarity_scores.append(best_score)
            self._stats.avg_similarity_score = sum(self._similarity_scores) / len(
                self._similarity_scores
            )
            best_match.hit_count += 1
            await self._store_entry(best_match)
            logger.debug(
                "Semantic cache hit",
                cache_id=best_match.cache_id,
                similarity=best_score,
            )
            return best_match, best_score

        self._stats.total_misses += 1
        logger.debug("Semantic cache miss", model=model, tenant=tenant)
        return None, best_score if best_match else 0.0

    async def set(
        self,
        prompt: str,
        model: str,
        response_content: str,
        response_data: dict[str, Any],
        tenant_id: str | None = None,
        temperature: float = 0.0,
        ttl_seconds: int | None = None,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticCacheEntry | None:
        """
        Cache a response with semantic embedding.

        Args:
            prompt: The prompt text
            model: Model ID
            response_content: Response content
            response_data: Full response data
            tenant_id: Tenant ID
            temperature: Temperature setting
            ttl_seconds: Custom TTL
            cost_usd: Cost of this request
            metadata: Additional metadata

        Returns:
            CacheEntry if cached, None if not cacheable
        """
        if not self._enabled or temperature > 0.3:
            return None

        tenant = tenant_id or "global"
        ttl = ttl_seconds or self._default_ttl
        now = datetime.now(timezone.utc)

        # Generate embedding
        embedding = await self._embedder.embed(prompt)

        # Generate cache ID
        cache_id = hashlib.sha256(
            f"{tenant}:{model}:{prompt}:{now.timestamp()}".encode()
        ).hexdigest()[:24]

        entry = SemanticCacheEntry(
            cache_id=cache_id,
            embedding=embedding,
            model=model,
            prompt_text=prompt[:1000],  # Store truncated for reference
            response_content=response_content,
            response_data=response_data,
            created_at=now,
            expires_at=datetime.fromtimestamp(now.timestamp() + ttl, tz=timezone.utc),
            hit_count=0,
            cost_saved_usd=cost_usd,
            tenant_id=tenant,
            metadata=metadata or {},
        )

        await self._store_entry(entry)

        # Also store exact hash for fast lookup
        exact_key = self._generate_exact_key(prompt, model, tenant)
        await self._store_exact_mapping(exact_key, cache_id)

        logger.debug("Semantic cache stored", cache_id=cache_id, model=model)
        return entry

    def _generate_exact_key(self, prompt: str, model: str, tenant: str) -> str:
        """Generate exact-match hash key."""
        key_data = f"{tenant}:{model}:{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def _get_exact(self, exact_key: str) -> SemanticCacheEntry | None:
        """Get entry by exact hash match."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                cache_id = await loop.run_in_executor(
                    None, self._redis.get, f"{self.INDEX_PREFIX}exact:{exact_key}"
                )
                if cache_id:
                    cache_id_str = (
                        cache_id.decode("utf-8")
                        if isinstance(cache_id, bytes)
                        else cache_id
                    )
                    return await self._get_by_id(cache_id_str)
            except Exception as e:
                logger.warning("Exact cache lookup failed", error=str(e))
        else:
            for entry in self._local_cache.values():
                check_key = self._generate_exact_key(
                    entry.prompt_text, entry.model, entry.tenant_id
                )
                if check_key == exact_key:
                    return entry
        return None

    async def _store_exact_mapping(self, exact_key: str, cache_id: str) -> None:
        """Store exact hash to cache ID mapping."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.setex(
                        f"{self.INDEX_PREFIX}exact:{exact_key}",
                        self._default_ttl,
                        cache_id,
                    ),
                )
            except Exception as e:
                logger.warning("Failed to store exact mapping", error=str(e))

    async def _search_similar(
        self,
        query_embedding: list[float],
        model: str,
        tenant: str,
    ) -> tuple[SemanticCacheEntry | None, float]:
        """Search for semantically similar cached entries."""
        best_entry: SemanticCacheEntry | None = None
        best_score = 0.0

        if self._redis:
            # Load all entries for this tenant/model and compute similarity
            # In production, use Redis vector search (RediSearch) for efficiency
            loop = asyncio.get_running_loop()
            try:
                pattern = f"{self.CACHE_PREFIX}{tenant}:{model}:*"
                keys = await loop.run_in_executor(
                    None, lambda: list(self._redis.scan_iter(pattern, count=1000))
                )

                for key in keys[:500]:  # Limit search space
                    data = await loop.run_in_executor(None, self._redis.get, key)
                    if data:
                        try:
                            entry = SemanticCacheEntry.from_dict(json.loads(data))
                            if datetime.now(timezone.utc) > entry.expires_at:
                                continue
                            score = cosine_similarity(query_embedding, entry.embedding)
                            if score > best_score:
                                best_score = score
                                best_entry = entry
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception as e:
                logger.warning("Semantic search failed", error=str(e))
        else:
            # Local cache search
            for entry in self._local_cache.values():
                if entry.model != model or entry.tenant_id != tenant:
                    continue
                if datetime.now(timezone.utc) > entry.expires_at:
                    continue
                score = cosine_similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        return best_entry, best_score

    async def _get_by_id(self, cache_id: str) -> SemanticCacheEntry | None:
        """Get entry by cache ID."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                # Search for the entry
                pattern = f"{self.CACHE_PREFIX}*:{cache_id}"
                keys = await loop.run_in_executor(
                    None, lambda: list(self._redis.scan_iter(pattern, count=100))
                )
                for key in keys:
                    data = await loop.run_in_executor(None, self._redis.get, key)
                    if data:
                        return SemanticCacheEntry.from_dict(json.loads(data))
            except Exception as e:
                logger.warning("Cache ID lookup failed", error=str(e))
        else:
            return self._local_cache.get(cache_id)
        return None

    async def _store_entry(self, entry: SemanticCacheEntry) -> None:
        """Store a cache entry."""
        full_key = f"{self.CACHE_PREFIX}{entry.tenant_id}:{entry.model}:{entry.cache_id}"

        if self._redis:
            ttl = int(
                (entry.expires_at - datetime.now(timezone.utc)).total_seconds()
            )
            if ttl <= 0:
                return
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.setex(full_key, ttl, json.dumps(entry.to_dict())),
                )
            except Exception as e:
                logger.warning("Failed to store semantic entry", error=str(e))
        else:
            self._local_cache[entry.cache_id] = entry
            # Evict old entries if needed
            if len(self._local_cache) > self._max_entries:
                sorted_entries = sorted(
                    self._local_cache.items(),
                    key=lambda x: x[1].created_at,
                )
                for key, _ in sorted_entries[: len(self._local_cache) - self._max_entries]:
                    del self._local_cache[key]

    async def get_stats(self) -> SemanticCacheStats:
        """Get cache statistics."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                count = 0
                for _ in await loop.run_in_executor(
                    None,
                    lambda: list(self._redis.scan_iter(f"{self.CACHE_PREFIX}*", count=1000)),
                ):
                    count += 1
                self._stats.cache_size = count
            except Exception as e:
                logger.warning("Failed to get semantic cache size", error=str(e))
        else:
            self._stats.cache_size = len(self._local_cache)
        return self._stats

    async def record_cost_saved(self, cost_usd: float) -> None:
        """Record cost saved from a cache hit."""
        self._stats.total_cost_saved_usd += cost_usd
