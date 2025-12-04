"""
Response caching for Orchestral.

Provides intelligent caching to reduce API costs for repeated queries.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """A cached response entry."""

    cache_key: str
    model: str
    prompt_hash: str
    response_content: str
    response_data: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    cost_saved_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_key": self.cache_key,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "response_content": self.response_content,
            "response_data": self.response_data,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count,
            "cost_saved_usd": self.cost_saved_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            cache_key=data["cache_key"],
            model=data["model"],
            prompt_hash=data["prompt_hash"],
            response_content=data["response_content"],
            response_data=data["response_data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            hit_count=data.get("hit_count", 0),
            cost_saved_usd=data.get("cost_saved_usd", 0.0),
        )


@dataclass
class CacheStats:
    """Cache statistics."""

    total_hits: int = 0
    total_misses: int = 0
    total_cost_saved_usd: float = 0.0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.total_hits + self.total_misses
        if total == 0:
            return 0.0
        return (self.total_hits / total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate_percent": self.hit_rate,
            "total_cost_saved_usd": self.total_cost_saved_usd,
            "cache_size": self.cache_size,
        }


class ResponseCache:
    """
    Intelligent response cache for reducing API costs.

    Caches responses based on prompt hash and model, with configurable TTL.
    Uses semantic hashing to match similar prompts.
    """

    CACHE_PREFIX = "orch:cache:"
    STATS_KEY = "orch:cache:stats"

    def __init__(
        self,
        redis_client: Any | None = None,
        default_ttl_seconds: int = 3600,
        max_entries: int = 10000,
        enabled: bool = True,
    ):
        """
        Initialize the response cache.

        Args:
            redis_client: Redis client for persistence
            default_ttl_seconds: Default cache TTL
            max_entries: Maximum cache entries (for local storage)
            enabled: Whether caching is enabled
        """
        self._redis = redis_client
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._enabled = enabled

        # Local fallback storage
        self._local_cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

    def _generate_cache_key(
        self,
        messages: list[dict[str, Any]] | str,
        model: str,
        temperature: float,
        max_tokens: int,
        tenant_id: str | None = None,
    ) -> str:
        """
        Generate a cache key for a request.

        Args:
            messages: The prompt/messages
            model: Model ID
            temperature: Temperature setting
            max_tokens: Max tokens setting
            tenant_id: API key ID for tenant isolation (prevents cross-tenant leaks)

        Returns:
            Cache key string
        """
        # Normalize messages
        if isinstance(messages, str):
            prompt_str = messages
        else:
            # Combine message contents
            prompt_str = "".join(
                f"{m.get('role', 'user')}:{m.get('content', '')}"
                for m in messages
            )

        # Create hash of request parameters
        # Note: Only cache for temperature=0 (deterministic) or low temperatures
        if temperature > 0.3:
            # Don't cache high-temperature requests (non-deterministic)
            return ""

        # Include tenant_id in cache key to prevent cross-tenant cache leaks
        tenant_prefix = tenant_id or "global"
        key_data = f"{tenant_prefix}:{model}:{temperature}:{max_tokens}:{prompt_str}"
        prompt_hash = hashlib.sha256(key_data.encode()).hexdigest()[:32]

        return f"{tenant_prefix}:{model}:{prompt_hash}"

    async def get(
        self,
        messages: list[dict[str, Any]] | str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tenant_id: str | None = None,
    ) -> CacheEntry | None:
        """
        Get a cached response if available.

        Args:
            messages: The prompt/messages
            model: Model ID
            temperature: Temperature setting
            max_tokens: Max tokens setting
            tenant_id: API key ID for tenant isolation

        Returns:
            CacheEntry if found, None otherwise
        """
        if not self._enabled:
            return None

        cache_key = self._generate_cache_key(messages, model, temperature, max_tokens, tenant_id)
        if not cache_key:
            return None

        entry = await self._get_entry(cache_key)

        if entry is None:
            self._stats.total_misses += 1
            logger.debug("Cache miss", cache_key=cache_key)
            return None

        # Check expiration
        if datetime.now(timezone.utc) > entry.expires_at:
            await self._delete_entry(cache_key)
            self._stats.total_misses += 1
            logger.debug("Cache expired", cache_key=cache_key)
            return None

        # Update hit count
        entry.hit_count += 1
        await self._store_entry(entry)

        self._stats.total_hits += 1

        logger.debug(
            "Cache hit",
            cache_key=cache_key,
            hit_count=entry.hit_count,
        )

        return entry

    async def set(
        self,
        messages: list[dict[str, Any]] | str,
        model: str,
        response_content: str,
        response_data: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        ttl_seconds: int | None = None,
        cost_usd: float = 0.0,
        tenant_id: str | None = None,
    ) -> CacheEntry | None:
        """
        Cache a response.

        Args:
            messages: The prompt/messages
            model: Model ID
            response_content: The response content
            response_data: Full response data
            temperature: Temperature setting
            max_tokens: Max tokens setting
            ttl_seconds: Custom TTL (uses default if None)
            cost_usd: Cost of this request (for savings calculation)
            tenant_id: API key ID for tenant isolation

        Returns:
            CacheEntry if cached, None if not cacheable
        """
        if not self._enabled:
            return None

        cache_key = self._generate_cache_key(messages, model, temperature, max_tokens, tenant_id)
        if not cache_key:
            return None

        ttl = ttl_seconds or self._default_ttl
        now = datetime.now(timezone.utc)

        # Create prompt hash for reference
        if isinstance(messages, str):
            prompt_str = messages
        else:
            prompt_str = str(messages)
        prompt_hash = hashlib.sha256(prompt_str.encode()).hexdigest()[:16]

        entry = CacheEntry(
            cache_key=cache_key,
            model=model,
            prompt_hash=prompt_hash,
            response_content=response_content,
            response_data=response_data,
            created_at=now,
            expires_at=datetime.fromtimestamp(now.timestamp() + ttl, tz=timezone.utc),
            hit_count=0,
            cost_saved_usd=cost_usd,
        )

        await self._store_entry(entry)

        logger.debug(
            "Response cached",
            cache_key=cache_key,
            ttl_seconds=ttl,
        )

        return entry

    async def invalidate(
        self,
        model: str | None = None,
        older_than: datetime | None = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            model: Invalidate only entries for this model
            older_than: Invalidate entries older than this time

        Returns:
            Number of entries invalidated
        """
        count = 0

        if self._redis:
            pattern = f"{self.CACHE_PREFIX}{model or '*'}:*"
            for key in self._redis.scan_iter(pattern):
                if older_than:
                    data = self._redis.get(key)
                    if data:
                        entry = CacheEntry.from_dict(json.loads(data))
                        if entry.created_at < older_than:
                            self._redis.delete(key)
                            count += 1
                else:
                    self._redis.delete(key)
                    count += 1
        else:
            keys_to_delete = []
            for key, entry in self._local_cache.items():
                if model and not key.startswith(model):
                    continue
                if older_than and entry.created_at >= older_than:
                    continue
                keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._local_cache[key]
                count += 1

        logger.info("Cache invalidated", entries=count, model=model)
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        if self._redis:
            self._stats.cache_size = len(list(self._redis.scan_iter(f"{self.CACHE_PREFIX}*")))
        else:
            self._stats.cache_size = len(self._local_cache)

        return self._stats

    async def record_cost_saved(self, cost_usd: float) -> None:
        """Record cost saved from a cache hit."""
        self._stats.total_cost_saved_usd += cost_usd

        if self._redis:
            self._redis.hincrbyfloat(self.STATS_KEY, "total_cost_saved_usd", cost_usd)

    async def _get_entry(self, cache_key: str) -> CacheEntry | None:
        """Get a cache entry."""
        full_key = f"{self.CACHE_PREFIX}{cache_key}"

        if self._redis:
            data = self._redis.get(full_key)
            if data:
                return CacheEntry.from_dict(json.loads(data))
            return None
        else:
            return self._local_cache.get(cache_key)

    async def _store_entry(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        full_key = f"{self.CACHE_PREFIX}{entry.cache_key}"

        if self._redis:
            ttl = int((entry.expires_at - datetime.now(timezone.utc)).total_seconds())
            # Guard against negative or zero TTL (entry already expired)
            if ttl <= 0:
                logger.debug("Skipping cache store for expired entry", cache_key=entry.cache_key)
                return
            self._redis.setex(full_key, ttl, json.dumps(entry.to_dict()))
        else:
            self._local_cache[entry.cache_key] = entry

            # Evict old entries if over limit
            if len(self._local_cache) > self._max_entries:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._local_cache.keys(),
                    key=lambda k: self._local_cache[k].created_at,
                )
                for key in sorted_keys[:len(self._local_cache) - self._max_entries]:
                    del self._local_cache[key]

    async def _delete_entry(self, cache_key: str) -> None:
        """Delete a cache entry."""
        full_key = f"{self.CACHE_PREFIX}{cache_key}"

        if self._redis:
            self._redis.delete(full_key)
        else:
            self._local_cache.pop(cache_key, None)
