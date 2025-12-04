"""
Redis-based rate limiting for Orchestral.

Provides persistent, distributed rate limiting that works across serverless instances.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp
    retry_after: int | None = None  # Seconds until retry allowed

    @property
    def headers(self) -> dict[str, str]:
        """Get rate limit headers for HTTP response."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class RateLimiter:
    """
    Redis-based sliding window rate limiter.

    Uses Redis sorted sets for efficient, distributed rate limiting
    that persists across serverless cold starts.
    """

    KEY_PREFIX = "orch:ratelimit:"

    def __init__(
        self,
        redis_client: Any | None = None,
        default_limit: int = 100,
        default_window: int = 60,
    ):
        """
        Initialize the rate limiter.

        Args:
            redis_client: Redis client for persistence
            default_limit: Default requests per window
            default_window: Default window size in seconds
        """
        self._redis = redis_client
        self._default_limit = default_limit
        self._default_window = default_window

        # Fallback in-memory store for when Redis is unavailable
        self._local_store: dict[str, list[float]] = {}

    async def check(
        self,
        identifier: str,
        limit: int | None = None,
        window: int | None = None,
    ) -> RateLimitResult:
        """
        Check if a request is allowed under rate limits.

        Args:
            identifier: Unique identifier (API key ID, IP address, etc.)
            limit: Request limit for this check (overrides default)
            window: Window size in seconds (overrides default)

        Returns:
            RateLimitResult with allowed status and metadata
        """
        limit = limit or self._default_limit
        window = window or self._default_window
        now = time.time()
        window_start = now - window

        key = f"{self.KEY_PREFIX}{identifier}"

        if self._redis:
            return await self._check_redis(key, limit, window, now, window_start)
        else:
            return self._check_local(key, limit, window, now, window_start)

    async def _check_redis(
        self,
        key: str,
        limit: int,
        window: int,
        now: float,
        window_start: float,
    ) -> RateLimitResult:
        """Check rate limit using Redis."""
        loop = asyncio.get_event_loop()

        # Run sync Redis pipeline in executor to avoid blocking
        def _execute_pipeline():
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window * 2)
            return pipe.execute()

        results = await loop.run_in_executor(None, _execute_pipeline)
        current_count = results[1]

        remaining = limit - current_count - 1
        allowed = current_count < limit
        reset_at = now + window

        if not allowed:
            # Calculate retry after - run in executor
            oldest = await loop.run_in_executor(
                None, lambda: self._redis.zrange(key, 0, 0, withscores=True)
            )
            if oldest:
                retry_after = int(oldest[0][1] + window - now) + 1
            else:
                retry_after = window

            logger.warning(
                "Rate limit exceeded",
                key=key,
                current=current_count,
                limit=limit,
            )

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=limit,
            reset_at=reset_at,
        )

    def _check_local(
        self,
        key: str,
        limit: int,
        window: int,
        now: float,
        window_start: float,
    ) -> RateLimitResult:
        """Check rate limit using local storage (fallback)."""
        if key not in self._local_store:
            self._local_store[key] = []

        # Remove old entries
        self._local_store[key] = [
            ts for ts in self._local_store[key] if ts > window_start
        ]

        current_count = len(self._local_store[key])
        remaining = limit - current_count - 1
        allowed = current_count < limit
        reset_at = now + window

        if allowed:
            self._local_store[key].append(now)
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                limit=limit,
                reset_at=reset_at,
            )
        else:
            retry_after = int(self._local_store[key][0] + window - now) + 1
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                reset_at=reset_at,
                retry_after=retry_after,
            )

    async def check_multiple(
        self,
        identifier: str,
        limits: list[tuple[str, int, int]],
    ) -> tuple[bool, RateLimitResult | None]:
        """
        Check multiple rate limits (per-minute, per-day, etc.).

        Args:
            identifier: Unique identifier
            limits: List of (name, limit, window) tuples

        Returns:
            Tuple of (all_allowed, first_failed_result)
        """
        for name, limit, window in limits:
            key = f"{identifier}:{name}"
            result = await self.check(key, limit, window)
            if not result.allowed:
                return False, result
        return True, None

    async def get_usage(self, identifier: str, window: int | None = None) -> int:
        """
        Get current usage count for an identifier.

        Args:
            identifier: Unique identifier
            window: Window size in seconds

        Returns:
            Current request count in window
        """
        window = window or self._default_window
        now = time.time()
        window_start = now - window
        key = f"{self.KEY_PREFIX}{identifier}"

        if self._redis:
            loop = asyncio.get_event_loop()

            def _get_count():
                self._redis.zremrangebyscore(key, 0, window_start)
                return self._redis.zcard(key)

            return await loop.run_in_executor(None, _get_count)
        else:
            if key in self._local_store:
                self._local_store[key] = [
                    ts for ts in self._local_store[key] if ts > window_start
                ]
                return len(self._local_store[key])
            return 0

    async def reset(self, identifier: str) -> None:
        """
        Reset rate limits for an identifier.

        Args:
            identifier: Unique identifier
        """
        key = f"{self.KEY_PREFIX}{identifier}"

        if self._redis:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._redis.delete, key)
        else:
            self._local_store.pop(key, None)

        logger.info("Rate limit reset", identifier=identifier)
