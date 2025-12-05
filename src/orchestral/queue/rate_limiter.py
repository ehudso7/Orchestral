"""
Rate limiting and request queue management.

Handles API rate limits gracefully with queuing, backoff, and prioritization.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, TypeVar, Generic
import heapq

from pydantic import BaseModel, Field


T = TypeVar("T")


class Priority(int, Enum):
    """Request priority levels."""

    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    # Requests per minute per model
    requests_per_minute: dict[str, int] = Field(
        default={
            "gpt-4o": 500,
            "gpt-4o-mini": 1000,
            "gpt-5.1": 200,
            "o1": 100,
            "claude-3-5-sonnet-20241022": 400,
            "claude-3-opus-20240229": 200,
            "claude-3-haiku-20240307": 1000,
            "gemini-3-pro-preview": 300,
            "gemini-2.5-flash": 1000,
        },
        description="Rate limits per model (requests/minute)",
    )

    # Tokens per minute per model
    tokens_per_minute: dict[str, int] = Field(
        default={
            "gpt-4o": 150000,
            "gpt-4o-mini": 500000,
            "gpt-5.1": 100000,
            "o1": 50000,
            "claude-3-5-sonnet-20241022": 150000,
            "claude-3-opus-20240229": 100000,
            "claude-3-haiku-20240307": 500000,
            "gemini-3-pro-preview": 200000,
            "gemini-2.5-flash": 500000,
        },
        description="Token limits per model (tokens/minute)",
    )

    # Queue settings
    max_queue_size: int = Field(default=1000, description="Maximum queue size")
    queue_timeout_seconds: int = Field(default=300, description="Max time in queue")

    # Backoff settings
    initial_backoff_ms: int = Field(default=1000, description="Initial backoff")
    max_backoff_ms: int = Field(default=60000, description="Maximum backoff")
    backoff_multiplier: float = Field(default=2.0, description="Backoff multiplier")


@dataclass(order=True)
class QueuedRequest(Generic[T]):
    """A request waiting in the queue."""

    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    model: str = field(compare=False)
    estimated_tokens: int = field(compare=False)
    callback: Callable[[], Awaitable[T]] = field(compare=False)
    future: asyncio.Future = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        # Ensure priority is the sort key
        pass


@dataclass
class QueueStats:
    """Queue statistics."""

    total_queued: int = 0
    total_processed: int = 0
    total_dropped: int = 0
    total_timeouts: int = 0
    current_queue_size: int = 0
    avg_wait_time_ms: float = 0.0
    rate_limit_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queued": self.total_queued,
            "total_processed": self.total_processed,
            "total_dropped": self.total_dropped,
            "total_timeouts": self.total_timeouts,
            "current_queue_size": self.current_queue_size,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "rate_limit_hits": self.rate_limit_hits,
        }


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: float):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        """
        Try to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = don't wait)

        Returns:
            True if tokens were acquired, False otherwise
        """
        start_time = time.monotonic()

        while True:
            async with self._lock:
                # Refill tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = min(tokens_needed / self.rate, timeout - elapsed)
                await asyncio.sleep(wait_time)
            else:
                return False

    def get_wait_time(self, tokens: float = 1.0) -> float:
        """Calculate wait time to acquire tokens."""
        if self.tokens >= tokens:
            return 0.0
        return (tokens - self.tokens) / self.rate


class RateLimiter:
    """
    Rate limiter for API requests.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Per-model rate limits
    - Automatic backoff on errors
    - Rate limit tracking

    Example:
        limiter = RateLimiter()

        # Check if request can proceed
        if await limiter.acquire("gpt-4o"):
            # Make request
            pass
        else:
            # Rate limited, wait or queue
            wait_time = limiter.get_wait_time("gpt-4o")
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()

        # Token buckets per model (for requests)
        self._request_buckets: dict[str, TokenBucket] = {}

        # Token buckets per model (for tokens)
        self._token_buckets: dict[str, TokenBucket] = {}

        # Backoff tracking
        self._backoff_until: dict[str, float] = {}
        self._consecutive_errors: dict[str, int] = {}

        # Stats
        self._stats: dict[str, dict[str, int]] = {}

    def _get_request_bucket(self, model: str) -> TokenBucket:
        """Get or create request bucket for model."""
        if model not in self._request_buckets:
            rpm = self.config.requests_per_minute.get(model, 100)
            rate = rpm / 60.0  # Requests per second
            self._request_buckets[model] = TokenBucket(rate, rpm)
        return self._request_buckets[model]

    def _get_token_bucket(self, model: str) -> TokenBucket:
        """Get or create token bucket for model."""
        if model not in self._token_buckets:
            tpm = self.config.tokens_per_minute.get(model, 100000)
            rate = tpm / 60.0  # Tokens per second
            self._token_buckets[model] = TokenBucket(rate, tpm)
        return self._token_buckets[model]

    def _is_in_backoff(self, model: str) -> bool:
        """Check if model is in backoff period."""
        if model not in self._backoff_until:
            return False
        return time.monotonic() < self._backoff_until[model]

    def _record_stat(self, model: str, stat: str) -> None:
        """Record a statistic."""
        if model not in self._stats:
            self._stats[model] = {}
        self._stats[model][stat] = self._stats[model].get(stat, 0) + 1

    async def acquire(
        self,
        model: str,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """
        Try to acquire permission to make a request.

        Args:
            model: Model identifier
            tokens: Estimated token count for the request
            timeout: Maximum time to wait

        Returns:
            True if request can proceed, False if rate limited
        """
        # Check backoff
        if self._is_in_backoff(model):
            self._record_stat(model, "backoff_rejects")
            return False

        # Acquire from request bucket
        request_bucket = self._get_request_bucket(model)
        if not await request_bucket.acquire(1.0, timeout):
            self._record_stat(model, "request_rate_limits")
            return False

        # Acquire from token bucket
        token_bucket = self._get_token_bucket(model)
        if not await token_bucket.acquire(float(tokens), timeout):
            self._record_stat(model, "token_rate_limits")
            return False

        self._record_stat(model, "acquired")
        return True

    def get_wait_time(self, model: str, tokens: int = 1) -> float:
        """
        Get estimated wait time before request can proceed.

        Returns:
            Wait time in seconds
        """
        # Check backoff
        if self._is_in_backoff(model):
            return self._backoff_until[model] - time.monotonic()

        request_bucket = self._get_request_bucket(model)
        token_bucket = self._get_token_bucket(model)

        request_wait = request_bucket.get_wait_time(1.0)
        token_wait = token_bucket.get_wait_time(float(tokens))

        return max(request_wait, token_wait)

    def record_success(self, model: str) -> None:
        """Record a successful request (resets backoff)."""
        self._consecutive_errors[model] = 0
        self._record_stat(model, "successes")

    def record_rate_limit_error(self, model: str, retry_after: float | None = None) -> None:
        """
        Record a rate limit error.

        Args:
            model: Model that was rate limited
            retry_after: Suggested retry time from API (if provided)
        """
        self._record_stat(model, "rate_limit_errors")

        # Increment consecutive errors
        errors = self._consecutive_errors.get(model, 0) + 1
        self._consecutive_errors[model] = errors

        # Calculate backoff
        if retry_after:
            backoff_seconds = retry_after
        else:
            base_backoff = self.config.initial_backoff_ms / 1000.0
            backoff_seconds = min(
                base_backoff * (self.config.backoff_multiplier ** (errors - 1)),
                self.config.max_backoff_ms / 1000.0,
            )

        self._backoff_until[model] = time.monotonic() + backoff_seconds

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get rate limiter statistics."""
        return dict(self._stats)


class RequestQueue:
    """
    Priority queue for rate-limited requests.

    Features:
    - Priority-based ordering
    - Timeout handling
    - Automatic processing
    - Backpressure support

    Example:
        queue = RequestQueue(rate_limiter)

        # Submit request
        future = await queue.submit(
            model="gpt-4o",
            callback=lambda: client.complete(...),
            priority=Priority.NORMAL,
        )

        # Wait for result
        result = await future
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        config: RateLimitConfig | None = None,
    ):
        self.rate_limiter = rate_limiter
        self.config = config or RateLimitConfig()

        self._queue: list[QueuedRequest] = []
        self._stats = QueueStats()
        self._running = False
        self._processor_task: asyncio.Task | None = None
        self._request_counter = 0
        self._lock = asyncio.Lock()
        self._total_wait_time = 0.0

    async def start(self) -> None:
        """Start the queue processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Stop the queue processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def submit(
        self,
        model: str,
        callback: Callable[[], Awaitable[T]],
        priority: Priority = Priority.NORMAL,
        estimated_tokens: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> asyncio.Future[T]:
        """
        Submit a request to the queue.

        Args:
            model: Model to use
            callback: Async function to execute when ready
            priority: Request priority
            estimated_tokens: Estimated token count
            metadata: Additional metadata

        Returns:
            Future that will contain the result
        """
        async with self._lock:
            if len(self._queue) >= self.config.max_queue_size:
                self._stats.total_dropped += 1
                raise RuntimeError("Queue is full")

            self._request_counter += 1
            request_id = f"req_{self._request_counter}"

            loop = asyncio.get_event_loop()
            future: asyncio.Future[T] = loop.create_future()

            request = QueuedRequest(
                priority=priority.value,
                timestamp=time.monotonic(),
                request_id=request_id,
                model=model,
                estimated_tokens=estimated_tokens,
                callback=callback,
                future=future,
                metadata=metadata or {},
            )

            heapq.heappush(self._queue, request)
            self._stats.total_queued += 1
            self._stats.current_queue_size = len(self._queue)

        return future

    async def _process_queue(self) -> None:
        """Process queued requests."""
        while self._running:
            try:
                # Get next request
                request = None
                async with self._lock:
                    if self._queue:
                        # Check for timeouts
                        now = time.monotonic()
                        while self._queue:
                            oldest = self._queue[0]
                            age = now - oldest.timestamp
                            if age > self.config.queue_timeout_seconds:
                                heapq.heappop(self._queue)
                                oldest.future.set_exception(
                                    TimeoutError("Request timed out in queue")
                                )
                                self._stats.total_timeouts += 1
                                continue

                            # Get the highest priority request
                            request = heapq.heappop(self._queue)
                            self._stats.current_queue_size = len(self._queue)
                            break

                if request:
                    # Try to acquire rate limit
                    wait_time = self.rate_limiter.get_wait_time(
                        request.model, request.estimated_tokens
                    )

                    if wait_time > 0:
                        # Check if we'd timeout while waiting
                        age = time.monotonic() - request.timestamp
                        if age + wait_time > self.config.queue_timeout_seconds:
                            request.future.set_exception(
                                TimeoutError("Would timeout waiting for rate limit")
                            )
                            self._stats.total_timeouts += 1
                        else:
                            # Put back in queue and wait
                            async with self._lock:
                                heapq.heappush(self._queue, request)
                                self._stats.current_queue_size = len(self._queue)
                            self._stats.rate_limit_hits += 1
                            await asyncio.sleep(min(wait_time, 1.0))
                        continue

                    # Acquire and execute
                    acquired = await self.rate_limiter.acquire(
                        request.model, request.estimated_tokens
                    )

                    if acquired:
                        wait_time_ms = (time.monotonic() - request.timestamp) * 1000
                        self._total_wait_time += wait_time_ms

                        try:
                            result = await request.callback()
                            request.future.set_result(result)
                            self.rate_limiter.record_success(request.model)
                        except Exception as e:
                            request.future.set_exception(e)
                            # Check if rate limit error
                            if "rate" in str(e).lower() and "limit" in str(e).lower():
                                self.rate_limiter.record_rate_limit_error(request.model)

                        self._stats.total_processed += 1
                        self._stats.avg_wait_time_ms = (
                            self._total_wait_time / self._stats.total_processed
                        )
                    else:
                        # Put back in queue
                        async with self._lock:
                            heapq.heappush(self._queue, request)
                            self._stats.current_queue_size = len(self._queue)
                else:
                    # Queue is empty, wait a bit
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but keep processing
                await asyncio.sleep(0.1)

    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        return self._stats

    def get_queue_position(self, request_id: str) -> int | None:
        """Get position of a request in the queue."""
        for i, req in enumerate(sorted(self._queue)):
            if req.request_id == request_id:
                return i
        return None
