"""
Retry utilities with exponential backoff.

Provides robust retry logic for handling transient failures.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar, Awaitable

import structlog

from orchestral.providers.base import ProviderError, RateLimitError

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        RateLimitError,
        ConnectionError,
        TimeoutError,
    )


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: float | None = None,
) -> float:
    """
    Calculate delay before next retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        retry_after: Optional server-specified delay

    Returns:
        Delay in seconds
    """
    if retry_after:
        delay = retry_after
    else:
        delay = config.base_delay * (config.exponential_base ** attempt)

    delay = min(delay, config.max_delay)

    if config.jitter:
        delay = delay * (0.5 + random.random())

    return delay


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration (uses defaults if not provided)

    Returns:
        Decorated function with retry logic
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(
                            "Max retries exceeded",
                            function=func.__name__,
                            attempts=attempt + 1,
                            error=str(e),
                        )
                        raise

                    # Get retry-after from rate limit errors
                    retry_after = None
                    if isinstance(e, RateLimitError):
                        retry_after = e.retry_after

                    delay = calculate_delay(attempt, config, retry_after)

                    logger.warning(
                        "Retrying after error",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_retries=config.max_retries,
                        delay=delay,
                        error=str(e),
                    )

                    await asyncio.sleep(delay)

                except Exception:
                    # Non-retryable exception
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments
        config: Retry configuration
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    config = config or RetryConfig()
    decorated = with_retry(config)(func)
    return await decorated(*args, **kwargs)


class RetryContext:
    """
    Context manager for retry logic.

    Usage:
        async with RetryContext(config) as ctx:
            while ctx.should_retry():
                try:
                    result = await some_operation()
                    ctx.success()
                    break
                except SomeError as e:
                    await ctx.handle_error(e)
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self.attempt = 0
        self._succeeded = False
        self._last_error: Exception | None = None

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_val and not self._succeeded:
            logger.error(
                "Retry context failed",
                attempts=self.attempt,
                error=str(exc_val),
            )
        return False

    def should_retry(self) -> bool:
        """Check if another retry attempt should be made."""
        return self.attempt <= self.config.max_retries and not self._succeeded

    def success(self) -> None:
        """Mark the operation as successful."""
        self._succeeded = True

    async def handle_error(self, error: Exception) -> None:
        """
        Handle an error and wait before retry.

        Args:
            error: The exception that occurred

        Raises:
            The error if max retries exceeded or not retryable
        """
        self._last_error = error

        if not isinstance(error, self.config.retryable_exceptions):
            raise error

        if self.attempt >= self.config.max_retries:
            raise error

        retry_after = None
        if isinstance(error, RateLimitError):
            retry_after = error.retry_after

        delay = calculate_delay(self.attempt, self.config, retry_after)

        logger.warning(
            "Retry after error",
            attempt=self.attempt + 1,
            delay=delay,
            error=str(error),
        )

        await asyncio.sleep(delay)
        self.attempt += 1
