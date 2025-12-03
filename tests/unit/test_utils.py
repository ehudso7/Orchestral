"""Tests for utility modules."""

import pytest
from unittest.mock import AsyncMock, patch
import asyncio

from orchestral.utils.metrics import Metrics, RequestMetrics
from orchestral.utils.retry import (
    RetryConfig,
    calculate_delay,
    with_retry,
    retry_async,
    RetryContext,
)
from orchestral.providers.base import RateLimitError, ProviderError, ModelProvider


class TestMetrics:
    """Tests for Metrics class."""

    def test_record_completion(self):
        metrics = Metrics()
        metrics.record_completion(
            provider="openai",
            model="gpt-4o",
            latency_ms=100.0,
            input_tokens=50,
            output_tokens=100,
        )

        summary = metrics.get_summary()
        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert "openai" in summary["providers"]
        assert summary["providers"]["openai"]["total_requests"] == 1

    def test_record_error(self):
        metrics = Metrics()
        metrics.record_error(
            model="gpt-4o",
            error_type="RateLimitError",
            provider="openai",
        )

        summary = metrics.get_summary()
        assert summary["failed_requests"] == 1
        assert summary["providers"]["openai"]["errors"]["RateLimitError"] == 1

    def test_success_rate(self):
        metrics = Metrics()

        # Record 3 successes and 1 failure
        for _ in range(3):
            metrics.record_completion("openai", "gpt-4o", 100.0, 50, 100)
        metrics.record_error("gpt-4o", "Error", "openai")

        summary = metrics.get_summary()
        assert summary["success_rate"] == 0.75

    def test_recent_requests(self):
        metrics = Metrics()
        for i in range(5):
            metrics.record_completion("openai", "gpt-4o", 100.0 + i, 50, 100)

        recent = metrics.get_recent_requests(limit=3)
        assert len(recent) == 3

    def test_reset(self):
        metrics = Metrics()
        metrics.record_completion("openai", "gpt-4o", 100.0, 50, 100)
        metrics.reset()

        summary = metrics.get_summary()
        assert summary["total_requests"] == 0


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        delay0 = calculate_delay(0, config)
        delay1 = calculate_delay(1, config)
        delay2 = calculate_delay(2, config)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        config = RetryConfig(base_delay=10.0, max_delay=30.0, jitter=False)

        delay = calculate_delay(5, config)  # Would be 320 without cap
        assert delay == 30.0

    def test_retry_after_override(self):
        config = RetryConfig(base_delay=1.0, jitter=False)

        delay = calculate_delay(0, config, retry_after=60.0)
        assert delay == 60.0

    def test_jitter_adds_variance(self):
        config = RetryConfig(base_delay=1.0, jitter=True)

        delays = [calculate_delay(1, config) for _ in range(10)]
        # With jitter, not all delays should be the same
        assert len(set(delays)) > 1


class TestWithRetry:
    """Tests for with_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", ModelProvider.OPENAI)
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        @with_retry(RetryConfig(max_retries=2, base_delay=0.01))
        async def always_fails():
            raise RateLimitError("Rate limited", ModelProvider.OPENAI)

        with pytest.raises(RateLimitError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        call_count = 0

        @with_retry(RetryConfig(max_retries=3))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raises_value_error()

        # Should not retry for ValueError
        assert call_count == 1


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        async def my_func(x: int) -> int:
            return x * 2

        result = await retry_async(my_func, 5)
        assert result == 10


class TestRetryContext:
    """Tests for RetryContext context manager."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        async with RetryContext(RetryConfig(max_retries=3)) as ctx:
            while ctx.should_retry():
                ctx.success()
                break

        assert ctx.attempt == 0
        assert ctx._succeeded is True

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        attempt = 0
        async with RetryContext(RetryConfig(max_retries=3, base_delay=0.01)) as ctx:
            while ctx.should_retry():
                attempt += 1
                try:
                    if attempt < 3:
                        raise RateLimitError("Rate limited", ModelProvider.OPENAI)
                    ctx.success()
                    break
                except RateLimitError as e:
                    await ctx.handle_error(e)

        assert ctx._succeeded is True
        assert ctx.attempt == 2
