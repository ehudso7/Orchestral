"""
Usage tracking and metering for Orchestral.

Provides detailed usage tracking per API key for billing and analytics.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from orchestral.core.models import MODEL_REGISTRY

logger = structlog.get_logger()


@dataclass
class UsageRecord:
    """A single usage record."""

    timestamp: datetime
    key_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    request_id: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "key_id": self.key_id,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "request_id": self.request_id,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            key_id=data["key_id"],
            model=data["model"],
            provider=data["provider"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            latency_ms=data["latency_ms"],
            cost_usd=data["cost_usd"],
            request_id=data["request_id"],
            success=data.get("success", True),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UsageSummary:
    """Aggregated usage summary."""

    key_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    models_used: dict[str, int] = field(default_factory=dict)
    providers_used: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "models_used": self.models_used,
            "providers_used": self.providers_used,
        }


class UsageTracker:
    """
    Tracks and aggregates API usage for billing.

    Stores detailed usage records and provides aggregation for billing.
    """

    USAGE_PREFIX = "orch:usage:"
    DAILY_PREFIX = "orch:daily:"
    MONTHLY_PREFIX = "orch:monthly:"

    def __init__(self, redis_client: Any | None = None):
        """
        Initialize the usage tracker.

        Args:
            redis_client: Redis client for persistence
        """
        self._redis = redis_client
        self._local_records: list[UsageRecord] = []
        self._local_daily: dict[str, UsageSummary] = {}
        self._local_monthly: dict[str, UsageSummary] = {}

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate the cost for a request.

        Args:
            model: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        spec = MODEL_REGISTRY.get(model)
        if not spec:
            # Default pricing if model not found
            return (input_tokens * 0.01 + output_tokens * 0.03) / 1_000_000

        input_cost = (input_tokens * spec.input_cost_per_million) / 1_000_000
        output_cost = (output_tokens * spec.output_cost_per_million) / 1_000_000

        return input_cost + output_cost

    async def record(
        self,
        key_id: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        request_id: str,
        success: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """
        Record a usage event.

        Args:
            key_id: API key ID
            model: Model used
            provider: Provider name
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Request latency
            request_id: Unique request ID
            success: Whether request succeeded
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            The created UsageRecord
        """
        cost_usd = self.calculate_cost(model, input_tokens, output_tokens) if success else 0.0

        record = UsageRecord(
            timestamp=datetime.now(timezone.utc),
            key_id=key_id,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            request_id=request_id,
            success=success,
            error=error,
            metadata=metadata or {},
        )

        # Store the record
        await self._store_record(record)

        # Update daily/monthly aggregates
        await self._update_aggregates(record)

        logger.debug(
            "Usage recorded",
            key_id=key_id,
            model=model,
            tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
        )

        return record

    async def get_usage(
        self,
        key_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[UsageRecord]:
        """
        Get usage records for a key.

        Args:
            key_id: API key ID
            start_time: Start of period
            end_time: End of period
            limit: Maximum records to return

        Returns:
            List of usage records
        """
        if self._redis:
            loop = asyncio.get_event_loop()
            key = f"{self.USAGE_PREFIX}{key_id}"

            # Get records from sorted set (in executor to avoid blocking)
            def _get_data():
                if start_time and end_time:
                    return self._redis.zrangebyscore(
                        key,
                        start_time.timestamp(),
                        end_time.timestamp(),
                        start=0,
                        num=limit,
                    )
                else:
                    return self._redis.zrevrange(key, 0, limit - 1)

            data = await loop.run_in_executor(None, _get_data)

            records = []
            for item in data:
                record_data = json.loads(item)
                records.append(UsageRecord.from_dict(record_data))

            return records
        else:
            # Filter local records
            records = [r for r in self._local_records if r.key_id == key_id]
            if start_time:
                records = [r for r in records if r.timestamp >= start_time]
            if end_time:
                records = [r for r in records if r.timestamp <= end_time]
            return records[-limit:]

    async def get_daily_summary(
        self,
        key_id: str,
        date: datetime | None = None,
    ) -> UsageSummary:
        """
        Get daily usage summary.

        Args:
            key_id: API key ID
            date: Date to get summary for (defaults to today)

        Returns:
            UsageSummary for the day
        """
        date = date or datetime.now(timezone.utc)
        date_key = date.strftime("%Y-%m-%d")
        full_key = f"{key_id}:{date_key}"

        # Calculate period bounds
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        if self._redis:
            loop = asyncio.get_event_loop()
            key = f"{self.DAILY_PREFIX}{full_key}"
            # Use hgetall since we store with hincrby/hincrbyfloat
            data = await loop.run_in_executor(None, self._redis.hgetall, key)
            if data:
                return self._parse_redis_aggregate(key_id, data, start, end)

        # Return from local or empty
        return self._local_daily.get(full_key, UsageSummary(
            key_id=key_id,
            period_start=start,
            period_end=end,
        ))

    async def get_monthly_summary(
        self,
        key_id: str,
        year: int | None = None,
        month: int | None = None,
    ) -> UsageSummary:
        """
        Get monthly usage summary.

        Args:
            key_id: API key ID
            year: Year (defaults to current)
            month: Month (defaults to current)

        Returns:
            UsageSummary for the month
        """
        now = datetime.now(timezone.utc)
        year = year or now.year
        month = month or now.month
        month_key = f"{year:04d}-{month:02d}"
        full_key = f"{key_id}:{month_key}"

        # Calculate period bounds
        from calendar import monthrange
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        last_day = monthrange(year, month)[1]
        end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        if self._redis:
            loop = asyncio.get_event_loop()
            key = f"{self.MONTHLY_PREFIX}{full_key}"
            # Use hgetall since we store with hincrby/hincrbyfloat
            data = await loop.run_in_executor(None, self._redis.hgetall, key)
            if data:
                return self._parse_redis_aggregate(key_id, data, start, end)

        # Return from local or empty
        return self._local_monthly.get(full_key, UsageSummary(
            key_id=key_id,
            period_start=start,
            period_end=end,
        ))

    def _parse_redis_aggregate(
        self,
        key_id: str,
        data: dict[bytes | str, bytes | str],
        period_start: datetime,
        period_end: datetime,
    ) -> UsageSummary:
        """
        Parse Redis hash aggregate data into UsageSummary.

        Args:
            key_id: API key ID
            data: Raw Redis hash data
            period_start: Start of the period
            period_end: End of the period

        Returns:
            UsageSummary with parsed data
        """
        # Helper to decode bytes and convert to proper types
        def get_int(key: str) -> int:
            val = data.get(key) or data.get(key.encode(), 0)
            if isinstance(val, bytes):
                val = val.decode()
            return int(val) if val else 0

        def get_float(key: str) -> float:
            val = data.get(key) or data.get(key.encode(), 0.0)
            if isinstance(val, bytes):
                val = val.decode()
            return float(val) if val else 0.0

        # Extract models and providers from prefixed keys
        models_used: dict[str, int] = {}
        providers_used: dict[str, int] = {}

        for k, v in data.items():
            key_str = k.decode() if isinstance(k, bytes) else k
            val_str = v.decode() if isinstance(v, bytes) else v

            if key_str.startswith("model:"):
                model_name = key_str[6:]  # Remove "model:" prefix
                models_used[model_name] = int(val_str)
            elif key_str.startswith("provider:"):
                provider_name = key_str[9:]  # Remove "provider:" prefix
                providers_used[provider_name] = int(val_str)

        total_requests = get_int("total_requests")
        total_latency = get_float("total_latency_ms")

        return UsageSummary(
            key_id=key_id,
            period_start=period_start,
            period_end=period_end,
            total_requests=total_requests,
            successful_requests=get_int("successful_requests"),
            failed_requests=get_int("failed_requests"),
            total_input_tokens=get_int("total_input_tokens"),
            total_output_tokens=get_int("total_output_tokens"),
            total_cost_usd=get_float("total_cost_usd"),
            avg_latency_ms=total_latency / total_requests if total_requests > 0 else 0.0,
            models_used=models_used,
            providers_used=providers_used,
        )

    async def check_budget(
        self,
        key_id: str,
        budget_usd: float,
    ) -> tuple[bool, float, float]:
        """
        Check if a key is within budget.

        Args:
            key_id: API key ID
            budget_usd: Monthly budget in USD

        Returns:
            Tuple of (within_budget, current_spend, remaining)
        """
        summary = await self.get_monthly_summary(key_id)
        current_spend = summary.total_cost_usd
        remaining = budget_usd - current_spend
        within_budget = remaining > 0

        return within_budget, current_spend, remaining

    async def _store_record(self, record: UsageRecord) -> None:
        """Store a usage record."""
        if self._redis:
            loop = asyncio.get_event_loop()
            key = f"{self.USAGE_PREFIX}{record.key_id}"

            def _store():
                import time as time_module
                self._redis.zadd(
                    key,
                    {json.dumps(record.to_dict()): record.timestamp.timestamp()},
                )
                # Keep only last 30 days of detailed records
                cutoff = time_module.time() - (30 * 24 * 60 * 60)
                self._redis.zremrangebyscore(key, 0, cutoff)

            await loop.run_in_executor(None, _store)
        else:
            self._local_records.append(record)
            # Keep only last 10000 records in memory
            if len(self._local_records) > 10000:
                self._local_records = self._local_records[-10000:]

    async def _update_aggregates(self, record: UsageRecord) -> None:
        """Update daily and monthly aggregates."""
        date_key = record.timestamp.strftime("%Y-%m-%d")
        month_key = record.timestamp.strftime("%Y-%m")

        daily_key = f"{record.key_id}:{date_key}"
        monthly_key = f"{record.key_id}:{month_key}"

        if self._redis:
            # Update daily aggregate
            await self._update_redis_aggregate(
                f"{self.DAILY_PREFIX}{daily_key}",
                record,
                expire_days=90,
            )
            # Update monthly aggregate
            await self._update_redis_aggregate(
                f"{self.MONTHLY_PREFIX}{monthly_key}",
                record,
                expire_days=400,
            )
        else:
            # Update local aggregates
            self._update_local_aggregate(self._local_daily, daily_key, record)
            self._update_local_aggregate(self._local_monthly, monthly_key, record)

    async def _update_redis_aggregate(
        self,
        key: str,
        record: UsageRecord,
        expire_days: int,
    ) -> None:
        """Update a Redis aggregate."""
        loop = asyncio.get_event_loop()

        def _execute_pipeline():
            pipe = self._redis.pipeline()
            pipe.hincrby(key, "total_requests", 1)
            if record.success:
                pipe.hincrby(key, "successful_requests", 1)
            else:
                pipe.hincrby(key, "failed_requests", 1)
            pipe.hincrby(key, "total_input_tokens", record.input_tokens)
            pipe.hincrby(key, "total_output_tokens", record.output_tokens)
            pipe.hincrbyfloat(key, "total_cost_usd", record.cost_usd)
            pipe.hincrbyfloat(key, "total_latency_ms", record.latency_ms)
            pipe.hincrby(key, f"model:{record.model}", 1)
            pipe.hincrby(key, f"provider:{record.provider}", 1)
            pipe.expire(key, expire_days * 24 * 60 * 60)
            pipe.execute()

        await loop.run_in_executor(None, _execute_pipeline)

    def _update_local_aggregate(
        self,
        store: dict[str, UsageSummary],
        key: str,
        record: UsageRecord,
    ) -> None:
        """Update a local aggregate."""
        if key not in store:
            store[key] = UsageSummary(
                key_id=record.key_id,
                period_start=record.timestamp.replace(hour=0, minute=0, second=0),
                period_end=record.timestamp.replace(hour=23, minute=59, second=59),
            )

        summary = store[key]
        summary.total_requests += 1
        if record.success:
            summary.successful_requests += 1
        else:
            summary.failed_requests += 1
        summary.total_input_tokens += record.input_tokens
        summary.total_output_tokens += record.output_tokens
        summary.total_cost_usd += record.cost_usd

        # Update average latency
        total_latency = summary.avg_latency_ms * (summary.total_requests - 1) + record.latency_ms
        summary.avg_latency_ms = total_latency / summary.total_requests

        # Update model/provider counts
        summary.models_used[record.model] = summary.models_used.get(record.model, 0) + 1
        summary.providers_used[record.provider] = summary.providers_used.get(record.provider, 0) + 1
