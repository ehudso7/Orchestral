"""
Distributed tracing for LLM operations.

Provides OpenTelemetry-compatible tracing with LLM-specific attributes,
enabling deep observability into model calls, routing decisions, and performance.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Iterator

import structlog

logger = structlog.get_logger()


class SpanKind(str, Enum):
    """Types of spans in the tracing system."""

    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_STORE = "cache_store"
    EMBEDDING = "embedding"
    ROUTING = "routing"
    RATE_LIMIT = "rate_limit"
    GUARDRAIL = "guardrail"
    EVALUATION = "evaluation"
    TOOL_CALL = "tool_call"
    PROMPT_RENDER = "prompt_render"
    RETRY = "retry"
    FALLBACK = "fallback"


@dataclass
class SpanAttributes:
    """Standard attributes for LLM spans."""

    # Request attributes
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    # Response attributes
    finish_reason: str | None = None
    response_id: str | None = None

    # Cost attributes
    cost_usd: float | None = None
    cached: bool = False
    cache_hit_type: str | None = None  # "exact" or "semantic"

    # Quality attributes
    latency_ms: float | None = None
    ttft_ms: float | None = None  # Time to first token
    tokens_per_second: float | None = None

    # Error attributes
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0

    # Routing attributes
    routing_strategy: str | None = None
    selected_model: str | None = None
    fallback_triggered: bool = False

    # Guardrail attributes
    guardrail_triggered: bool = False
    guardrail_action: str | None = None

    # Custom attributes
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and key != "custom":
                result[key] = value
        result.update(self.custom)
        return result


@dataclass
class Span:
    """A span represents a single operation in a trace."""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: datetime | None = None
    attributes: SpanAttributes = field(default_factory=SpanAttributes)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # "ok", "error", "unset"

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to this span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_error(self, error_type: str, message: str) -> None:
        """Mark span as errored."""
        self.status = "error"
        self.attributes.error_type = error_type
        self.attributes.error_message = message

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now(timezone.utc)
        if self.attributes.latency_ms is None and self.end_time and self.start_time:
            self.attributes.latency_ms = (
                self.end_time.timestamp() - self.start_time.timestamp()
            ) * 1000

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time.timestamp() - self.start_time.timestamp()) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/export."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes.to_dict(),
            "events": self.events,
            "status": self.status,
        }


@dataclass
class Trace:
    """A trace represents a complete request through the system."""

    trace_id: str
    root_span_id: str
    name: str
    start_time: datetime
    end_time: datetime | None = None
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Request context
    tenant_id: str | None = None
    api_key_id: str | None = None
    request_id: str | None = None
    user_id: str | None = None

    # Aggregated metrics
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    model_calls: int = 0
    cache_hits: int = 0

    def add_span(self, span: Span) -> None:
        """Add a span to this trace."""
        self.spans.append(span)

        # Update aggregates
        if span.attributes.total_tokens:
            self.total_tokens += span.attributes.total_tokens
        if span.attributes.cost_usd:
            self.total_cost_usd += span.attributes.cost_usd
        if span.kind == SpanKind.LLM_REQUEST:
            self.model_calls += 1
        if span.attributes.cached:
            self.cache_hits += 1

    def end(self) -> None:
        """End the trace."""
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.total_latency_ms = (
                self.end_time.timestamp() - self.start_time.timestamp()
            ) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/export."""
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "tenant_id": self.tenant_id,
            "api_key_id": self.api_key_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
            "spans": [s.to_dict() for s in self.spans],
        }


class TraceContext:
    """Context holder for the current trace and span."""

    def __init__(self):
        self._trace: Trace | None = None
        self._current_span: Span | None = None
        self._span_stack: list[Span] = []

    @property
    def trace(self) -> Trace | None:
        return self._trace

    @property
    def current_span(self) -> Span | None:
        return self._current_span

    def set_trace(self, trace: Trace) -> None:
        self._trace = trace

    def push_span(self, span: Span) -> None:
        if self._current_span:
            self._span_stack.append(self._current_span)
        self._current_span = span

    def pop_span(self) -> Span | None:
        ended_span = self._current_span
        self._current_span = self._span_stack.pop() if self._span_stack else None
        return ended_span


# Thread-local context storage
import contextvars

_trace_context: contextvars.ContextVar[TraceContext] = contextvars.ContextVar(
    "trace_context", default=None
)


def get_current_context() -> TraceContext:
    """Get the current trace context, creating one if needed."""
    ctx = _trace_context.get()
    if ctx is None:
        ctx = TraceContext()
        _trace_context.set(ctx)
    return ctx


class TraceExporter:
    """Base class for trace exporters."""

    async def export(self, trace: Trace) -> None:
        """Export a completed trace."""
        raise NotImplementedError


class RedisTraceExporter(TraceExporter):
    """Export traces to Redis for storage and querying."""

    TRACE_PREFIX = "orch:trace:"
    TRACE_INDEX = "orch:traces:index:"

    def __init__(self, redis_client: Any, ttl_seconds: int = 86400 * 7):
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def export(self, trace: Trace) -> None:
        """Export trace to Redis."""
        if not self._redis:
            return

        loop = asyncio.get_running_loop()
        try:
            trace_data = json.dumps(trace.to_dict())
            key = f"{self.TRACE_PREFIX}{trace.trace_id}"

            await loop.run_in_executor(
                None, lambda: self._redis.setex(key, self._ttl, trace_data)
            )

            # Index by tenant
            if trace.tenant_id:
                index_key = f"{self.TRACE_INDEX}{trace.tenant_id}"
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.zadd(
                        index_key,
                        {trace.trace_id: trace.start_time.timestamp()},
                    ),
                )
                # Trim old entries
                await loop.run_in_executor(
                    None, lambda: self._redis.zremrangebyrank(index_key, 0, -10001)
                )

            logger.debug("Trace exported", trace_id=trace.trace_id)
        except Exception as e:
            logger.warning("Failed to export trace", error=str(e))


class ConsoleTraceExporter(TraceExporter):
    """Export traces to console/logs for debugging."""

    async def export(self, trace: Trace) -> None:
        """Log trace summary."""
        logger.info(
            "Trace completed",
            trace_id=trace.trace_id,
            name=trace.name,
            duration_ms=trace.total_latency_ms,
            model_calls=trace.model_calls,
            total_tokens=trace.total_tokens,
            total_cost_usd=trace.total_cost_usd,
            cache_hits=trace.cache_hits,
            spans=len(trace.spans),
        )


class Tracer:
    """
    Main tracer class for creating and managing traces.

    Usage:
        tracer = Tracer()

        async with tracer.trace("completion_request") as trace:
            async with tracer.span("cache_lookup", SpanKind.CACHE_LOOKUP) as span:
                # ... cache lookup logic
                span.attributes.cached = True

            async with tracer.span("llm_call", SpanKind.LLM_REQUEST) as span:
                # ... LLM call logic
                span.attributes.model = "gpt-4o"
                span.attributes.total_tokens = 150
    """

    def __init__(
        self,
        service_name: str = "orchestral",
        exporters: list[TraceExporter] | None = None,
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        self.service_name = service_name
        self._exporters = exporters or [ConsoleTraceExporter()]
        self._enabled = enabled
        self._sample_rate = sample_rate

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random

        return random.random() < self._sample_rate

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return uuid.uuid4().hex[:16]

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        tenant_id: str | None = None,
        api_key_id: str | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[Trace]:
        """Create a new trace context."""
        if not self._enabled or not self._should_sample():
            # Return a dummy trace that does nothing
            yield Trace(
                trace_id="",
                root_span_id="",
                name=name,
                start_time=datetime.now(timezone.utc),
            )
            return

        trace_id = self._generate_id()
        root_span_id = self._generate_id()

        trace = Trace(
            trace_id=trace_id,
            root_span_id=root_span_id,
            name=name,
            start_time=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            request_id=request_id or self._generate_id(),
            user_id=user_id,
            metadata=metadata or {},
        )

        ctx = get_current_context()
        ctx.set_trace(trace)

        try:
            yield trace
        finally:
            trace.end()
            # Export trace
            for exporter in self._exporters:
                try:
                    await exporter.export(trace)
                except Exception as e:
                    logger.warning(
                        "Trace export failed",
                        exporter=type(exporter).__name__,
                        error=str(e),
                    )

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: SpanKind,
        attributes: SpanAttributes | None = None,
    ) -> AsyncIterator[Span]:
        """Create a new span within the current trace."""
        ctx = get_current_context()
        trace = ctx.trace

        if not self._enabled or trace is None or not trace.trace_id:
            # Return a dummy span
            yield Span(
                span_id="",
                trace_id="",
                parent_span_id=None,
                name=name,
                kind=kind,
                start_time=datetime.now(timezone.utc),
            )
            return

        span_id = self._generate_id()
        parent_span_id = ctx.current_span.span_id if ctx.current_span else None

        span = Span(
            span_id=span_id,
            trace_id=trace.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=datetime.now(timezone.utc),
            attributes=attributes or SpanAttributes(),
        )

        ctx.push_span(span)

        try:
            yield span
        except Exception as e:
            span.set_error(type(e).__name__, str(e))
            raise
        finally:
            span.end()
            ctx.pop_span()
            trace.add_span(span)


# Global tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def configure_tracer(
    service_name: str = "orchestral",
    exporters: list[TraceExporter] | None = None,
    enabled: bool = True,
    sample_rate: float = 1.0,
) -> Tracer:
    """Configure the global tracer."""
    global _tracer
    _tracer = Tracer(
        service_name=service_name,
        exporters=exporters,
        enabled=enabled,
        sample_rate=sample_rate,
    )
    return _tracer
