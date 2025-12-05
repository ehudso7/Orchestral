"""
Event system for Orchestral.

Provides real-time event streaming and webhook notifications
for monitoring, alerting, and integration with external systems.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
import hashlib
import hmac

import httpx
import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    """Types of events emitted by Orchestral."""

    # Request lifecycle
    REQUEST_STARTED = "request.started"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"

    # Cache events
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    CACHE_STORED = "cache.stored"
    SEMANTIC_CACHE_HIT = "cache.semantic_hit"

    # Rate limiting
    RATE_LIMIT_WARNING = "rate_limit.warning"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"

    # Budget
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Model routing
    MODEL_SELECTED = "routing.model_selected"
    MODEL_FALLBACK = "routing.fallback"
    MODEL_COMPARISON = "routing.comparison"

    # Guardrails
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    CONTENT_FILTERED = "guardrail.content_filtered"
    PII_DETECTED = "guardrail.pii_detected"

    # Quality
    EVALUATION_COMPLETED = "evaluation.completed"
    QUALITY_ALERT = "quality.alert"

    # API Keys
    KEY_CREATED = "key.created"
    KEY_REVOKED = "key.revoked"
    KEY_EXPIRED = "key.expired"

    # System
    PROVIDER_HEALTHY = "system.provider_healthy"
    PROVIDER_UNHEALTHY = "system.provider_unhealthy"
    ERROR = "system.error"


@dataclass
class Event:
    """An event in the Orchestral system."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    data: dict[str, Any]
    tenant_id: str | None = None
    api_key_id: str | None = None
    trace_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "tenant_id": self.tenant_id,
            "api_key_id": self.api_key_id,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class WebhookTarget:
    """A webhook endpoint to receive events."""

    url: str
    secret: str | None = None  # For HMAC signing
    events: list[EventType] | None = None  # None = all events
    enabled: bool = True
    retry_count: int = 3
    timeout_seconds: float = 10.0
    headers: dict[str, str] = field(default_factory=dict)

    def should_receive(self, event_type: EventType) -> bool:
        """Check if this webhook should receive the event."""
        if not self.enabled:
            return False
        if self.events is None:
            return True
        return event_type in self.events

    def sign_payload(self, payload: str) -> str | None:
        """Generate HMAC signature for payload."""
        if not self.secret:
            return None
        return hmac.new(
            self.secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()


EventHandler = Callable[[Event], Awaitable[None]]


class EventEmitter:
    """
    Central event emitter for the Orchestral system.

    Supports:
    - In-process event handlers
    - Webhook delivery with retries and signing
    - Redis pub/sub for distributed systems
    - Event filtering by type and tenant
    """

    CHANNEL_PREFIX = "orch:events:"

    def __init__(
        self,
        redis_client: Any | None = None,
        webhooks: list[WebhookTarget] | None = None,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self._webhooks = webhooks or []
        self._enabled = enabled
        self._handlers: dict[EventType | None, list[EventHandler]] = {}
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def add_webhook(self, webhook: WebhookTarget) -> None:
        """Add a webhook target."""
        self._webhooks.append(webhook)

    def remove_webhook(self, url: str) -> None:
        """Remove a webhook by URL."""
        self._webhooks = [w for w in self._webhooks if w.url != url]

    def on(
        self,
        event_type: EventType | None = None,
        handler: EventHandler | None = None,
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Register an event handler.

        Can be used as a decorator:
            @emitter.on(EventType.REQUEST_COMPLETED)
            async def handle_completion(event):
                ...

        Or directly:
            emitter.on(EventType.REQUEST_COMPLETED, handler_func)
        """

        def decorator(fn: EventHandler) -> EventHandler:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(fn)
            return fn

        if handler:
            decorator(handler)
            return lambda fn: fn

        return decorator

    async def emit(
        self,
        event_type: EventType,
        data: dict[str, Any],
        tenant_id: str | None = None,
        api_key_id: str | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """
        Emit an event to all registered handlers and webhooks.

        Args:
            event_type: Type of event
            data: Event payload data
            tenant_id: Tenant identifier
            api_key_id: API key identifier
            trace_id: Trace ID for correlation
            request_id: Request ID for correlation
            metadata: Additional metadata

        Returns:
            The emitted Event object
        """
        if not self._enabled:
            return Event(
                event_id="",
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                data=data,
            )

        import uuid

        event = Event(
            event_id=uuid.uuid4().hex[:16],
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            trace_id=trace_id,
            request_id=request_id,
            metadata=metadata or {},
        )

        # Fire and forget - don't block on handlers
        asyncio.create_task(self._dispatch(event))

        return event

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all handlers."""
        # Call in-process handlers
        handlers = self._handlers.get(event.event_type, [])
        handlers.extend(self._handlers.get(None, []))  # Global handlers

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.warning(
                    "Event handler failed",
                    handler=handler.__name__,
                    event_type=event.event_type.value,
                    error=str(e),
                )

        # Publish to Redis for distributed consumers
        if self._redis:
            await self._publish_redis(event)

        # Send to webhooks
        await self._send_webhooks(event)

    async def _publish_redis(self, event: Event) -> None:
        """Publish event to Redis pub/sub."""
        try:
            loop = asyncio.get_running_loop()
            channel = f"{self.CHANNEL_PREFIX}{event.event_type.value}"
            await loop.run_in_executor(
                None,
                lambda: self._redis.publish(channel, event.to_json()),
            )
        except Exception as e:
            logger.warning("Redis publish failed", error=str(e))

    async def _send_webhooks(self, event: Event) -> None:
        """Send event to all matching webhooks."""
        matching_webhooks = [
            w for w in self._webhooks if w.should_receive(event.event_type)
        ]

        if not matching_webhooks:
            return

        client = await self._get_http_client()
        payload = event.to_json()

        for webhook in matching_webhooks:
            asyncio.create_task(
                self._send_webhook_with_retry(client, webhook, payload, event)
            )

    async def _send_webhook_with_retry(
        self,
        client: httpx.AsyncClient,
        webhook: WebhookTarget,
        payload: str,
        event: Event,
    ) -> None:
        """Send webhook with exponential backoff retry."""
        headers = {
            "Content-Type": "application/json",
            "X-Orchestral-Event": event.event_type.value,
            "X-Orchestral-Event-ID": event.event_id,
            "X-Orchestral-Timestamp": event.timestamp.isoformat(),
            **webhook.headers,
        }

        # Add signature if secret configured
        signature = webhook.sign_payload(payload)
        if signature:
            headers["X-Orchestral-Signature"] = f"sha256={signature}"

        for attempt in range(webhook.retry_count):
            try:
                response = await client.post(
                    webhook.url,
                    content=payload,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )
                if response.status_code < 400:
                    logger.debug(
                        "Webhook delivered",
                        url=webhook.url,
                        event_type=event.event_type.value,
                        status=response.status_code,
                    )
                    return
                else:
                    logger.warning(
                        "Webhook failed",
                        url=webhook.url,
                        status=response.status_code,
                        attempt=attempt + 1,
                    )
            except Exception as e:
                logger.warning(
                    "Webhook error",
                    url=webhook.url,
                    error=str(e),
                    attempt=attempt + 1,
                )

            # Exponential backoff
            if attempt < webhook.retry_count - 1:
                await asyncio.sleep(2**attempt)

        logger.error(
            "Webhook delivery failed after retries",
            url=webhook.url,
            event_type=event.event_type.value,
        )

    async def close(self) -> None:
        """Close resources."""
        if self._http_client:
            await self._http_client.aclose()


# Global event emitter
_emitter: EventEmitter | None = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter."""
    global _emitter
    if _emitter is None:
        _emitter = EventEmitter()
    return _emitter


def configure_event_emitter(
    redis_client: Any | None = None,
    webhooks: list[WebhookTarget] | None = None,
    enabled: bool = True,
) -> EventEmitter:
    """Configure the global event emitter."""
    global _emitter
    _emitter = EventEmitter(
        redis_client=redis_client,
        webhooks=webhooks,
        enabled=enabled,
    )
    return _emitter
