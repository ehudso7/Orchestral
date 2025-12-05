"""
Compliance and audit logging for Orchestral.

Provides immutable audit trails for:
- All API requests and responses
- Authentication events
- Configuration changes
- Data access patterns

Designed for SOC2, HIPAA, and GDPR compliance requirements.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AuditAction(str, Enum):
    """Types of auditable actions."""

    # Authentication
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    KEY_CREATED = "key.created"
    KEY_REVOKED = "key.revoked"
    KEY_USED = "key.used"

    # Data operations
    REQUEST_RECEIVED = "request.received"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"

    # Model operations
    MODEL_CALLED = "model.called"
    MODEL_RESPONSE = "model.response"
    MODEL_ERROR = "model.error"

    # Cache operations
    CACHE_HIT = "cache.hit"
    CACHE_STORE = "cache.store"

    # Admin operations
    CONFIG_CHANGED = "config.changed"
    RATE_LIMIT_APPLIED = "rate_limit.applied"
    BUDGET_LIMIT_APPLIED = "budget.limit_applied"

    # Safety
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    CONTENT_FILTERED = "content.filtered"
    PII_DETECTED = "pii.detected"

    # Export/access
    DATA_EXPORTED = "data.exported"
    DATA_DELETED = "data.deleted"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """An immutable audit log entry."""

    entry_id: str
    timestamp: datetime
    action: AuditAction
    severity: AuditSeverity

    # Actor information
    actor_type: str  # "user", "system", "api_key", "admin"
    actor_id: str | None
    actor_ip: str | None = None

    # Resource information
    resource_type: str | None = None  # "request", "key", "config", etc.
    resource_id: str | None = None

    # Context
    tenant_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None

    # Action details (non-sensitive)
    details: dict[str, Any] = field(default_factory=dict)

    # Integrity
    checksum: str = ""  # SHA-256 of entry content

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate integrity checksum."""
        content = json.dumps({
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "details": self.details,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify entry has not been tampered with."""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "severity": self.severity.value,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "actor_ip": self.actor_ip,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "details": self.details,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        entry = cls(
            entry_id=data["entry_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=AuditAction(data["action"]),
            severity=AuditSeverity(data.get("severity", "info")),
            actor_type=data["actor_type"],
            actor_id=data.get("actor_id"),
            actor_ip=data.get("actor_ip"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            tenant_id=data.get("tenant_id"),
            request_id=data.get("request_id"),
            trace_id=data.get("trace_id"),
            details=data.get("details", {}),
            checksum=data.get("checksum", ""),
        )
        return entry


class AuditLogger:
    """
    Immutable audit logger for compliance.

    Features:
    - Tamper-evident logging with checksums
    - Structured log format
    - Retention policy support
    - Query and export capabilities
    """

    AUDIT_PREFIX = "orch:audit:"
    AUDIT_INDEX = "orch:audit:idx:"

    def __init__(
        self,
        redis_client: Any | None = None,
        retention_days: int = 90,
        enabled: bool = True,
        log_to_console: bool = True,
    ):
        self._redis = redis_client
        self._retention_days = retention_days
        self._enabled = enabled
        self._log_to_console = log_to_console

        # Local buffer for batching
        self._buffer: list[AuditEntry] = []
        self._buffer_size = 100

    async def log(
        self,
        action: AuditAction,
        actor_type: str,
        actor_id: str | None = None,
        actor_ip: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        tenant_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> AuditEntry:
        """
        Create an audit log entry.

        Args:
            action: The action being logged
            actor_type: Type of actor (user, system, api_key, admin)
            actor_id: ID of the actor
            actor_ip: IP address of actor
            resource_type: Type of resource being acted upon
            resource_id: ID of the resource
            tenant_id: Tenant ID for multi-tenant isolation
            request_id: Request ID for correlation
            trace_id: Trace ID for distributed tracing
            details: Additional details (must not contain sensitive data)
            severity: Severity level

        Returns:
            The created AuditEntry
        """
        import uuid

        entry = AuditEntry(
            entry_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            action=action,
            severity=severity,
            actor_type=actor_type,
            actor_id=actor_id,
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            tenant_id=tenant_id,
            request_id=request_id,
            trace_id=trace_id,
            details=details or {},
        )

        # Log to console
        if self._log_to_console:
            logger.info(
                "AUDIT",
                action=action.value,
                actor_type=actor_type,
                actor_id=actor_id,
                resource_type=resource_type,
                resource_id=resource_id,
                tenant_id=tenant_id,
                severity=severity.value,
            )

        # Store entry
        if self._enabled:
            await self._store_entry(entry)

        return entry

    async def _store_entry(self, entry: AuditEntry) -> None:
        """Store an audit entry."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                # Store entry
                key = f"{self.AUDIT_PREFIX}{entry.entry_id}"
                ttl = self._retention_days * 86400
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.setex(key, ttl, json.dumps(entry.to_dict())),
                )

                # Index by tenant
                if entry.tenant_id:
                    index_key = f"{self.AUDIT_INDEX}tenant:{entry.tenant_id}"
                    await loop.run_in_executor(
                        None,
                        lambda: self._redis.zadd(
                            index_key,
                            {entry.entry_id: entry.timestamp.timestamp()},
                        ),
                    )

                # Index by action
                action_key = f"{self.AUDIT_INDEX}action:{entry.action.value}"
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.zadd(
                        action_key,
                        {entry.entry_id: entry.timestamp.timestamp()},
                    ),
                )

                # Index by date for efficient queries
                date_key = f"{self.AUDIT_INDEX}date:{entry.timestamp.strftime('%Y-%m-%d')}"
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.zadd(
                        date_key,
                        {entry.entry_id: entry.timestamp.timestamp()},
                    ),
                )

            except Exception as e:
                logger.error("Failed to store audit entry", error=str(e))
        else:
            self._buffer.append(entry)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size:]

    async def query(
        self,
        tenant_id: str | None = None,
        action: AuditAction | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        actor_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Query audit entries.

        Args:
            tenant_id: Filter by tenant
            action: Filter by action type
            start_time: Start of time range
            end_time: End of time range
            actor_id: Filter by actor
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        entries = []

        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                # Determine which index to use
                if tenant_id:
                    index_key = f"{self.AUDIT_INDEX}tenant:{tenant_id}"
                elif action:
                    index_key = f"{self.AUDIT_INDEX}action:{action.value}"
                else:
                    # Query by date range
                    now = datetime.now(timezone.utc)
                    date_key = f"{self.AUDIT_INDEX}date:{now.strftime('%Y-%m-%d')}"
                    index_key = date_key

                # Get entry IDs from index
                min_score = start_time.timestamp() if start_time else "-inf"
                max_score = end_time.timestamp() if end_time else "+inf"

                entry_ids = await loop.run_in_executor(
                    None,
                    lambda: self._redis.zrangebyscore(
                        index_key, min_score, max_score, start=0, num=limit
                    ),
                )

                # Fetch entries
                for entry_id in entry_ids:
                    eid = entry_id.decode("utf-8") if isinstance(entry_id, bytes) else entry_id
                    data = await loop.run_in_executor(
                        None, self._redis.get, f"{self.AUDIT_PREFIX}{eid}"
                    )
                    if data:
                        entry = AuditEntry.from_dict(json.loads(data))
                        # Apply additional filters
                        if actor_id and entry.actor_id != actor_id:
                            continue
                        if action and entry.action != action:
                            continue
                        entries.append(entry)

            except Exception as e:
                logger.error("Failed to query audit entries", error=str(e))
        else:
            # Query local buffer
            for entry in reversed(self._buffer):
                if tenant_id and entry.tenant_id != tenant_id:
                    continue
                if action and entry.action != action:
                    continue
                if actor_id and entry.actor_id != actor_id:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                entries.append(entry)
                if len(entries) >= limit:
                    break

        return entries

    async def export(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
    ) -> str:
        """
        Export audit entries for compliance reporting.

        Args:
            tenant_id: Tenant to export
            start_time: Start of export range
            end_time: End of export range
            format: Export format (json, csv)

        Returns:
            Exported data as string
        """
        entries = await self.query(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        # Log the export action
        await self.log(
            action=AuditAction.DATA_EXPORTED,
            actor_type="admin",
            resource_type="audit_log",
            tenant_id=tenant_id,
            details={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "entry_count": len(entries),
                "format": format,
            },
        )

        if format == "csv":
            lines = ["entry_id,timestamp,action,actor_type,actor_id,resource_type,resource_id"]
            for entry in entries:
                lines.append(
                    f"{entry.entry_id},{entry.timestamp.isoformat()},{entry.action.value},"
                    f"{entry.actor_type},{entry.actor_id or ''},{entry.resource_type or ''},"
                    f"{entry.resource_id or ''}"
                )
            return "\n".join(lines)
        else:
            return json.dumps([e.to_dict() for e in entries], indent=2)


# Global audit logger
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def configure_audit_logger(
    redis_client: Any | None = None,
    retention_days: int = 90,
    enabled: bool = True,
) -> AuditLogger:
    """Configure the global audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(
        redis_client=redis_client,
        retention_days=retention_days,
        enabled=enabled,
    )
    return _audit_logger
