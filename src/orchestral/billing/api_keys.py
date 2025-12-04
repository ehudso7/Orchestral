"""
API Key management for Orchestral.

Provides secure API key generation, validation, and tier management.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class KeyTier(str, Enum):
    """API key tier levels with different limits and pricing."""

    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Rate and usage limits for a tier."""

    requests_per_minute: int
    requests_per_day: int
    tokens_per_month: int
    max_concurrent_requests: int
    monthly_budget_usd: float
    allowed_models: list[str] | None = None  # None means all models
    priority: int = 0  # Higher = more priority in queue


# Default tier configurations
TIER_LIMITS: dict[KeyTier, TierLimits] = {
    KeyTier.FREE: TierLimits(
        requests_per_minute=10,
        requests_per_day=100,
        tokens_per_month=100_000,
        max_concurrent_requests=2,
        monthly_budget_usd=5.0,
        allowed_models=["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash"],
        priority=0,
    ),
    KeyTier.STARTER: TierLimits(
        requests_per_minute=60,
        requests_per_day=1_000,
        tokens_per_month=1_000_000,
        max_concurrent_requests=5,
        monthly_budget_usd=50.0,
        priority=1,
    ),
    KeyTier.PRO: TierLimits(
        requests_per_minute=300,
        requests_per_day=10_000,
        tokens_per_month=10_000_000,
        max_concurrent_requests=20,
        monthly_budget_usd=500.0,
        priority=2,
    ),
    KeyTier.ENTERPRISE: TierLimits(
        requests_per_minute=1000,
        requests_per_day=100_000,
        tokens_per_month=100_000_000,
        max_concurrent_requests=100,
        monthly_budget_usd=10_000.0,
        priority=3,
    ),
}


@dataclass
class APIKey:
    """Represents an API key with metadata."""

    key_id: str
    key_hash: str  # We never store the raw key
    name: str
    tier: KeyTier
    owner_id: str
    created_at: datetime
    expires_at: datetime | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    # Usage tracking
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Budget controls
    monthly_budget_usd: float | None = None  # Override tier default
    budget_alert_threshold: float = 0.8  # Alert at 80% of budget

    @property
    def limits(self) -> TierLimits:
        """Get the limits for this key's tier."""
        return TIER_LIMITS[self.tier]

    @property
    def effective_monthly_budget(self) -> float:
        """Get effective monthly budget (override or tier default)."""
        if self.monthly_budget_usd is not None:
            return self.monthly_budget_usd
        return self.limits.monthly_budget_usd

    @property
    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the key is valid for use."""
        return self.is_active and not self.is_expired

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "name": self.name,
            "tier": self.tier.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "monthly_budget_usd": self.monthly_budget_usd,
            "budget_alert_threshold": self.budget_alert_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "APIKey":
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            name=data["name"],
            tier=KeyTier(data["tier"]),
            owner_id=data["owner_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
            total_requests=data.get("total_requests", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            monthly_budget_usd=data.get("monthly_budget_usd"),
            budget_alert_threshold=data.get("budget_alert_threshold", 0.8),
        )


class APIKeyManager:
    """
    Manages API keys with Redis backend for persistence.

    Handles key generation, validation, and lifecycle management.
    """

    KEY_PREFIX = "orch:apikey:"
    LOOKUP_PREFIX = "orch:keylookup:"

    def __init__(self, redis_client: Any | None = None):
        """
        Initialize the API key manager.

        Args:
            redis_client: Redis client for persistence (optional for in-memory mode)
        """
        self._redis = redis_client
        self._local_cache: dict[str, APIKey] = {}
        self._secret_key = secrets.token_bytes(32)

    def generate_key(
        self,
        name: str,
        tier: KeyTier,
        owner_id: str,
        expires_in_days: int | None = None,
        monthly_budget_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.

        Args:
            name: Human-readable name for the key
            tier: Key tier level
            owner_id: Owner identifier
            expires_in_days: Days until expiration (None = no expiration)
            monthly_budget_usd: Custom monthly budget override
            metadata: Additional metadata

        Returns:
            Tuple of (raw_key, APIKey object)
        """
        # Generate a secure random key
        key_id = f"orch_{secrets.token_hex(8)}"
        raw_key = f"{key_id}_{secrets.token_urlsafe(32)}"

        # Hash the key for storage
        key_hash = self._hash_key(raw_key)

        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = now + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            tier=tier,
            owner_id=owner_id,
            created_at=now,
            expires_at=expires_at,
            monthly_budget_usd=monthly_budget_usd,
            metadata=metadata or {},
        )

        # Store the key
        self._store_key(api_key)

        logger.info(
            "API key generated",
            key_id=key_id,
            tier=tier.value,
            owner_id=owner_id,
        )

        return raw_key, api_key

    def validate_key(self, raw_key: str) -> APIKey | None:
        """
        Validate an API key and return its metadata.

        Args:
            raw_key: The raw API key to validate

        Returns:
            APIKey if valid, None otherwise
        """
        if not raw_key or "_" not in raw_key:
            return None

        key_id = raw_key.split("_")[0] + "_" + raw_key.split("_")[1]
        key_hash = self._hash_key(raw_key)

        # Try to get the key
        api_key = self._get_key(key_id)

        if api_key is None:
            logger.warning("API key not found", key_id=key_id)
            return None

        # Verify the hash matches (constant-time comparison)
        if not hmac.compare_digest(api_key.key_hash, key_hash):
            logger.warning("API key hash mismatch", key_id=key_id)
            return None

        # Check if valid
        if not api_key.is_valid:
            logger.warning(
                "API key invalid",
                key_id=key_id,
                is_active=api_key.is_active,
                is_expired=api_key.is_expired,
            )
            return None

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: The key ID to revoke

        Returns:
            True if revoked, False if not found
        """
        api_key = self._get_key(key_id)
        if api_key is None:
            return False

        api_key.is_active = False
        self._store_key(api_key)

        logger.info("API key revoked", key_id=key_id)
        return True

    def update_usage(
        self,
        key_id: str,
        requests: int = 0,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> APIKey | None:
        """
        Update usage counters for a key.

        Args:
            key_id: The key ID
            requests: Number of requests to add
            tokens: Number of tokens to add
            cost_usd: Cost to add

        Returns:
            Updated APIKey or None if not found
        """
        api_key = self._get_key(key_id)
        if api_key is None:
            return None

        api_key.total_requests += requests
        api_key.total_tokens += tokens
        api_key.total_cost_usd += cost_usd

        self._store_key(api_key)

        # Check budget alerts
        if api_key.total_cost_usd >= api_key.effective_monthly_budget * api_key.budget_alert_threshold:
            logger.warning(
                "API key approaching budget limit",
                key_id=key_id,
                current_cost=api_key.total_cost_usd,
                budget=api_key.effective_monthly_budget,
                threshold=api_key.budget_alert_threshold,
            )

        return api_key

    def get_key(self, key_id: str) -> APIKey | None:
        """Get an API key by ID."""
        return self._get_key(key_id)

    def list_keys(self, owner_id: str | None = None) -> list[APIKey]:
        """
        List all API keys, optionally filtered by owner.

        Args:
            owner_id: Filter by owner (optional)

        Returns:
            List of API keys
        """
        if self._redis:
            keys = []
            pattern = f"{self.KEY_PREFIX}*"
            for key in self._redis.scan_iter(pattern):
                data = self._redis.hgetall(key)
                if data:
                    api_key = APIKey.from_dict(
                        {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in data.items()}
                    )
                    if owner_id is None or api_key.owner_id == owner_id:
                        keys.append(api_key)
            return keys
        else:
            return [
                k for k in self._local_cache.values()
                if owner_id is None or k.owner_id == owner_id
            ]

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(
            raw_key.encode() + self._secret_key
        ).hexdigest()

    def _store_key(self, api_key: APIKey) -> None:
        """Store an API key."""
        if self._redis:
            import json
            key = f"{self.KEY_PREFIX}{api_key.key_id}"
            self._redis.set(key, json.dumps(api_key.to_dict()))
        else:
            self._local_cache[api_key.key_id] = api_key

    def _get_key(self, key_id: str) -> APIKey | None:
        """Get an API key by ID."""
        if self._redis:
            import json
            key = f"{self.KEY_PREFIX}{key_id}"
            data = self._redis.get(key)
            if data:
                return APIKey.from_dict(json.loads(data))
            return None
        else:
            return self._local_cache.get(key_id)
