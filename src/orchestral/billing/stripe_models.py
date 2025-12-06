"""
Stripe data models for Orchestral.

Provides dataclasses for Stripe entities like customers, subscriptions, and invoices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SubscriptionStatus(str, Enum):
    """Stripe subscription status."""

    ACTIVE = "active"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"
    PAUSED = "paused"


class BillingInterval(str, Enum):
    """Billing interval."""

    MONTHLY = "month"
    YEARLY = "year"


class PaymentStatus(str, Enum):
    """Payment intent status."""

    REQUIRES_PAYMENT_METHOD = "requires_payment_method"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"


@dataclass
class StripeCustomer:
    """Stripe customer linked to Orchestral user."""

    customer_id: str  # Stripe customer ID (cus_xxx)
    owner_id: str  # Orchestral owner ID
    email: str
    name: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    # Payment method
    default_payment_method_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "customer_id": self.customer_id,
            "owner_id": self.owner_id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "default_payment_method_id": self.default_payment_method_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StripeCustomer":
        """Create from dictionary."""
        return cls(
            customer_id=data["customer_id"],
            owner_id=data["owner_id"],
            email=data["email"],
            name=data.get("name"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            default_payment_method_id=data.get("default_payment_method_id"),
        )


@dataclass
class StripeSubscription:
    """Stripe subscription for an Orchestral customer."""

    subscription_id: str  # Stripe subscription ID (sub_xxx)
    customer_id: str  # Stripe customer ID
    owner_id: str  # Orchestral owner ID
    product_id: str  # Stripe product ID
    price_id: str  # Stripe price ID
    status: SubscriptionStatus
    tier: str  # Maps to KeyTier (free, starter, pro, enterprise)
    interval: BillingInterval

    # Billing dates
    current_period_start: datetime
    current_period_end: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    canceled_at: datetime | None = None
    cancel_at_period_end: bool = False

    # Trial
    trial_start: datetime | None = None
    trial_end: datetime | None = None

    # Metered usage item (for token billing)
    metered_item_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]

    @property
    def is_trialing(self) -> bool:
        """Check if subscription is in trial period."""
        return self.status == SubscriptionStatus.TRIALING

    @property
    def days_until_renewal(self) -> int:
        """Days until next billing date."""
        now = datetime.now(timezone.utc)
        delta = self.current_period_end - now
        return max(0, delta.days)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "subscription_id": self.subscription_id,
            "customer_id": self.customer_id,
            "owner_id": self.owner_id,
            "product_id": self.product_id,
            "price_id": self.price_id,
            "status": self.status.value,
            "tier": self.tier,
            "interval": self.interval.value,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "created_at": self.created_at.isoformat(),
            "canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
            "cancel_at_period_end": self.cancel_at_period_end,
            "trial_start": self.trial_start.isoformat() if self.trial_start else None,
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "metered_item_id": self.metered_item_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StripeSubscription":
        """Create from dictionary."""
        return cls(
            subscription_id=data["subscription_id"],
            customer_id=data["customer_id"],
            owner_id=data["owner_id"],
            product_id=data["product_id"],
            price_id=data["price_id"],
            status=SubscriptionStatus(data["status"]),
            tier=data["tier"],
            interval=BillingInterval(data["interval"]),
            current_period_start=datetime.fromisoformat(data["current_period_start"]),
            current_period_end=datetime.fromisoformat(data["current_period_end"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            canceled_at=(
                datetime.fromisoformat(data["canceled_at"])
                if data.get("canceled_at")
                else None
            ),
            cancel_at_period_end=data.get("cancel_at_period_end", False),
            trial_start=(
                datetime.fromisoformat(data["trial_start"])
                if data.get("trial_start")
                else None
            ),
            trial_end=(
                datetime.fromisoformat(data["trial_end"])
                if data.get("trial_end")
                else None
            ),
            metered_item_id=data.get("metered_item_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Invoice:
    """Stripe invoice record."""

    invoice_id: str
    customer_id: str
    subscription_id: str | None
    status: str  # draft, open, paid, uncollectible, void
    amount_due: int  # cents
    amount_paid: int
    currency: str
    created_at: datetime
    due_date: datetime | None
    paid_at: datetime | None = None
    invoice_pdf: str | None = None
    hosted_invoice_url: str | None = None

    @property
    def amount_due_dollars(self) -> float:
        """Amount due in dollars."""
        return self.amount_due / 100.0

    @property
    def amount_paid_dollars(self) -> float:
        """Amount paid in dollars."""
        return self.amount_paid / 100.0

    @property
    def is_paid(self) -> bool:
        """Check if invoice is paid."""
        return self.status == "paid"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invoice_id": self.invoice_id,
            "customer_id": self.customer_id,
            "subscription_id": self.subscription_id,
            "status": self.status,
            "amount_due": self.amount_due,
            "amount_paid": self.amount_paid,
            "currency": self.currency,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "invoice_pdf": self.invoice_pdf,
            "hosted_invoice_url": self.hosted_invoice_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Invoice":
        """Create from dictionary."""
        return cls(
            invoice_id=data["invoice_id"],
            customer_id=data["customer_id"],
            subscription_id=data.get("subscription_id"),
            status=data["status"],
            amount_due=data["amount_due"],
            amount_paid=data["amount_paid"],
            currency=data["currency"],
            created_at=datetime.fromisoformat(data["created_at"]),
            due_date=(
                datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None
            ),
            paid_at=(
                datetime.fromisoformat(data["paid_at"]) if data.get("paid_at") else None
            ),
            invoice_pdf=data.get("invoice_pdf"),
            hosted_invoice_url=data.get("hosted_invoice_url"),
        )


@dataclass
class PaymentMethod:
    """Stripe payment method."""

    payment_method_id: str
    type: str  # card, us_bank_account, etc.
    customer_id: str | None = None

    # Card details (if type == "card")
    card_brand: str | None = None  # visa, mastercard, amex, etc.
    card_last4: str | None = None
    card_exp_month: int | None = None
    card_exp_year: int | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        if self.type == "card" and self.card_brand and self.card_last4:
            return f"{self.card_brand.title()} ending in {self.card_last4}"
        return f"{self.type} payment method"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "payment_method_id": self.payment_method_id,
            "type": self.type,
            "customer_id": self.customer_id,
            "card_brand": self.card_brand,
            "card_last4": self.card_last4,
            "card_exp_month": self.card_exp_month,
            "card_exp_year": self.card_exp_year,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class StripeProduct:
    """Stripe product representing a subscription tier."""

    product_id: str
    name: str
    description: str | None = None
    tier: str = "starter"  # Maps to KeyTier
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "description": self.description,
            "tier": self.tier,
            "active": self.active,
            "metadata": self.metadata,
        }


@dataclass
class StripePrice:
    """Stripe price for a product."""

    price_id: str
    product_id: str
    nickname: str | None = None
    unit_amount: int = 0  # cents
    currency: str = "usd"
    interval: BillingInterval = BillingInterval.MONTHLY
    interval_count: int = 1
    type: str = "recurring"  # recurring or one_time
    active: bool = True

    # For metered billing
    usage_type: str | None = None  # licensed or metered
    aggregate_usage: str | None = None  # sum, last_during_period, last_ever, max

    @property
    def amount_dollars(self) -> float:
        """Price in dollars."""
        return self.unit_amount / 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price_id": self.price_id,
            "product_id": self.product_id,
            "nickname": self.nickname,
            "unit_amount": self.unit_amount,
            "currency": self.currency,
            "interval": self.interval.value,
            "interval_count": self.interval_count,
            "type": self.type,
            "active": self.active,
            "usage_type": self.usage_type,
            "aggregate_usage": self.aggregate_usage,
        }


# Tier to product name mapping
TIER_PRODUCT_NAMES: dict[str, str] = {
    "free": "Orchestral Free",
    "starter": "Orchestral Starter",
    "pro": "Orchestral Pro",
    "enterprise": "Orchestral Enterprise",
}

# Tier pricing (in cents) for display purposes
TIER_PRICING: dict[str, dict[str, int]] = {
    "free": {"monthly": 0, "yearly": 0},
    "starter": {"monthly": 2900, "yearly": 29000},  # $29/mo, $290/yr
    "pro": {"monthly": 9900, "yearly": 99000},  # $99/mo, $990/yr
    "enterprise": {"monthly": 49900, "yearly": 499000},  # $499/mo, $4990/yr
}

# Tier features for display
TIER_FEATURES: dict[str, list[str]] = {
    "free": [
        "100 requests/day",
        "Basic models only (GPT-4o-mini, Claude Haiku, Gemini Flash)",
        "Community support",
        "Basic usage analytics",
    ],
    "starter": [
        "1,000 requests/day",
        "All models available",
        "Email support",
        "Full usage analytics",
        "Response caching",
        "1M tokens/month included",
    ],
    "pro": [
        "10,000 requests/day",
        "All models + priority access",
        "Priority support (24h response)",
        "Advanced analytics & insights",
        "Semantic caching",
        "10M tokens/month included",
        "Team access (up to 5 seats)",
        "Webhook integrations",
    ],
    "enterprise": [
        "100,000 requests/day",
        "All models + dedicated capacity",
        "Dedicated support (4h response)",
        "Custom analytics & reporting",
        "Full enterprise features",
        "100M tokens/month included",
        "Unlimited team seats",
        "SSO/SAML integration",
        "Custom SLA",
        "On-premise deployment option",
    ],
}
