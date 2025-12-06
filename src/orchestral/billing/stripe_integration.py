"""
Stripe payment integration for Orchestral.

Handles subscription management, payment processing, and webhook events.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# Stripe SDK - imported conditionally
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None


class SubscriptionStatus(str, Enum):
    """Subscription status values."""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    TRIALING = "trialing"
    UNPAID = "unpaid"


class PricingPlan(BaseModel):
    """Pricing plan configuration."""
    id: str
    name: str
    stripe_price_id: str
    tier: str  # FREE, STARTER, PRO, ENTERPRISE
    price_monthly: float
    price_yearly: float
    requests_per_minute: int
    requests_per_day: int
    tokens_per_month: int
    monthly_budget: float
    features: list[str] = Field(default_factory=list)


# Default pricing plans - configure via environment or database
DEFAULT_PLANS: dict[str, PricingPlan] = {
    "free": PricingPlan(
        id="free",
        name="Free",
        stripe_price_id="",  # No Stripe price for free tier
        tier="FREE",
        price_monthly=0,
        price_yearly=0,
        requests_per_minute=10,
        requests_per_day=100,
        tokens_per_month=100_000,
        monthly_budget=5,
        features=[
            "100K tokens/month",
            "10 requests/minute",
            "Community support",
            "Basic models only",
        ],
    ),
    "starter": PricingPlan(
        id="starter",
        name="Starter",
        stripe_price_id=os.getenv("STRIPE_STARTER_PRICE_ID", "price_starter"),
        tier="STARTER",
        price_monthly=29,
        price_yearly=290,
        requests_per_minute=60,
        requests_per_day=1_000,
        tokens_per_month=1_000_000,
        monthly_budget=50,
        features=[
            "1M tokens/month",
            "60 requests/minute",
            "All models",
            "Email support",
            "Usage analytics",
        ],
    ),
    "pro": PricingPlan(
        id="pro",
        name="Pro",
        stripe_price_id=os.getenv("STRIPE_PRO_PRICE_ID", "price_pro"),
        tier="PRO",
        price_monthly=99,
        price_yearly=990,
        requests_per_minute=300,
        requests_per_day=10_000,
        tokens_per_month=10_000_000,
        monthly_budget=500,
        features=[
            "10M tokens/month",
            "300 requests/minute",
            "All models + priority",
            "Priority support",
            "Advanced analytics",
            "Semantic caching",
            "Custom routing",
        ],
    ),
    "enterprise": PricingPlan(
        id="enterprise",
        name="Enterprise",
        stripe_price_id=os.getenv("STRIPE_ENTERPRISE_PRICE_ID", "price_enterprise"),
        tier="ENTERPRISE",
        price_monthly=499,
        price_yearly=4990,
        requests_per_minute=1000,
        requests_per_day=100_000,
        tokens_per_month=100_000_000,
        monthly_budget=10_000,
        features=[
            "100M tokens/month",
            "1000 requests/minute",
            "All models + dedicated",
            "24/7 support",
            "Full analytics suite",
            "Custom integrations",
            "SLA guarantee",
            "Dedicated account manager",
        ],
    ),
}


class CustomerData(BaseModel):
    """Customer data model."""
    id: str
    email: str
    name: str | None = None
    stripe_customer_id: str | None = None
    subscription_id: str | None = None
    subscription_status: SubscriptionStatus = SubscriptionStatus.INCOMPLETE
    plan_id: str = "free"
    api_key_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class StripePayments:
    """Stripe payment processing integration."""

    def __init__(
        self,
        api_key: str | None = None,
        webhook_secret: str | None = None,
    ):
        self.api_key = api_key or os.getenv("STRIPE_SECRET_KEY")
        self.webhook_secret = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET")
        self.plans = DEFAULT_PLANS.copy()

        if not STRIPE_AVAILABLE:
            logger.warning("Stripe SDK not installed. Run: pip install stripe")
            return

        if self.api_key:
            stripe.api_key = self.api_key
            logger.info("Stripe payments initialized")
        else:
            logger.warning("Stripe API key not configured")

    def is_configured(self) -> bool:
        """Check if Stripe is properly configured."""
        return STRIPE_AVAILABLE and bool(self.api_key)

    async def create_customer(
        self,
        email: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Create a Stripe customer."""
        if not self.is_configured():
            logger.error("Stripe not configured")
            return None

        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=metadata or {},
            )
            logger.info("Created Stripe customer", customer_id=customer.id, email=email)
            return customer.id
        except stripe.StripeError as e:
            logger.error("Failed to create customer", error=str(e))
            return None

    async def create_checkout_session(
        self,
        customer_id: str,
        plan_id: str,
        success_url: str,
        cancel_url: str,
        annual: bool = False,
    ) -> dict[str, Any] | None:
        """Create a Stripe Checkout session for subscription."""
        if not self.is_configured():
            return None

        plan = self.plans.get(plan_id)
        if not plan or plan_id == "free":
            logger.error("Invalid plan for checkout", plan_id=plan_id)
            return None

        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price": plan.stripe_price_id,
                    "quantity": 1,
                }],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "plan_id": plan_id,
                    "billing_cycle": "annual" if annual else "monthly",
                },
            )
            logger.info(
                "Created checkout session",
                session_id=session.id,
                plan_id=plan_id,
            )
            return {
                "session_id": session.id,
                "url": session.url,
            }
        except stripe.StripeError as e:
            logger.error("Failed to create checkout session", error=str(e))
            return None

    async def create_billing_portal_session(
        self,
        customer_id: str,
        return_url: str,
    ) -> str | None:
        """Create a Stripe Billing Portal session for self-service."""
        if not self.is_configured():
            return None

        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            return session.url
        except stripe.StripeError as e:
            logger.error("Failed to create portal session", error=str(e))
            return None

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> bool:
        """Cancel a subscription."""
        if not self.is_configured():
            return False

        try:
            if at_period_end:
                stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                )
            else:
                stripe.Subscription.delete(subscription_id)

            logger.info(
                "Canceled subscription",
                subscription_id=subscription_id,
                immediate=not at_period_end,
            )
            return True
        except stripe.StripeError as e:
            logger.error("Failed to cancel subscription", error=str(e))
            return False

    async def get_subscription(
        self,
        subscription_id: str,
    ) -> dict[str, Any] | None:
        """Get subscription details."""
        if not self.is_configured():
            return None

        try:
            sub = stripe.Subscription.retrieve(subscription_id)
            return {
                "id": sub.id,
                "status": sub.status,
                "current_period_start": sub.current_period_start,
                "current_period_end": sub.current_period_end,
                "cancel_at_period_end": sub.cancel_at_period_end,
                "plan": sub["items"]["data"][0]["price"]["id"] if sub["items"]["data"] else None,
            }
        except stripe.StripeError as e:
            logger.error("Failed to get subscription", error=str(e))
            return None

    async def record_usage(
        self,
        subscription_item_id: str,
        quantity: int,
        timestamp: int | None = None,
    ) -> bool:
        """Record metered usage for usage-based billing."""
        if not self.is_configured():
            return False

        try:
            stripe.SubscriptionItem.create_usage_record(
                subscription_item_id,
                quantity=quantity,
                timestamp=timestamp or int(datetime.now(timezone.utc).timestamp()),
                action="increment",
            )
            logger.debug(
                "Recorded usage",
                subscription_item_id=subscription_item_id,
                quantity=quantity,
            )
            return True
        except stripe.StripeError as e:
            logger.error("Failed to record usage", error=str(e))
            return False

    def verify_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> dict[str, Any] | None:
        """Verify and parse a Stripe webhook event."""
        if not self.is_configured() or not self.webhook_secret:
            return None

        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.webhook_secret,
            )
            return {
                "type": event.type,
                "data": event.data.object,
            }
        except stripe.SignatureVerificationError as e:
            logger.error("Webhook signature verification failed", error=str(e))
            return None
        except Exception as e:
            logger.error("Webhook processing failed", error=str(e))
            return None

    async def handle_webhook_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a Stripe webhook event."""
        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.paid": self._handle_invoice_paid,
            "invoice.payment_failed": self._handle_payment_failed,
        }

        handler = handlers.get(event_type)
        if handler:
            return await handler(data)

        logger.debug("Unhandled webhook event", event_type=event_type)
        return {"handled": False, "event_type": event_type}

    async def _handle_checkout_completed(self, data: dict) -> dict:
        """Handle successful checkout."""
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")
        plan_id = data.get("metadata", {}).get("plan_id")

        logger.info(
            "Checkout completed",
            customer_id=customer_id,
            subscription_id=subscription_id,
            plan_id=plan_id,
        )

        return {
            "handled": True,
            "action": "activate_subscription",
            "customer_id": customer_id,
            "subscription_id": subscription_id,
            "plan_id": plan_id,
        }

    async def _handle_subscription_created(self, data: dict) -> dict:
        """Handle new subscription."""
        return {
            "handled": True,
            "action": "subscription_created",
            "subscription_id": data.get("id"),
            "customer_id": data.get("customer"),
            "status": data.get("status"),
        }

    async def _handle_subscription_updated(self, data: dict) -> dict:
        """Handle subscription update."""
        return {
            "handled": True,
            "action": "subscription_updated",
            "subscription_id": data.get("id"),
            "customer_id": data.get("customer"),
            "status": data.get("status"),
            "cancel_at_period_end": data.get("cancel_at_period_end"),
        }

    async def _handle_subscription_deleted(self, data: dict) -> dict:
        """Handle subscription cancellation."""
        return {
            "handled": True,
            "action": "subscription_canceled",
            "subscription_id": data.get("id"),
            "customer_id": data.get("customer"),
        }

    async def _handle_invoice_paid(self, data: dict) -> dict:
        """Handle successful payment."""
        return {
            "handled": True,
            "action": "payment_succeeded",
            "invoice_id": data.get("id"),
            "customer_id": data.get("customer"),
            "amount_paid": data.get("amount_paid", 0) / 100,  # Convert from cents
        }

    async def _handle_payment_failed(self, data: dict) -> dict:
        """Handle failed payment."""
        return {
            "handled": True,
            "action": "payment_failed",
            "invoice_id": data.get("id"),
            "customer_id": data.get("customer"),
            "attempt_count": data.get("attempt_count", 0),
        }


# Singleton instance
_stripe_payments: StripePayments | None = None


def get_stripe_payments() -> StripePayments:
    """Get or create the Stripe payments instance."""
    global _stripe_payments
    if _stripe_payments is None:
        _stripe_payments = StripePayments()
    return _stripe_payments
