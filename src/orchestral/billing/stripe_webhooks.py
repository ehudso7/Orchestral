"""
Stripe webhook handlers for Orchestral.

Processes incoming Stripe webhook events for subscription lifecycle,
payments, and customer updates.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import structlog

from orchestral.core.config import get_settings
from orchestral.billing.stripe_service import get_stripe_service, StripeService
from orchestral.billing.stripe_models import (
    StripeCustomer,
    StripeSubscription,
    SubscriptionStatus,
    BillingInterval,
)

logger = structlog.get_logger()

# Lazy import stripe
stripe = None


def _get_stripe():
    """Lazy load stripe module."""
    global stripe
    if stripe is None:
        import stripe as stripe_module
        stripe = stripe_module
    return stripe


class StripeWebhookHandler:
    """Handles incoming Stripe webhook events."""

    def __init__(self, stripe_service: StripeService | None = None):
        """
        Initialize the webhook handler.

        Args:
            stripe_service: Stripe service instance (uses global if not provided)
        """
        self.settings = get_settings()
        self._stripe_service = stripe_service

    @property
    def stripe_service(self) -> StripeService:
        """Get the Stripe service instance."""
        if self._stripe_service is None:
            self._stripe_service = get_stripe_service()
        return self._stripe_service

    def verify_signature(self, payload: bytes, signature: str) -> Any | None:
        """
        Verify webhook signature and construct event.

        Args:
            payload: Raw request body
            signature: Stripe-Signature header value

        Returns:
            Stripe Event if valid, None otherwise
        """
        if not self.settings.stripe.webhook_secret:
            logger.error("Stripe webhook secret not configured")
            return None

        stripe_mod = _get_stripe()

        try:
            event = stripe_mod.Webhook.construct_event(
                payload,
                signature,
                self.settings.stripe.webhook_secret.get_secret_value(),
            )
            return event
        except stripe_mod.error.SignatureVerificationError as e:
            logger.error("Invalid Stripe webhook signature", error=str(e))
            return None
        except Exception as e:
            logger.error("Failed to construct Stripe event", error=str(e))
            return None

    async def handle_event(self, event: Any) -> dict[str, Any]:
        """
        Route and handle a Stripe webhook event.

        Args:
            event: Stripe Event object

        Returns:
            Response dict with status and message
        """
        event_type = event.type
        data = event.data.object

        logger.info(
            "Processing Stripe webhook",
            event_type=event_type,
            event_id=event.id,
        )

        # Map event types to handlers
        handlers = {
            # Customer events
            "customer.created": self._handle_customer_created,
            "customer.updated": self._handle_customer_updated,
            "customer.deleted": self._handle_customer_deleted,
            # Subscription events
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "customer.subscription.trial_will_end": self._handle_trial_ending,
            "customer.subscription.paused": self._handle_subscription_paused,
            "customer.subscription.resumed": self._handle_subscription_resumed,
            # Payment events
            "invoice.paid": self._handle_invoice_paid,
            "invoice.payment_failed": self._handle_payment_failed,
            "invoice.upcoming": self._handle_invoice_upcoming,
            "invoice.finalized": self._handle_invoice_finalized,
            # Checkout events
            "checkout.session.completed": self._handle_checkout_completed,
            "checkout.session.expired": self._handle_checkout_expired,
            # Payment method events
            "payment_method.attached": self._handle_payment_method_attached,
            "payment_method.detached": self._handle_payment_method_detached,
        }

        handler = handlers.get(event_type)
        if handler:
            try:
                await handler(data)
                return {"status": "success", "event_type": event_type}
            except Exception as e:
                logger.error(
                    "Webhook handler failed",
                    event_type=event_type,
                    error=str(e),
                )
                return {"status": "error", "event_type": event_type, "error": str(e)}
        else:
            logger.debug("Unhandled webhook event type", event_type=event_type)
            return {"status": "ignored", "event_type": event_type}

    # ==================== CUSTOMER HANDLERS ====================

    async def _handle_customer_created(self, data: dict[str, Any]) -> None:
        """Handle customer.created event."""
        owner_id = data.get("metadata", {}).get("owner_id")
        customer_id = data["id"]

        if not owner_id:
            logger.warning(
                "Customer created without owner_id in metadata",
                customer_id=customer_id,
            )
            return

        # Check if we already have this customer
        existing = await self.stripe_service.get_customer(customer_id)
        if existing:
            logger.debug("Customer already exists locally", customer_id=customer_id)
            return

        # Create local record
        customer = StripeCustomer(
            customer_id=customer_id,
            owner_id=owner_id,
            email=data.get("email", ""),
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )
        await self.stripe_service._store_customer(customer)

        logger.info(
            "Customer created via webhook",
            customer_id=customer_id,
            owner_id=owner_id,
        )

    async def _handle_customer_updated(self, data: dict[str, Any]) -> None:
        """Handle customer.updated event."""
        customer_id = data["id"]
        customer = await self.stripe_service.get_customer(customer_id)

        if customer:
            # Update local record
            customer.email = data.get("email", customer.email)
            customer.name = data.get("name", customer.name)

            invoice_settings = data.get("invoice_settings", {})
            if invoice_settings.get("default_payment_method"):
                customer.default_payment_method_id = invoice_settings[
                    "default_payment_method"
                ]

            await self.stripe_service._store_customer(customer)
            logger.info("Customer updated via webhook", customer_id=customer_id)
        else:
            logger.warning(
                "Customer update received for unknown customer",
                customer_id=customer_id,
            )

    async def _handle_customer_deleted(self, data: dict[str, Any]) -> None:
        """Handle customer.deleted event."""
        customer_id = data["id"]
        logger.info(
            "Customer deleted in Stripe",
            customer_id=customer_id,
        )
        # Could clean up local records here if needed

    # ==================== SUBSCRIPTION HANDLERS ====================

    async def _handle_subscription_created(self, data: dict[str, Any]) -> None:
        """Handle customer.subscription.created event."""
        subscription_id = data["id"]

        # Check if we already have this subscription
        existing = await self.stripe_service.get_subscription(subscription_id)
        if existing:
            logger.debug(
                "Subscription already exists locally",
                subscription_id=subscription_id,
            )
            return

        # Sync from Stripe
        subscription = await self.stripe_service.sync_subscription_from_stripe(
            subscription_id
        )

        if subscription:
            logger.info(
                "Subscription created via webhook",
                subscription_id=subscription_id,
                tier=subscription.tier,
                status=subscription.status.value,
            )

            # Update API key tier if active
            if subscription.is_active:
                await self.stripe_service._update_api_key_tier(
                    subscription.owner_id, subscription.tier
                )

    async def _handle_subscription_updated(self, data: dict[str, Any]) -> None:
        """Handle customer.subscription.updated event."""
        subscription_id = data["id"]
        subscription = await self.stripe_service.get_subscription(subscription_id)

        if subscription:
            old_status = subscription.status
            new_status = SubscriptionStatus(data["status"])

            # Update subscription fields
            subscription.status = new_status
            subscription.current_period_start = datetime.fromtimestamp(
                data["current_period_start"], tz=timezone.utc
            )
            subscription.current_period_end = datetime.fromtimestamp(
                data["current_period_end"], tz=timezone.utc
            )
            subscription.cancel_at_period_end = data.get("cancel_at_period_end", False)

            if data.get("canceled_at"):
                subscription.canceled_at = datetime.fromtimestamp(
                    data["canceled_at"], tz=timezone.utc
                )

            # Update tier if changed
            new_tier = data.get("metadata", {}).get("tier")
            if new_tier and new_tier != subscription.tier:
                subscription.tier = new_tier

            await self.stripe_service._store_subscription(subscription)

            # Handle status transitions
            if old_status != new_status:
                await self._handle_status_change(subscription, old_status, new_status)

            logger.info(
                "Subscription updated via webhook",
                subscription_id=subscription_id,
                status=new_status.value,
            )
        else:
            # Sync from Stripe if we don't have it
            await self.stripe_service.sync_subscription_from_stripe(subscription_id)

    async def _handle_subscription_deleted(self, data: dict[str, Any]) -> None:
        """Handle customer.subscription.deleted event."""
        subscription_id = data["id"]
        subscription = await self.stripe_service.get_subscription(subscription_id)

        if subscription:
            subscription.status = SubscriptionStatus.CANCELED
            subscription.canceled_at = datetime.now(timezone.utc)
            await self.stripe_service._store_subscription(subscription)

            # Downgrade to free tier
            await self.stripe_service._update_api_key_tier(
                subscription.owner_id, "free"
            )

            logger.info(
                "Subscription deleted via webhook",
                subscription_id=subscription_id,
                owner_id=subscription.owner_id,
            )

    async def _handle_trial_ending(self, data: dict[str, Any]) -> None:
        """Handle customer.subscription.trial_will_end event."""
        subscription_id = data["id"]
        subscription = await self.stripe_service.get_subscription(subscription_id)

        if subscription:
            trial_end = data.get("trial_end")
            trial_end_dt = (
                datetime.fromtimestamp(trial_end, tz=timezone.utc)
                if trial_end
                else None
            )

            logger.info(
                "Trial ending soon",
                subscription_id=subscription_id,
                owner_id=subscription.owner_id,
                trial_end=trial_end_dt.isoformat() if trial_end_dt else None,
            )

            # TODO: Send email notification about trial ending
            # This would integrate with an email service

    async def _handle_subscription_paused(self, data: dict[str, Any]) -> None:
        """Handle customer.subscription.paused event."""
        subscription_id = data["id"]
        subscription = await self.stripe_service.get_subscription(subscription_id)

        if subscription:
            subscription.status = SubscriptionStatus.PAUSED
            await self.stripe_service._store_subscription(subscription)

            # Downgrade access while paused
            await self.stripe_service._update_api_key_tier(
                subscription.owner_id, "free"
            )

            logger.info(
                "Subscription paused",
                subscription_id=subscription_id,
            )

    async def _handle_subscription_resumed(self, data: dict[str, Any]) -> None:
        """Handle customer.subscription.resumed event."""
        subscription_id = data["id"]
        subscription = await self.stripe_service.get_subscription(subscription_id)

        if subscription:
            subscription.status = SubscriptionStatus.ACTIVE
            await self.stripe_service._store_subscription(subscription)

            # Restore tier access
            await self.stripe_service._update_api_key_tier(
                subscription.owner_id, subscription.tier
            )

            logger.info(
                "Subscription resumed",
                subscription_id=subscription_id,
            )

    async def _handle_status_change(
        self,
        subscription: StripeSubscription,
        old_status: SubscriptionStatus,
        new_status: SubscriptionStatus,
    ) -> None:
        """Handle subscription status transitions."""
        logger.info(
            "Subscription status changed",
            subscription_id=subscription.subscription_id,
            old_status=old_status.value,
            new_status=new_status.value,
        )

        if (
            new_status == SubscriptionStatus.ACTIVE
            and old_status != SubscriptionStatus.ACTIVE
        ):
            # Subscription became active - upgrade tier
            await self.stripe_service._update_api_key_tier(
                subscription.owner_id, subscription.tier
            )
            logger.info(
                "Subscription activated - tier upgraded",
                subscription_id=subscription.subscription_id,
                tier=subscription.tier,
            )

        elif new_status == SubscriptionStatus.PAST_DUE:
            # Payment failed - could restrict access
            logger.warning(
                "Subscription past due - payment required",
                subscription_id=subscription.subscription_id,
                owner_id=subscription.owner_id,
            )
            # TODO: Send dunning email

        elif new_status in [
            SubscriptionStatus.CANCELED,
            SubscriptionStatus.UNPAID,
        ]:
            # Downgrade to free
            await self.stripe_service._update_api_key_tier(
                subscription.owner_id, "free"
            )
            logger.info(
                "Subscription ended - tier downgraded to free",
                subscription_id=subscription.subscription_id,
            )

    # ==================== PAYMENT HANDLERS ====================

    async def _handle_invoice_paid(self, data: dict[str, Any]) -> None:
        """Handle invoice.paid event - successful payment."""
        invoice_id = data["id"]
        customer_id = data["customer"]
        amount = data.get("amount_paid", 0)
        subscription_id = data.get("subscription")

        logger.info(
            "Invoice paid",
            invoice_id=invoice_id,
            customer_id=customer_id,
            amount_cents=amount,
            subscription_id=subscription_id,
        )

        # Subscription should already be active from subscription.updated event
        # This is just for logging/auditing

    async def _handle_payment_failed(self, data: dict[str, Any]) -> None:
        """Handle invoice.payment_failed event."""
        invoice_id = data["id"]
        customer_id = data["customer"]
        attempt_count = data.get("attempt_count", 1)
        next_attempt = data.get("next_payment_attempt")

        logger.warning(
            "Payment failed",
            invoice_id=invoice_id,
            customer_id=customer_id,
            attempt_count=attempt_count,
            next_attempt=next_attempt,
        )

        # TODO: Send payment failed email
        # Stripe will retry automatically based on retry settings

    async def _handle_invoice_upcoming(self, data: dict[str, Any]) -> None:
        """Handle invoice.upcoming event - invoice will be sent soon."""
        customer_id = data["customer"]
        amount_due = data.get("amount_due", 0)

        logger.info(
            "Upcoming invoice",
            customer_id=customer_id,
            amount_due_cents=amount_due,
        )

        # TODO: Could notify customer of upcoming charge

    async def _handle_invoice_finalized(self, data: dict[str, Any]) -> None:
        """Handle invoice.finalized event."""
        invoice_id = data["id"]
        logger.debug("Invoice finalized", invoice_id=invoice_id)

    # ==================== CHECKOUT HANDLERS ====================

    async def _handle_checkout_completed(self, data: dict[str, Any]) -> None:
        """Handle checkout.session.completed event."""
        session_id = data["id"]
        owner_id = data.get("metadata", {}).get("owner_id")
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")
        mode = data.get("mode")

        logger.info(
            "Checkout completed",
            session_id=session_id,
            mode=mode,
            owner_id=owner_id,
            customer_id=customer_id,
            subscription_id=subscription_id,
        )

        # Handle subscription checkout
        if mode == "subscription" and customer_id and owner_id:
            # Link customer to owner if not already
            customer = await self.stripe_service.get_customer(customer_id)
            if not customer:
                stripe_mod = _get_stripe()
                stripe_customer = stripe_mod.Customer.retrieve(customer_id)

                customer = StripeCustomer(
                    customer_id=customer_id,
                    owner_id=owner_id,
                    email=stripe_customer.email or "",
                    name=stripe_customer.name,
                    metadata={"created_via": "checkout"},
                )
                await self.stripe_service._store_customer(customer)
                logger.info(
                    "Customer created from checkout",
                    customer_id=customer_id,
                    owner_id=owner_id,
                )

            # Sync subscription if provided
            if subscription_id:
                await self.stripe_service.sync_subscription_from_stripe(subscription_id)

    async def _handle_checkout_expired(self, data: dict[str, Any]) -> None:
        """Handle checkout.session.expired event."""
        session_id = data["id"]
        owner_id = data.get("metadata", {}).get("owner_id")

        logger.info(
            "Checkout session expired",
            session_id=session_id,
            owner_id=owner_id,
        )

    # ==================== PAYMENT METHOD HANDLERS ====================

    async def _handle_payment_method_attached(self, data: dict[str, Any]) -> None:
        """Handle payment_method.attached event."""
        payment_method_id = data["id"]
        customer_id = data.get("customer")

        if customer_id:
            logger.info(
                "Payment method attached",
                payment_method_id=payment_method_id,
                customer_id=customer_id,
            )

    async def _handle_payment_method_detached(self, data: dict[str, Any]) -> None:
        """Handle payment_method.detached event."""
        payment_method_id = data["id"]

        logger.info(
            "Payment method detached",
            payment_method_id=payment_method_id,
        )


# Global handler instance
_webhook_handler: StripeWebhookHandler | None = None
_webhook_handler_lock = threading.Lock()


def get_webhook_handler() -> StripeWebhookHandler:
    """Get the global webhook handler."""
    global _webhook_handler
    if _webhook_handler is None:
        with _webhook_handler_lock:
            if _webhook_handler is None:
                _webhook_handler = StripeWebhookHandler()
    return _webhook_handler


def configure_webhook_handler(
    stripe_service: StripeService | None = None,
) -> StripeWebhookHandler:
    """Configure the global webhook handler."""
    global _webhook_handler
    with _webhook_handler_lock:
        _webhook_handler = StripeWebhookHandler(stripe_service=stripe_service)
        return _webhook_handler
