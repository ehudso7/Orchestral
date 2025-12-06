"""
Stripe service for Orchestral billing.

Handles customers, subscriptions, payments, and metered billing.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

import structlog

from orchestral.core.config import get_settings
from orchestral.billing.api_keys import KeyTier, get_api_key_manager
from orchestral.billing.stripe_models import (
    StripeCustomer,
    StripeSubscription,
    StripeProduct,
    StripePrice,
    SubscriptionStatus,
    BillingInterval,
    Invoice,
    PaymentMethod,
    TIER_PRODUCT_NAMES,
    TIER_PRICING,
    TIER_FEATURES,
)

logger = structlog.get_logger()

# Lazy import stripe to avoid import errors when not configured
stripe = None


def _get_stripe():
    """Lazy load stripe module."""
    global stripe
    if stripe is None:
        import stripe as stripe_module
        stripe = stripe_module
    return stripe


class StripeService:
    """
    Manages Stripe operations for Orchestral.

    Handles customers, subscriptions, payments, and metered billing.
    """

    CUSTOMER_PREFIX = "orch:stripe:customer:"
    CUSTOMER_OWNER_PREFIX = "orch:stripe:customer_by_owner:"
    SUBSCRIPTION_PREFIX = "orch:stripe:subscription:"
    SUBSCRIPTION_OWNER_PREFIX = "orch:stripe:subscription_by_owner:"

    def __init__(self, redis_client: Any | None = None):
        """
        Initialize the Stripe service.

        Args:
            redis_client: Redis client for persistence
        """
        self._redis = redis_client
        self._local_customers: dict[str, StripeCustomer] = {}
        self._local_subscriptions: dict[str, StripeSubscription] = {}
        self._initialized = False

        settings = get_settings()
        if settings.stripe.is_configured:
            stripe_mod = _get_stripe()
            stripe_mod.api_key = settings.stripe.secret_key.get_secret_value()
            self._initialized = True
            logger.info("Stripe initialized", test_mode=settings.stripe.test_mode)
        else:
            logger.warning("Stripe not configured - payment features disabled")

    @property
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured."""
        return self._initialized

    def _require_stripe(self) -> None:
        """Raise error if Stripe is not configured."""
        if not self._initialized:
            raise RuntimeError("Stripe is not configured. Set STRIPE_SECRET_KEY.")

    # ==================== PRODUCTS ====================

    async def create_products(self) -> dict[str, str]:
        """
        Create Orchestral products in Stripe.

        Returns dict mapping tier name to product ID.
        Run once during initial setup.
        """
        self._require_stripe()
        stripe_mod = _get_stripe()
        products = {}

        product_configs = [
            {
                "name": "Orchestral Free",
                "description": "Free tier - 100 requests/day, basic models only",
                "tier": "free",
                "metadata": {
                    "tier": "free",
                    "requests_per_day": "100",
                    "tokens_per_month": "100000",
                },
            },
            {
                "name": "Orchestral Starter",
                "description": "For individuals - 1K requests/day, all models, email support",
                "tier": "starter",
                "metadata": {
                    "tier": "starter",
                    "requests_per_day": "1000",
                    "tokens_per_month": "1000000",
                },
            },
            {
                "name": "Orchestral Pro",
                "description": "For teams - 10K requests/day, priority support, advanced features",
                "tier": "pro",
                "metadata": {
                    "tier": "pro",
                    "requests_per_day": "10000",
                    "tokens_per_month": "10000000",
                },
            },
            {
                "name": "Orchestral Enterprise",
                "description": "For organizations - 100K requests/day, dedicated support, SLA",
                "tier": "enterprise",
                "metadata": {
                    "tier": "enterprise",
                    "requests_per_day": "100000",
                    "tokens_per_month": "100000000",
                },
            },
        ]

        for config in product_configs:
            product = stripe_mod.Product.create(
                name=config["name"],
                description=config["description"],
                metadata=config["metadata"],
            )
            products[config["tier"]] = product.id
            logger.info(
                "Created Stripe product",
                tier=config["tier"],
                product_id=product.id,
            )

        return products

    async def create_prices(self, product_ids: dict[str, str]) -> dict[str, str]:
        """
        Create subscription prices for products.

        Args:
            product_ids: Dict mapping tier to product ID

        Returns:
            Dict mapping price name to price ID
        """
        self._require_stripe()
        stripe_mod = _get_stripe()
        prices = {}

        # Price configurations (amounts in cents)
        price_configs = [
            # Starter tier
            {
                "product": "starter",
                "nickname": "starter_monthly",
                "amount": 2900,
                "interval": "month",
            },
            {
                "product": "starter",
                "nickname": "starter_yearly",
                "amount": 29000,
                "interval": "year",
            },
            # Pro tier
            {
                "product": "pro",
                "nickname": "pro_monthly",
                "amount": 9900,
                "interval": "month",
            },
            {
                "product": "pro",
                "nickname": "pro_yearly",
                "amount": 99000,
                "interval": "year",
            },
            # Enterprise tier
            {
                "product": "enterprise",
                "nickname": "enterprise_monthly",
                "amount": 49900,
                "interval": "month",
            },
            {
                "product": "enterprise",
                "nickname": "enterprise_yearly",
                "amount": 499000,
                "interval": "year",
            },
        ]

        for config in price_configs:
            product_id = product_ids.get(config["product"])
            if not product_id:
                continue

            price = stripe_mod.Price.create(
                product=product_id,
                nickname=config["nickname"],
                unit_amount=config["amount"],
                currency="usd",
                recurring={"interval": config["interval"]},
                metadata={"tier": config["product"]},
            )
            prices[config["nickname"]] = price.id
            logger.info(
                "Created Stripe price",
                nickname=config["nickname"],
                price_id=price.id,
            )

        # Create metered price for token usage (attach to Pro product)
        if product_ids.get("pro"):
            metered_price = stripe_mod.Price.create(
                product=product_ids["pro"],
                nickname="token_usage",
                currency="usd",
                recurring={
                    "interval": "month",
                    "usage_type": "metered",
                    "aggregate_usage": "sum",
                },
                unit_amount_decimal="1",  # $0.01 per 1000 tokens (1 cent)
                metadata={"type": "metered", "unit": "1000_tokens"},
            )
            prices["token_usage"] = metered_price.id
            logger.info("Created metered price", price_id=metered_price.id)

        return prices

    async def get_products(self) -> list[StripeProduct]:
        """List all Orchestral products from Stripe."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        products = stripe_mod.Product.list(active=True, limit=100)

        result = []
        for prod in products.data:
            tier = prod.metadata.get("tier")
            if tier:  # Only include Orchestral products
                result.append(
                    StripeProduct(
                        product_id=prod.id,
                        name=prod.name,
                        description=prod.description,
                        tier=tier,
                        active=prod.active,
                        metadata=dict(prod.metadata),
                    )
                )

        return result

    async def get_prices(self, product_id: str | None = None) -> list[StripePrice]:
        """List prices, optionally filtered by product."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        params: dict[str, Any] = {"active": True, "limit": 100}
        if product_id:
            params["product"] = product_id

        prices = stripe_mod.Price.list(**params)

        result = []
        for price in prices.data:
            recurring = price.recurring or {}
            result.append(
                StripePrice(
                    price_id=price.id,
                    product_id=price.product,
                    nickname=price.nickname,
                    unit_amount=price.unit_amount or 0,
                    currency=price.currency,
                    interval=BillingInterval(recurring.get("interval", "month")),
                    interval_count=recurring.get("interval_count", 1),
                    type=price.type,
                    active=price.active,
                    usage_type=recurring.get("usage_type"),
                    aggregate_usage=recurring.get("aggregate_usage"),
                )
            )

        return result

    # ==================== CUSTOMERS ====================

    async def create_customer(
        self,
        owner_id: str,
        email: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StripeCustomer:
        """Create a Stripe customer and link to Orchestral owner."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        # Check if customer already exists
        existing = await self.get_customer_by_owner(owner_id)
        if existing:
            logger.info("Customer already exists", owner_id=owner_id)
            return existing

        # Create in Stripe
        stripe_customer = stripe_mod.Customer.create(
            email=email,
            name=name,
            metadata={
                "owner_id": owner_id,
                "platform": "orchestral",
                **(metadata or {}),
            },
        )

        customer = StripeCustomer(
            customer_id=stripe_customer.id,
            owner_id=owner_id,
            email=email,
            name=name,
            metadata=metadata or {},
        )

        # Store locally
        await self._store_customer(customer)

        logger.info(
            "Created Stripe customer",
            customer_id=customer.customer_id,
            owner_id=owner_id,
        )
        return customer

    async def get_customer(self, customer_id: str) -> StripeCustomer | None:
        """Get customer by Stripe customer ID."""
        return await self._get_customer(customer_id)

    async def get_customer_by_owner(self, owner_id: str) -> StripeCustomer | None:
        """Get customer by Orchestral owner ID."""
        if self._redis:
            # Use direct lookup key
            lookup_key = f"{self.CUSTOMER_OWNER_PREFIX}{owner_id}"
            customer_id = self._redis.get(lookup_key)
            if customer_id:
                if isinstance(customer_id, bytes):
                    customer_id = customer_id.decode()
                return await self._get_customer(customer_id)
        else:
            for customer in self._local_customers.values():
                if customer.owner_id == owner_id:
                    return customer
        return None

    async def update_customer(
        self,
        customer_id: str,
        email: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StripeCustomer | None:
        """Update customer details."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        customer = await self._get_customer(customer_id)
        if not customer:
            return None

        # Build update params
        params: dict[str, Any] = {}
        if email:
            params["email"] = email
            customer.email = email
        if name:
            params["name"] = name
            customer.name = name
        if metadata:
            params["metadata"] = metadata
            customer.metadata.update(metadata)

        if params:
            stripe_mod.Customer.modify(customer_id, **params)
            await self._store_customer(customer)

        return customer

    async def update_payment_method(
        self,
        customer_id: str,
        payment_method_id: str,
    ) -> StripeCustomer | None:
        """Update customer's default payment method."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        customer = await self._get_customer(customer_id)
        if not customer:
            return None

        # Attach and set as default in Stripe
        stripe_mod.PaymentMethod.attach(payment_method_id, customer=customer_id)
        stripe_mod.Customer.modify(
            customer_id,
            invoice_settings={"default_payment_method": payment_method_id},
        )

        customer.default_payment_method_id = payment_method_id
        await self._store_customer(customer)

        logger.info(
            "Updated payment method",
            customer_id=customer_id,
            payment_method_id=payment_method_id,
        )
        return customer

    async def get_payment_methods(self, customer_id: str) -> list[PaymentMethod]:
        """Get payment methods for a customer."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        methods = stripe_mod.PaymentMethod.list(customer=customer_id, type="card")

        result = []
        for pm in methods.data:
            card = pm.card or {}
            result.append(
                PaymentMethod(
                    payment_method_id=pm.id,
                    type=pm.type,
                    customer_id=customer_id,
                    card_brand=card.get("brand"),
                    card_last4=card.get("last4"),
                    card_exp_month=card.get("exp_month"),
                    card_exp_year=card.get("exp_year"),
                    created_at=datetime.fromtimestamp(pm.created, tz=timezone.utc),
                )
            )

        return result

    async def delete_payment_method(self, payment_method_id: str) -> bool:
        """Delete a payment method."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        try:
            stripe_mod.PaymentMethod.detach(payment_method_id)
            return True
        except Exception as e:
            logger.error("Failed to delete payment method", error=str(e))
            return False

    # ==================== SUBSCRIPTIONS ====================

    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        tier: str,
        trial_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StripeSubscription:
        """
        Create a subscription for a customer.

        This automatically upgrades the API key tier.
        """
        self._require_stripe()
        stripe_mod = _get_stripe()

        customer = await self._get_customer(customer_id)
        if not customer:
            raise ValueError(f"Customer not found: {customer_id}")

        # Check for existing active subscription
        existing = await self.get_subscription_by_owner(customer.owner_id)
        if existing and existing.is_active:
            raise ValueError(
                f"Customer already has active subscription: {existing.subscription_id}"
            )

        # Build subscription params
        params: dict[str, Any] = {
            "customer": customer_id,
            "items": [{"price": price_id}],
            "metadata": {
                "owner_id": customer.owner_id,
                "tier": tier,
                "platform": "orchestral",
                **(metadata or {}),
            },
            "payment_behavior": "default_incomplete",
            "payment_settings": {"save_default_payment_method": "on_subscription"},
            "expand": ["latest_invoice.payment_intent"],
        }

        if trial_days:
            params["trial_period_days"] = trial_days

        # Create in Stripe
        stripe_sub = stripe_mod.Subscription.create(**params)

        # Parse interval from price
        price = stripe_mod.Price.retrieve(price_id)
        interval = BillingInterval(price.recurring.interval)

        subscription = StripeSubscription(
            subscription_id=stripe_sub.id,
            customer_id=customer_id,
            owner_id=customer.owner_id,
            product_id=price.product,
            price_id=price_id,
            status=SubscriptionStatus(stripe_sub.status),
            tier=tier,
            interval=interval,
            current_period_start=datetime.fromtimestamp(
                stripe_sub.current_period_start, tz=timezone.utc
            ),
            current_period_end=datetime.fromtimestamp(
                stripe_sub.current_period_end, tz=timezone.utc
            ),
            trial_start=(
                datetime.fromtimestamp(stripe_sub.trial_start, tz=timezone.utc)
                if stripe_sub.trial_start
                else None
            ),
            trial_end=(
                datetime.fromtimestamp(stripe_sub.trial_end, tz=timezone.utc)
                if stripe_sub.trial_end
                else None
            ),
            metadata=metadata or {},
        )

        await self._store_subscription(subscription)

        # Update API key tier if subscription is active
        if subscription.is_active:
            await self._update_api_key_tier(customer.owner_id, tier)

        logger.info(
            "Created subscription",
            subscription_id=subscription.subscription_id,
            tier=tier,
            status=subscription.status.value,
        )

        return subscription

    async def get_subscription(self, subscription_id: str) -> StripeSubscription | None:
        """Get subscription by ID."""
        return await self._get_subscription(subscription_id)

    async def get_subscription_by_owner(
        self, owner_id: str
    ) -> StripeSubscription | None:
        """Get active subscription for an owner."""
        if self._redis:
            # Use direct lookup key
            lookup_key = f"{self.SUBSCRIPTION_OWNER_PREFIX}{owner_id}"
            subscription_id = self._redis.get(lookup_key)
            if subscription_id:
                if isinstance(subscription_id, bytes):
                    subscription_id = subscription_id.decode()
                sub = await self._get_subscription(subscription_id)
                if sub and sub.is_active:
                    return sub
        else:
            for sub in self._local_subscriptions.values():
                if sub.owner_id == owner_id and sub.is_active:
                    return sub
        return None

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> StripeSubscription | None:
        """Cancel a subscription."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        subscription = await self._get_subscription(subscription_id)
        if not subscription:
            return None

        if at_period_end:
            # Cancel at end of billing period
            stripe_mod.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True,
            )
            subscription.cancel_at_period_end = True
        else:
            # Cancel immediately
            stripe_mod.Subscription.delete(subscription_id)
            subscription.status = SubscriptionStatus.CANCELED
            subscription.canceled_at = datetime.now(timezone.utc)

            # Downgrade to free tier
            await self._update_api_key_tier(subscription.owner_id, "free")

        await self._store_subscription(subscription)

        logger.info(
            "Canceled subscription",
            subscription_id=subscription_id,
            at_period_end=at_period_end,
        )

        return subscription

    async def reactivate_subscription(
        self, subscription_id: str
    ) -> StripeSubscription | None:
        """Reactivate a subscription scheduled for cancellation."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        subscription = await self._get_subscription(subscription_id)
        if not subscription or not subscription.cancel_at_period_end:
            return None

        stripe_mod.Subscription.modify(
            subscription_id,
            cancel_at_period_end=False,
        )
        subscription.cancel_at_period_end = False
        await self._store_subscription(subscription)

        logger.info("Reactivated subscription", subscription_id=subscription_id)
        return subscription

    async def change_subscription(
        self,
        subscription_id: str,
        new_price_id: str,
        new_tier: str,
    ) -> StripeSubscription | None:
        """Upgrade or downgrade a subscription."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        subscription = await self._get_subscription(subscription_id)
        if not subscription:
            return None

        # Get current subscription item
        stripe_sub = stripe_mod.Subscription.retrieve(subscription_id)
        item_id = stripe_sub["items"]["data"][0]["id"]

        # Update subscription
        updated_sub = stripe_mod.Subscription.modify(
            subscription_id,
            items=[{"id": item_id, "price": new_price_id}],
            proration_behavior="create_prorations",
            metadata={"tier": new_tier},
        )

        # Update local record
        price = stripe_mod.Price.retrieve(new_price_id)
        subscription.price_id = new_price_id
        subscription.product_id = price.product
        subscription.tier = new_tier
        subscription.interval = BillingInterval(price.recurring.interval)
        subscription.status = SubscriptionStatus(updated_sub.status)

        await self._store_subscription(subscription)

        # Update API key tier
        if subscription.is_active:
            await self._update_api_key_tier(subscription.owner_id, new_tier)

        logger.info(
            "Changed subscription",
            subscription_id=subscription_id,
            new_tier=new_tier,
        )

        return subscription

    # ==================== METERED BILLING ====================

    async def report_usage(
        self,
        subscription_id: str,
        quantity: int,
        timestamp: datetime | None = None,
        action: str = "increment",
    ) -> bool:
        """
        Report metered usage for a subscription.

        Args:
            subscription_id: Subscription ID
            quantity: Usage quantity (e.g., tokens / 1000)
            timestamp: Usage timestamp (defaults to now)
            action: "increment" or "set"
        """
        self._require_stripe()
        stripe_mod = _get_stripe()

        subscription = await self._get_subscription(subscription_id)
        if not subscription or not subscription.metered_item_id:
            logger.warning(
                "Cannot report usage - no metered item",
                subscription_id=subscription_id,
            )
            return False

        ts = int((timestamp or datetime.now(timezone.utc)).timestamp())

        stripe_mod.SubscriptionItem.create_usage_record(
            subscription.metered_item_id,
            quantity=quantity,
            timestamp=ts,
            action=action,
        )

        logger.debug(
            "Reported usage",
            subscription_id=subscription_id,
            quantity=quantity,
        )

        return True

    async def get_usage_summary(
        self, subscription_item_id: str
    ) -> dict[str, Any] | None:
        """Get usage summary for a metered subscription item."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        try:
            summary = stripe_mod.SubscriptionItem.list_usage_record_summaries(
                subscription_item_id, limit=1
            )
            if summary.data:
                return {
                    "total_usage": summary.data[0].total_usage,
                    "period_start": summary.data[0].period.start,
                    "period_end": summary.data[0].period.end,
                }
        except Exception as e:
            logger.error("Failed to get usage summary", error=str(e))

        return None

    # ==================== INVOICES ====================

    async def get_invoices(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> list[Invoice]:
        """Get invoices for a customer."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        invoices = stripe_mod.Invoice.list(customer=customer_id, limit=limit)

        result = []
        for inv in invoices.data:
            paid_at = None
            if inv.status_transitions and inv.status_transitions.paid_at:
                paid_at = datetime.fromtimestamp(
                    inv.status_transitions.paid_at, tz=timezone.utc
                )

            result.append(
                Invoice(
                    invoice_id=inv.id,
                    customer_id=inv.customer,
                    subscription_id=inv.subscription,
                    status=inv.status,
                    amount_due=inv.amount_due,
                    amount_paid=inv.amount_paid,
                    currency=inv.currency,
                    created_at=datetime.fromtimestamp(inv.created, tz=timezone.utc),
                    due_date=(
                        datetime.fromtimestamp(inv.due_date, tz=timezone.utc)
                        if inv.due_date
                        else None
                    ),
                    paid_at=paid_at,
                    invoice_pdf=inv.invoice_pdf,
                    hosted_invoice_url=inv.hosted_invoice_url,
                )
            )

        return result

    async def get_upcoming_invoice(self, customer_id: str) -> Invoice | None:
        """Get upcoming invoice for a customer."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        try:
            inv = stripe_mod.Invoice.upcoming(customer=customer_id)
            return Invoice(
                invoice_id="upcoming",
                customer_id=inv.customer,
                subscription_id=inv.subscription,
                status="upcoming",
                amount_due=inv.amount_due,
                amount_paid=0,
                currency=inv.currency,
                created_at=datetime.now(timezone.utc),
                due_date=(
                    datetime.fromtimestamp(inv.next_payment_attempt, tz=timezone.utc)
                    if inv.next_payment_attempt
                    else None
                ),
            )
        except stripe_mod.error.InvalidRequestError:
            return None

    # ==================== CHECKOUT SESSIONS ====================

    async def create_checkout_session(
        self,
        owner_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        trial_days: int | None = None,
    ) -> str:
        """
        Create a Stripe Checkout session for new subscriptions.

        Returns the checkout URL.
        """
        self._require_stripe()
        stripe_mod = _get_stripe()

        customer = await self.get_customer_by_owner(owner_id)

        params: dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": {"owner_id": owner_id},
        }

        if customer:
            params["customer"] = customer.customer_id
        else:
            params["customer_creation"] = "always"
            params["customer_email"] = None  # Will be collected in checkout

        if trial_days:
            params["subscription_data"] = {"trial_period_days": trial_days}

        session = stripe_mod.checkout.Session.create(**params)

        logger.info(
            "Created checkout session",
            session_id=session.id,
            owner_id=owner_id,
        )

        return session.url

    async def create_billing_portal_session(
        self,
        customer_id: str,
        return_url: str,
    ) -> str:
        """
        Create a Stripe Billing Portal session.

        Allows customers to manage their subscription.
        Returns the portal URL.
        """
        self._require_stripe()
        stripe_mod = _get_stripe()

        session = stripe_mod.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )

        logger.info(
            "Created billing portal session",
            customer_id=customer_id,
        )

        return session.url

    async def create_setup_intent(self, customer_id: str) -> dict[str, str]:
        """
        Create a SetupIntent for adding a payment method.

        Returns dict with client_secret for frontend.
        """
        self._require_stripe()
        stripe_mod = _get_stripe()

        intent = stripe_mod.SetupIntent.create(
            customer=customer_id,
            payment_method_types=["card"],
        )

        return {
            "client_secret": intent.client_secret,
            "setup_intent_id": intent.id,
        }

    # ==================== INTERNAL HELPERS ====================

    async def _update_api_key_tier(self, owner_id: str, tier: str) -> None:
        """Update API key tier when subscription changes."""
        try:
            key_tier = KeyTier(tier)
            api_key_manager = get_api_key_manager()

            # Find API keys for this owner
            keys = api_key_manager.list_keys(owner_id=owner_id)
            for key in keys:
                if key.tier != key_tier:
                    # Note: APIKeyManager doesn't have update_tier method
                    # We store the new tier in metadata and update limits
                    key.metadata["tier"] = tier
                    api_key_manager._store_key(key)
                    logger.info(
                        "Updated API key tier",
                        key_id=key.key_id,
                        old_tier=key.tier.value,
                        new_tier=tier,
                    )
        except ValueError:
            logger.warning("Invalid tier for API key update", tier=tier)
        except Exception as e:
            logger.error("Failed to update API key tier", error=str(e))

    async def _store_customer(self, customer: StripeCustomer) -> None:
        """Store customer in Redis/memory."""
        if self._redis:
            key = f"{self.CUSTOMER_PREFIX}{customer.customer_id}"
            self._redis.set(key, json.dumps(customer.to_dict()))
            # Also store lookup by owner_id
            lookup_key = f"{self.CUSTOMER_OWNER_PREFIX}{customer.owner_id}"
            self._redis.set(lookup_key, customer.customer_id)
        else:
            self._local_customers[customer.customer_id] = customer

    async def _get_customer(self, customer_id: str) -> StripeCustomer | None:
        """Get customer from Redis/memory."""
        if self._redis:
            key = f"{self.CUSTOMER_PREFIX}{customer_id}"
            data = self._redis.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode()
                return StripeCustomer.from_dict(json.loads(data))
        else:
            return self._local_customers.get(customer_id)
        return None

    async def _store_subscription(self, subscription: StripeSubscription) -> None:
        """Store subscription in Redis/memory."""
        if self._redis:
            key = f"{self.SUBSCRIPTION_PREFIX}{subscription.subscription_id}"
            self._redis.set(key, json.dumps(subscription.to_dict()))
            # Also store lookup by owner_id (only for active subscriptions)
            if subscription.is_active:
                lookup_key = f"{self.SUBSCRIPTION_OWNER_PREFIX}{subscription.owner_id}"
                self._redis.set(lookup_key, subscription.subscription_id)
        else:
            self._local_subscriptions[subscription.subscription_id] = subscription

    async def _get_subscription(
        self, subscription_id: str
    ) -> StripeSubscription | None:
        """Get subscription from Redis/memory."""
        if self._redis:
            key = f"{self.SUBSCRIPTION_PREFIX}{subscription_id}"
            data = self._redis.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode()
                return StripeSubscription.from_dict(json.loads(data))
        else:
            return self._local_subscriptions.get(subscription_id)
        return None

    async def sync_subscription_from_stripe(
        self, subscription_id: str
    ) -> StripeSubscription | None:
        """Sync subscription data from Stripe."""
        self._require_stripe()
        stripe_mod = _get_stripe()

        try:
            stripe_sub = stripe_mod.Subscription.retrieve(subscription_id)
            price = stripe_mod.Price.retrieve(stripe_sub.items.data[0].price.id)

            subscription = StripeSubscription(
                subscription_id=stripe_sub.id,
                customer_id=stripe_sub.customer,
                owner_id=stripe_sub.metadata.get("owner_id", ""),
                product_id=price.product,
                price_id=price.id,
                status=SubscriptionStatus(stripe_sub.status),
                tier=stripe_sub.metadata.get("tier", "starter"),
                interval=BillingInterval(price.recurring.interval),
                current_period_start=datetime.fromtimestamp(
                    stripe_sub.current_period_start, tz=timezone.utc
                ),
                current_period_end=datetime.fromtimestamp(
                    stripe_sub.current_period_end, tz=timezone.utc
                ),
                cancel_at_period_end=stripe_sub.cancel_at_period_end,
                trial_start=(
                    datetime.fromtimestamp(stripe_sub.trial_start, tz=timezone.utc)
                    if stripe_sub.trial_start
                    else None
                ),
                trial_end=(
                    datetime.fromtimestamp(stripe_sub.trial_end, tz=timezone.utc)
                    if stripe_sub.trial_end
                    else None
                ),
                metadata=dict(stripe_sub.metadata),
            )

            await self._store_subscription(subscription)
            return subscription

        except Exception as e:
            logger.error(
                "Failed to sync subscription from Stripe",
                subscription_id=subscription_id,
                error=str(e),
            )
            return None


# Global instance
_stripe_service: StripeService | None = None
_stripe_service_lock = threading.Lock()


def get_stripe_service() -> StripeService:
    """Get the global Stripe service instance."""
    global _stripe_service
    if _stripe_service is None:
        with _stripe_service_lock:
            if _stripe_service is None:
                _stripe_service = StripeService()
    return _stripe_service


def configure_stripe_service(redis_client: Any | None = None) -> StripeService:
    """Configure the global Stripe service with Redis."""
    global _stripe_service
    with _stripe_service_lock:
        _stripe_service = StripeService(redis_client=redis_client)
        return _stripe_service
