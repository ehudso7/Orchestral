"""
Subscription management API endpoints for Orchestral.

Provides REST API for managing Stripe subscriptions, customers, and billing.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Header, Request, Depends
from pydantic import BaseModel, Field

from orchestral.core.config import get_settings
from orchestral.billing.api_keys import get_api_key_manager, APIKey
from orchestral.billing.stripe_service import get_stripe_service
from orchestral.billing.stripe_webhooks import get_webhook_handler
from orchestral.billing.stripe_models import (
    TIER_FEATURES,
    TIER_PRICING,
    TIER_PRODUCT_NAMES,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/billing", tags=["billing"])


# ==================== REQUEST/RESPONSE MODELS ====================


class CreateCustomerRequest(BaseModel):
    """Request to create a customer."""

    email: str = Field(..., description="Customer email address")
    name: str | None = Field(None, description="Customer name")


class UpdateCustomerRequest(BaseModel):
    """Request to update a customer."""

    email: str | None = Field(None, description="New email address")
    name: str | None = Field(None, description="New name")


class CreateSubscriptionRequest(BaseModel):
    """Request to create a subscription."""

    price_id: str = Field(..., description="Stripe price ID")
    trial_days: int | None = Field(None, description="Number of trial days")


class ChangeSubscriptionRequest(BaseModel):
    """Request to change subscription plan."""

    price_id: str = Field(..., description="New Stripe price ID")
    tier: str = Field(..., description="New tier (starter, pro, enterprise)")


class CheckoutRequest(BaseModel):
    """Request to create checkout session."""

    price_id: str = Field(..., description="Stripe price ID")
    success_url: str = Field(..., description="URL to redirect after success")
    cancel_url: str = Field(..., description="URL to redirect after cancel")
    trial_days: int | None = Field(None, description="Number of trial days")


class PortalRequest(BaseModel):
    """Request to create billing portal session."""

    return_url: str = Field(..., description="URL to return to after portal")


class CustomerResponse(BaseModel):
    """Customer response."""

    customer_id: str
    owner_id: str
    email: str
    name: str | None
    has_payment_method: bool
    created_at: str


class SubscriptionResponse(BaseModel):
    """Subscription response."""

    subscription_id: str
    status: str
    tier: str
    interval: str
    current_period_start: str
    current_period_end: str
    cancel_at_period_end: bool
    trial_end: str | None = None
    days_until_renewal: int


class InvoiceResponse(BaseModel):
    """Invoice response."""

    invoice_id: str
    status: str
    amount_due: int
    amount_paid: int
    amount_due_dollars: float
    currency: str
    created_at: str
    due_date: str | None
    invoice_pdf: str | None
    hosted_invoice_url: str | None


class PaymentMethodResponse(BaseModel):
    """Payment method response."""

    payment_method_id: str
    type: str
    card_brand: str | None
    card_last4: str | None
    card_exp_month: int | None
    card_exp_year: int | None
    display_name: str


class PlanResponse(BaseModel):
    """Available plan response."""

    tier: str
    name: str
    monthly_price_id: str | None
    yearly_price_id: str | None
    monthly_price_cents: int
    yearly_price_cents: int
    monthly_price_dollars: float
    yearly_price_dollars: float
    features: list[str]


class SetupIntentResponse(BaseModel):
    """Setup intent response for adding payment methods."""

    client_secret: str
    setup_intent_id: str


class BillingStatusResponse(BaseModel):
    """Overall billing status response."""

    has_customer: bool
    customer_id: str | None
    has_subscription: bool
    subscription_status: str | None
    current_tier: str
    has_payment_method: bool


# ==================== HELPER FUNCTIONS ====================


async def get_current_api_key(x_api_key: str = Header(...)) -> APIKey:
    """Get and validate current API key."""
    api_key_manager = get_api_key_manager()
    api_key = api_key_manager.validate_key(x_api_key)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


async def get_current_owner(api_key: APIKey = Depends(get_current_api_key)) -> str:
    """Get current owner from API key."""
    return api_key.owner_id


def get_tier_from_price_id(price_id: str) -> str:
    """Determine tier from price ID."""
    settings = get_settings()

    if price_id in [
        settings.stripe.price_starter_monthly_id,
        settings.stripe.price_starter_yearly_id,
    ]:
        return "starter"
    elif price_id in [
        settings.stripe.price_pro_monthly_id,
        settings.stripe.price_pro_yearly_id,
    ]:
        return "pro"
    elif price_id in [
        settings.stripe.price_enterprise_monthly_id,
        settings.stripe.price_enterprise_yearly_id,
    ]:
        return "enterprise"
    else:
        return "starter"  # Default


# ==================== BILLING STATUS ENDPOINT ====================


@router.get("/status", response_model=BillingStatusResponse)
async def get_billing_status(owner_id: str = Depends(get_current_owner)):
    """Get overall billing status for current user."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    subscription = await stripe_service.get_subscription_by_owner(owner_id)

    return BillingStatusResponse(
        has_customer=customer is not None,
        customer_id=customer.customer_id if customer else None,
        has_subscription=subscription is not None,
        subscription_status=subscription.status.value if subscription else None,
        current_tier=subscription.tier if subscription else "free",
        has_payment_method=(
            customer.default_payment_method_id is not None if customer else False
        ),
    )


# ==================== PLANS ENDPOINT ====================


@router.get("/plans", response_model=list[PlanResponse])
async def list_plans():
    """List available subscription plans."""
    settings = get_settings()

    plans = [
        PlanResponse(
            tier="free",
            name=TIER_PRODUCT_NAMES.get("free", "Free"),
            monthly_price_id=None,
            yearly_price_id=None,
            monthly_price_cents=TIER_PRICING["free"]["monthly"],
            yearly_price_cents=TIER_PRICING["free"]["yearly"],
            monthly_price_dollars=TIER_PRICING["free"]["monthly"] / 100,
            yearly_price_dollars=TIER_PRICING["free"]["yearly"] / 100,
            features=TIER_FEATURES.get("free", []),
        ),
        PlanResponse(
            tier="starter",
            name=TIER_PRODUCT_NAMES.get("starter", "Starter"),
            monthly_price_id=settings.stripe.price_starter_monthly_id,
            yearly_price_id=settings.stripe.price_starter_yearly_id,
            monthly_price_cents=TIER_PRICING["starter"]["monthly"],
            yearly_price_cents=TIER_PRICING["starter"]["yearly"],
            monthly_price_dollars=TIER_PRICING["starter"]["monthly"] / 100,
            yearly_price_dollars=TIER_PRICING["starter"]["yearly"] / 100,
            features=TIER_FEATURES.get("starter", []),
        ),
        PlanResponse(
            tier="pro",
            name=TIER_PRODUCT_NAMES.get("pro", "Pro"),
            monthly_price_id=settings.stripe.price_pro_monthly_id,
            yearly_price_id=settings.stripe.price_pro_yearly_id,
            monthly_price_cents=TIER_PRICING["pro"]["monthly"],
            yearly_price_cents=TIER_PRICING["pro"]["yearly"],
            monthly_price_dollars=TIER_PRICING["pro"]["monthly"] / 100,
            yearly_price_dollars=TIER_PRICING["pro"]["yearly"] / 100,
            features=TIER_FEATURES.get("pro", []),
        ),
        PlanResponse(
            tier="enterprise",
            name=TIER_PRODUCT_NAMES.get("enterprise", "Enterprise"),
            monthly_price_id=settings.stripe.price_enterprise_monthly_id,
            yearly_price_id=settings.stripe.price_enterprise_yearly_id,
            monthly_price_cents=TIER_PRICING["enterprise"]["monthly"],
            yearly_price_cents=TIER_PRICING["enterprise"]["yearly"],
            monthly_price_dollars=TIER_PRICING["enterprise"]["monthly"] / 100,
            yearly_price_dollars=TIER_PRICING["enterprise"]["yearly"] / 100,
            features=TIER_FEATURES.get("enterprise", []),
        ),
    ]

    return plans


# ==================== CUSTOMER ENDPOINTS ====================


@router.post("/customers", response_model=CustomerResponse)
async def create_customer(
    request: CreateCustomerRequest,
    owner_id: str = Depends(get_current_owner),
):
    """Create or get Stripe customer for current user."""
    stripe_service = get_stripe_service()

    if not stripe_service.is_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    customer = await stripe_service.create_customer(
        owner_id=owner_id,
        email=request.email,
        name=request.name,
    )

    return CustomerResponse(
        customer_id=customer.customer_id,
        owner_id=customer.owner_id,
        email=customer.email,
        name=customer.name,
        has_payment_method=customer.default_payment_method_id is not None,
        created_at=customer.created_at.isoformat(),
    )


@router.get("/customers/me", response_model=CustomerResponse)
async def get_current_customer(owner_id: str = Depends(get_current_owner)):
    """Get current user's Stripe customer."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    return CustomerResponse(
        customer_id=customer.customer_id,
        owner_id=customer.owner_id,
        email=customer.email,
        name=customer.name,
        has_payment_method=customer.default_payment_method_id is not None,
        created_at=customer.created_at.isoformat(),
    )


@router.patch("/customers/me", response_model=CustomerResponse)
async def update_current_customer(
    request: UpdateCustomerRequest,
    owner_id: str = Depends(get_current_owner),
):
    """Update current user's Stripe customer."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    updated = await stripe_service.update_customer(
        customer_id=customer.customer_id,
        email=request.email,
        name=request.name,
    )

    return CustomerResponse(
        customer_id=updated.customer_id,
        owner_id=updated.owner_id,
        email=updated.email,
        name=updated.name,
        has_payment_method=updated.default_payment_method_id is not None,
        created_at=updated.created_at.isoformat(),
    )


# ==================== PAYMENT METHOD ENDPOINTS ====================


@router.get("/payment-methods", response_model=list[PaymentMethodResponse])
async def list_payment_methods(owner_id: str = Depends(get_current_owner)):
    """List payment methods for current user."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    methods = await stripe_service.get_payment_methods(customer.customer_id)

    return [
        PaymentMethodResponse(
            payment_method_id=pm.payment_method_id,
            type=pm.type,
            card_brand=pm.card_brand,
            card_last4=pm.card_last4,
            card_exp_month=pm.card_exp_month,
            card_exp_year=pm.card_exp_year,
            display_name=pm.display_name,
        )
        for pm in methods
    ]


@router.post("/payment-methods/setup", response_model=SetupIntentResponse)
async def create_setup_intent(owner_id: str = Depends(get_current_owner)):
    """Create a SetupIntent for adding a new payment method."""
    stripe_service = get_stripe_service()

    if not stripe_service.is_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(
            status_code=400, detail="Customer not found. Create customer first."
        )

    result = await stripe_service.create_setup_intent(customer.customer_id)

    return SetupIntentResponse(
        client_secret=result["client_secret"],
        setup_intent_id=result["setup_intent_id"],
    )


@router.post("/payment-methods/{payment_method_id}/set-default")
async def set_default_payment_method(
    payment_method_id: str,
    owner_id: str = Depends(get_current_owner),
):
    """Set a payment method as default."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    await stripe_service.update_payment_method(customer.customer_id, payment_method_id)

    return {"status": "success", "default_payment_method_id": payment_method_id}


@router.delete("/payment-methods/{payment_method_id}")
async def delete_payment_method(
    payment_method_id: str,
    owner_id: str = Depends(get_current_owner),
):
    """Delete a payment method."""
    stripe_service = get_stripe_service()

    success = await stripe_service.delete_payment_method(payment_method_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete payment method")

    return {"status": "deleted", "payment_method_id": payment_method_id}


# ==================== SUBSCRIPTION ENDPOINTS ====================


@router.get("/subscriptions/current", response_model=SubscriptionResponse | None)
async def get_current_subscription(owner_id: str = Depends(get_current_owner)):
    """Get current user's active subscription."""
    stripe_service = get_stripe_service()

    subscription = await stripe_service.get_subscription_by_owner(owner_id)
    if not subscription:
        return None

    return SubscriptionResponse(
        subscription_id=subscription.subscription_id,
        status=subscription.status.value,
        tier=subscription.tier,
        interval=subscription.interval.value,
        current_period_start=subscription.current_period_start.isoformat(),
        current_period_end=subscription.current_period_end.isoformat(),
        cancel_at_period_end=subscription.cancel_at_period_end,
        trial_end=(
            subscription.trial_end.isoformat() if subscription.trial_end else None
        ),
        days_until_renewal=subscription.days_until_renewal,
    )


@router.post("/subscriptions", response_model=SubscriptionResponse)
async def create_subscription(
    request: CreateSubscriptionRequest,
    owner_id: str = Depends(get_current_owner),
):
    """Create a new subscription."""
    stripe_service = get_stripe_service()

    if not stripe_service.is_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    # Get customer
    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(
            status_code=400, detail="Customer not found. Create customer first."
        )

    if not customer.default_payment_method_id:
        raise HTTPException(status_code=400, detail="No payment method on file")

    # Determine tier from price_id
    tier = get_tier_from_price_id(request.price_id)

    subscription = await stripe_service.create_subscription(
        customer_id=customer.customer_id,
        price_id=request.price_id,
        tier=tier,
        trial_days=request.trial_days,
    )

    return SubscriptionResponse(
        subscription_id=subscription.subscription_id,
        status=subscription.status.value,
        tier=subscription.tier,
        interval=subscription.interval.value,
        current_period_start=subscription.current_period_start.isoformat(),
        current_period_end=subscription.current_period_end.isoformat(),
        cancel_at_period_end=subscription.cancel_at_period_end,
        trial_end=(
            subscription.trial_end.isoformat() if subscription.trial_end else None
        ),
        days_until_renewal=subscription.days_until_renewal,
    )


@router.post(
    "/subscriptions/{subscription_id}/change", response_model=SubscriptionResponse
)
async def change_subscription(
    subscription_id: str,
    request: ChangeSubscriptionRequest,
    owner_id: str = Depends(get_current_owner),
):
    """Upgrade or downgrade subscription."""
    stripe_service = get_stripe_service()

    # Verify ownership
    subscription = await stripe_service.get_subscription(subscription_id)
    if not subscription or subscription.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Subscription not found")

    updated = await stripe_service.change_subscription(
        subscription_id=subscription_id,
        new_price_id=request.price_id,
        new_tier=request.tier,
    )

    return SubscriptionResponse(
        subscription_id=updated.subscription_id,
        status=updated.status.value,
        tier=updated.tier,
        interval=updated.interval.value,
        current_period_start=updated.current_period_start.isoformat(),
        current_period_end=updated.current_period_end.isoformat(),
        cancel_at_period_end=updated.cancel_at_period_end,
        trial_end=updated.trial_end.isoformat() if updated.trial_end else None,
        days_until_renewal=updated.days_until_renewal,
    )


@router.post("/subscriptions/{subscription_id}/cancel")
async def cancel_subscription(
    subscription_id: str,
    at_period_end: bool = True,
    owner_id: str = Depends(get_current_owner),
):
    """Cancel subscription."""
    stripe_service = get_stripe_service()

    # Verify ownership
    subscription = await stripe_service.get_subscription(subscription_id)
    if not subscription or subscription.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Subscription not found")

    await stripe_service.cancel_subscription(
        subscription_id=subscription_id,
        at_period_end=at_period_end,
    )

    return {
        "status": "canceled",
        "at_period_end": at_period_end,
        "subscription_id": subscription_id,
    }


@router.post("/subscriptions/{subscription_id}/reactivate")
async def reactivate_subscription(
    subscription_id: str,
    owner_id: str = Depends(get_current_owner),
):
    """Reactivate a subscription scheduled for cancellation."""
    stripe_service = get_stripe_service()

    # Verify ownership
    subscription = await stripe_service.get_subscription(subscription_id)
    if not subscription or subscription.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Subscription not found")

    if not subscription.cancel_at_period_end:
        raise HTTPException(
            status_code=400, detail="Subscription is not scheduled for cancellation"
        )

    await stripe_service.reactivate_subscription(subscription_id)

    return {"status": "reactivated", "subscription_id": subscription_id}


# ==================== CHECKOUT ENDPOINTS ====================


@router.post("/checkout")
async def create_checkout(
    request: CheckoutRequest,
    owner_id: str = Depends(get_current_owner),
):
    """Create Stripe Checkout session."""
    stripe_service = get_stripe_service()

    if not stripe_service.is_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    url = await stripe_service.create_checkout_session(
        owner_id=owner_id,
        price_id=request.price_id,
        success_url=request.success_url,
        cancel_url=request.cancel_url,
        trial_days=request.trial_days,
    )

    return {"checkout_url": url}


@router.post("/portal")
async def create_portal(
    request: PortalRequest,
    owner_id: str = Depends(get_current_owner),
):
    """Create Stripe Billing Portal session."""
    stripe_service = get_stripe_service()

    if not stripe_service.is_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    url = await stripe_service.create_billing_portal_session(
        customer_id=customer.customer_id,
        return_url=request.return_url,
    )

    return {"portal_url": url}


# ==================== INVOICE ENDPOINTS ====================


@router.get("/invoices", response_model=list[InvoiceResponse])
async def list_invoices(
    limit: int = 10,
    owner_id: str = Depends(get_current_owner),
):
    """List invoices for current user."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        return []

    invoices = await stripe_service.get_invoices(
        customer_id=customer.customer_id,
        limit=limit,
    )

    return [
        InvoiceResponse(
            invoice_id=inv.invoice_id,
            status=inv.status,
            amount_due=inv.amount_due,
            amount_paid=inv.amount_paid,
            amount_due_dollars=inv.amount_due_dollars,
            currency=inv.currency,
            created_at=inv.created_at.isoformat(),
            due_date=inv.due_date.isoformat() if inv.due_date else None,
            invoice_pdf=inv.invoice_pdf,
            hosted_invoice_url=inv.hosted_invoice_url,
        )
        for inv in invoices
    ]


@router.get("/invoices/upcoming", response_model=InvoiceResponse | None)
async def get_upcoming_invoice(owner_id: str = Depends(get_current_owner)):
    """Get upcoming invoice for current user."""
    stripe_service = get_stripe_service()

    customer = await stripe_service.get_customer_by_owner(owner_id)
    if not customer:
        return None

    invoice = await stripe_service.get_upcoming_invoice(customer.customer_id)
    if not invoice:
        return None

    return InvoiceResponse(
        invoice_id=invoice.invoice_id,
        status=invoice.status,
        amount_due=invoice.amount_due,
        amount_paid=invoice.amount_paid,
        amount_due_dollars=invoice.amount_due_dollars,
        currency=invoice.currency,
        created_at=invoice.created_at.isoformat(),
        due_date=invoice.due_date.isoformat() if invoice.due_date else None,
        invoice_pdf=invoice.invoice_pdf,
        hosted_invoice_url=invoice.hosted_invoice_url,
    )


# ==================== WEBHOOK ENDPOINT ====================


@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.

    This endpoint receives webhooks from Stripe for subscription lifecycle,
    payment events, and customer updates.
    """
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    handler = get_webhook_handler()

    # Verify signature
    event = handler.verify_signature(payload, signature)
    if not event:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # Process event
    result = await handler.handle_event(event)

    return result


# ==================== ADMIN ENDPOINTS ====================


@router.post("/admin/setup-products", include_in_schema=False)
async def setup_stripe_products(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
):
    """
    Create Stripe products and prices.

    This is a one-time setup endpoint. Run this to create products
    in your Stripe account, then save the IDs to your configuration.

    Requires admin API key.
    """
    settings = get_settings()

    # Verify admin key
    if not settings.server.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API not configured")

    if x_admin_key != settings.server.admin_api_key.get_secret_value():
        raise HTTPException(status_code=403, detail="Invalid admin key")

    stripe_service = get_stripe_service()

    if not stripe_service.is_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    # Create products
    products = await stripe_service.create_products()

    # Create prices
    prices = await stripe_service.create_prices(products)

    return {
        "status": "success",
        "message": "Products and prices created. Add these IDs to your configuration.",
        "products": products,
        "prices": prices,
        "configuration_example": {
            "STRIPE_PRODUCT_FREE_ID": products.get("free"),
            "STRIPE_PRODUCT_STARTER_ID": products.get("starter"),
            "STRIPE_PRODUCT_PRO_ID": products.get("pro"),
            "STRIPE_PRODUCT_ENTERPRISE_ID": products.get("enterprise"),
            "STRIPE_PRICE_STARTER_MONTHLY_ID": prices.get("starter_monthly"),
            "STRIPE_PRICE_STARTER_YEARLY_ID": prices.get("starter_yearly"),
            "STRIPE_PRICE_PRO_MONTHLY_ID": prices.get("pro_monthly"),
            "STRIPE_PRICE_PRO_YEARLY_ID": prices.get("pro_yearly"),
            "STRIPE_PRICE_ENTERPRISE_MONTHLY_ID": prices.get("enterprise_monthly"),
            "STRIPE_PRICE_ENTERPRISE_YEARLY_ID": prices.get("enterprise_yearly"),
            "STRIPE_PRICE_METERED_TOKENS_ID": prices.get("token_usage"),
        },
    }


@router.get("/admin/products", include_in_schema=False)
async def list_stripe_products(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
):
    """List all Orchestral products in Stripe."""
    settings = get_settings()

    if not settings.server.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API not configured")

    if x_admin_key != settings.server.admin_api_key.get_secret_value():
        raise HTTPException(status_code=403, detail="Invalid admin key")

    stripe_service = get_stripe_service()

    products = await stripe_service.get_products()

    return {"products": [p.to_dict() for p in products]}


@router.get("/admin/prices", include_in_schema=False)
async def list_stripe_prices(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
):
    """List all prices in Stripe."""
    settings = get_settings()

    if not settings.server.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API not configured")

    if x_admin_key != settings.server.admin_api_key.get_secret_value():
        raise HTTPException(status_code=403, detail="Invalid admin key")

    stripe_service = get_stripe_service()

    prices = await stripe_service.get_prices()

    return {"prices": [p.to_dict() for p in prices]}
