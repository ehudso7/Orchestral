"""
Dashboard API endpoints for Orchestral.

Provides endpoints for the dashboard to display usage statistics and metrics.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from orchestral.api.auth import get_current_user
from orchestral.api.db import db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get dashboard statistics for the current user."""

    # Get user's API keys
    api_keys = db.get_user_api_keys(current_user["id"])

    # Mock data for development - in production, this would come from actual usage tracking
    today = datetime.now(timezone.utc).date()

    # Generate some realistic-looking mock data
    api_calls_today = random.randint(50, 500)
    tokens_used = random.randint(10000, 100000)
    monthly_cost = round(random.uniform(0, 50), 2)

    return {
        "user_id": current_user["id"],
        "tier": current_user.get("tier", "free"),
        "api_calls_today": api_calls_today,
        "tokens_used": tokens_used,
        "tokens_limit": 100000 if current_user.get("tier") == "free" else 1000000,
        "monthly_cost": monthly_cost,
        "api_keys_count": len(api_keys),
        "requests_today": api_calls_today,
        "cost_usd": monthly_cost,
        "usage_history": generate_usage_history(),
    }


@router.get("/usage")
async def get_usage_data(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed usage data for the current user."""

    # Mock usage data for development
    return {
        "daily": {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "requests": random.randint(50, 500),
            "tokens": random.randint(10000, 100000),
            "cost_usd": round(random.uniform(0, 10), 2),
        },
        "monthly": {
            "month": datetime.now(timezone.utc).strftime("%Y-%m"),
            "requests": random.randint(1000, 10000),
            "tokens": random.randint(100000, 1000000),
            "cost_usd": round(random.uniform(0, 50), 2),
        },
        "tier": current_user.get("tier", "free"),
        "limits": {
            "requests_per_minute": 10 if current_user.get("tier") == "free" else 100,
            "requests_per_day": 1000 if current_user.get("tier") == "free" else 10000,
            "tokens_per_month": 100000 if current_user.get("tier") == "free" else 1000000,
            "monthly_budget_usd": 0 if current_user.get("tier") == "free" else 100,
        }
    }


@router.get("/billing/status")
async def get_billing_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get billing status for the current user."""

    return {
        "current_tier": current_user.get("tier", "free"),
        "subscription_status": current_user.get("subscription_status", "active"),
        "has_customer": False,  # No Stripe customer in dev
        "payment_methods": [],
        "invoices": [],
    }


@router.get("/billing/invoices")
async def get_invoices(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> list[Dict[str, Any]]:
    """Get billing invoices for the current user."""

    # Return empty list for development
    return []


@router.get("/billing/payment-methods")
async def get_payment_methods(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> list[Dict[str, Any]]:
    """Get payment methods for the current user."""

    # Return empty list for development
    return []


def generate_usage_history() -> list[Dict[str, Any]]:
    """Generate mock usage history for the chart."""
    history = []
    for i in range(7):
        date = datetime.now(timezone.utc) - timedelta(days=6-i)
        history.append({
            "date": date.strftime("%a"),
            "calls": random.randint(30, 100),
            "tokens": random.randint(5000, 20000),
        })
    return history