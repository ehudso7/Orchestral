"""
Cost tracking and budget management for AI API usage.

Provides real-time cost tracking, budget alerts, and spend optimization insights.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    """Budget alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BudgetPeriod(str, Enum):
    """Budget time periods."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BudgetConfig(BaseModel):
    """Budget configuration."""

    daily_limit: float | None = Field(default=None, description="Daily spend limit in USD")
    weekly_limit: float | None = Field(default=None, description="Weekly spend limit in USD")
    monthly_limit: float | None = Field(default=None, description="Monthly spend limit in USD")
    per_request_limit: float | None = Field(default=None, description="Max cost per request")
    warning_threshold: float = Field(default=0.8, description="Alert at this % of limit")
    critical_threshold: float = Field(default=0.95, description="Critical alert threshold")
    model_limits: dict[str, float] = Field(default_factory=dict, description="Per-model limits")


@dataclass
class CostEntry:
    """A single cost entry."""

    id: str
    timestamp: datetime
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    request_type: str = "completion"
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": f"${self.input_cost:.6f}",
            "output_cost": f"${self.output_cost:.6f}",
            "total_cost": f"${self.total_cost:.6f}",
            "request_type": self.request_type,
            "cached": self.cached,
        }


@dataclass
class BudgetAlert:
    """A budget alert notification."""

    level: AlertLevel
    message: str
    period: BudgetPeriod
    current_spend: float
    limit: float
    percentage: float
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "period": self.period.value,
            "current_spend": f"${self.current_spend:.4f}",
            "limit": f"${self.limit:.2f}",
            "percentage": f"{self.percentage:.1%}",
            "triggered_at": self.triggered_at.isoformat(),
        }


@dataclass
class CostSummary:
    """Cost summary for a time period."""

    period: str
    start_time: datetime
    end_time: datetime
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    by_model: dict[str, float]
    by_provider: dict[str, float]
    avg_cost_per_request: float
    cache_savings: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_cost": f"${self.total_cost:.4f}",
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_model": {k: f"${v:.4f}" for k, v in self.by_model.items()},
            "by_provider": {k: f"${v:.4f}" for k, v in self.by_provider.items()},
            "avg_cost_per_request": f"${self.avg_cost_per_request:.6f}",
            "cache_savings": f"${self.cache_savings:.4f}",
        }


class CostTracker:
    """
    Tracks and analyzes API costs in real-time.

    Features:
    - Real-time cost tracking per request
    - Budget limits and alerts
    - Cost breakdown by model/provider
    - Cache savings tracking
    - Historical cost analysis

    Example:
        tracker = CostTracker(budget=BudgetConfig(daily_limit=10.0))

        # Track a request
        entry = tracker.track(
            model="gpt-4o",
            provider="openai",
            input_tokens=500,
            output_tokens=200,
        )

        # Check budget status
        alerts = tracker.check_budgets()
        summary = tracker.get_summary("daily")
    """

    # Pricing per million tokens (as of late 2024)
    PRICING = {
        # OpenAI
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-5.1": {"input": 5.00, "output": 15.00},
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        # Anthropic
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        # Google
        "gemini-3-pro-preview": {"input": 3.50, "output": 10.50},
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    }

    def __init__(
        self,
        budget: BudgetConfig | None = None,
        alert_callback: Callable[[BudgetAlert], None] | None = None,
    ):
        self.budget = budget or BudgetConfig()
        self.alert_callback = alert_callback
        self._entries: list[CostEntry] = []
        self._cache_savings: float = 0.0
        self._entry_counter = 0
        self._lock = asyncio.Lock()
        self._last_alerts: dict[str, datetime] = {}

    def _get_pricing(self, model: str) -> tuple[float, float]:
        """Get pricing for a model (per million tokens)."""
        if model in self.PRICING:
            p = self.PRICING[model]
            return p["input"], p["output"]
        # Default pricing for unknown models
        return 1.0, 3.0

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, float, float]:
        """Calculate cost for a request."""
        input_rate, output_rate = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        return input_cost, output_cost, input_cost + output_cost

    def track(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str = "completion",
        cached: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """
        Track a new API request.

        Args:
            model: Model identifier
            provider: Provider name (openai, anthropic, google)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_type: Type of request (completion, embedding, etc.)
            cached: Whether response was from cache
            metadata: Additional metadata

        Returns:
            CostEntry for the tracked request
        """
        input_cost, output_cost, total_cost = self._calculate_cost(
            model, input_tokens, output_tokens
        )

        if cached:
            # Track what would have been spent
            self._cache_savings += total_cost
            total_cost = 0
            input_cost = 0
            output_cost = 0

        self._entry_counter += 1
        entry = CostEntry(
            id=f"cost_{self._entry_counter}",
            timestamp=datetime.now(timezone.utc),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            request_type=request_type,
            cached=cached,
            metadata=metadata or {},
        )

        self._entries.append(entry)

        # Check budgets after tracking
        self._check_and_alert()

        return entry

    def _get_entries_in_period(
        self, period: BudgetPeriod
    ) -> list[CostEntry]:
        """Get entries within a budget period."""
        now = datetime.now(timezone.utc)

        if period == BudgetPeriod.HOURLY:
            cutoff = now - timedelta(hours=1)
        elif period == BudgetPeriod.DAILY:
            cutoff = now - timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        else:  # MONTHLY
            cutoff = now - timedelta(days=30)

        return [e for e in self._entries if e.timestamp >= cutoff]

    def _get_period_spend(self, period: BudgetPeriod) -> float:
        """Get total spend for a period."""
        entries = self._get_entries_in_period(period)
        return sum(e.total_cost for e in entries)

    def _check_and_alert(self) -> None:
        """Check budgets and trigger alerts if needed."""
        alerts = self.check_budgets()
        if self.alert_callback:
            for alert in alerts:
                # Debounce alerts (don't repeat within 5 minutes)
                key = f"{alert.period}_{alert.level}"
                if key in self._last_alerts:
                    if (datetime.now(timezone.utc) - self._last_alerts[key]).seconds < 300:
                        continue
                self._last_alerts[key] = datetime.now(timezone.utc)
                self.alert_callback(alert)

    def check_budgets(self) -> list[BudgetAlert]:
        """Check all budget limits and return any alerts."""
        alerts = []

        checks = [
            (BudgetPeriod.DAILY, self.budget.daily_limit),
            (BudgetPeriod.WEEKLY, self.budget.weekly_limit),
            (BudgetPeriod.MONTHLY, self.budget.monthly_limit),
        ]

        for period, limit in checks:
            if limit is None:
                continue

            spend = self._get_period_spend(period)
            percentage = spend / limit

            if percentage >= self.budget.critical_threshold:
                alerts.append(BudgetAlert(
                    level=AlertLevel.CRITICAL,
                    message=f"{period.value.title()} budget critical: ${spend:.2f} of ${limit:.2f}",
                    period=period,
                    current_spend=spend,
                    limit=limit,
                    percentage=percentage,
                ))
            elif percentage >= self.budget.warning_threshold:
                alerts.append(BudgetAlert(
                    level=AlertLevel.WARNING,
                    message=f"{period.value.title()} budget warning: ${spend:.2f} of ${limit:.2f}",
                    period=period,
                    current_spend=spend,
                    limit=limit,
                    percentage=percentage,
                ))

        return alerts

    def get_summary(self, period: str = "daily") -> CostSummary:
        """
        Get cost summary for a time period.

        Args:
            period: "hourly", "daily", "weekly", or "monthly"

        Returns:
            CostSummary with aggregated statistics
        """
        period_enum = BudgetPeriod(period)
        entries = self._get_entries_in_period(period_enum)

        now = datetime.now(timezone.utc)
        if period_enum == BudgetPeriod.HOURLY:
            start_time = now - timedelta(hours=1)
        elif period_enum == BudgetPeriod.DAILY:
            start_time = now - timedelta(days=1)
        elif period_enum == BudgetPeriod.WEEKLY:
            start_time = now - timedelta(weeks=1)
        else:
            start_time = now - timedelta(days=30)

        by_model: dict[str, float] = defaultdict(float)
        by_provider: dict[str, float] = defaultdict(float)
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for entry in entries:
            by_model[entry.model] += entry.total_cost
            by_provider[entry.provider] += entry.total_cost
            total_input += entry.input_tokens
            total_output += entry.output_tokens
            total_cost += entry.total_cost

        return CostSummary(
            period=period,
            start_time=start_time,
            end_time=now,
            total_cost=total_cost,
            total_requests=len(entries),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            by_model=dict(by_model),
            by_provider=dict(by_provider),
            avg_cost_per_request=total_cost / len(entries) if entries else 0,
            cache_savings=self._cache_savings,
        )

    def get_model_costs(self) -> dict[str, dict[str, Any]]:
        """Get cost breakdown by model."""
        model_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
        )

        for entry in self._entries:
            stats = model_stats[entry.model]
            stats["requests"] += 1
            stats["cost"] += entry.total_cost
            stats["input_tokens"] += entry.input_tokens
            stats["output_tokens"] += entry.output_tokens

        # Add avg cost per request
        for model, stats in model_stats.items():
            if stats["requests"] > 0:
                stats["avg_cost"] = stats["cost"] / stats["requests"]
            stats["cost"] = f"${stats['cost']:.4f}"
            stats["avg_cost"] = f"${stats.get('avg_cost', 0):.6f}"

        return dict(model_stats)

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost before making a request."""
        _, _, total = self._calculate_cost(model, input_tokens, output_tokens)
        return total

    def get_optimization_suggestions(self) -> list[str]:
        """Get suggestions for cost optimization."""
        suggestions = []

        model_costs = self.get_model_costs()

        # Check if expensive models are overused
        expensive_models = {"gpt-4-turbo", "claude-3-opus-20240229", "o1"}
        for model in expensive_models:
            if model in model_costs:
                stats = model_costs[model]
                if stats["requests"] > 100:
                    suggestions.append(
                        f"Consider using a smaller model instead of {model} for simple tasks"
                    )

        # Check cache hit rate
        if self._cache_savings > 0:
            total_would_have_spent = sum(e.total_cost for e in self._entries) + self._cache_savings
            if total_would_have_spent > 0:
                cache_rate = self._cache_savings / total_would_have_spent
                if cache_rate < 0.2:
                    suggestions.append(
                        "Enable or tune semantic caching to reduce costs (current savings: "
                        f"{cache_rate:.1%})"
                    )

        if not suggestions:
            suggestions.append("Cost usage appears optimized")

        return suggestions

    def export_entries(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Export recent cost entries for analysis."""
        return [e.to_dict() for e in self._entries[-limit:]]
