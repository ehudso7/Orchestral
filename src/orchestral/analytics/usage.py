"""
Usage analytics and pattern detection.

Analyzes usage patterns to provide insights for optimization.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

import math


class InsightType(str, Enum):
    """Types of usage insights."""

    PERFORMANCE = "performance"
    COST = "cost"
    RELIABILITY = "reliability"
    PATTERN = "pattern"
    RECOMMENDATION = "recommendation"


@dataclass
class UsageEvent:
    """A single usage event."""

    timestamp: datetime
    model: str
    provider: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    error_type: str | None = None
    task_category: str | None = None
    quality_score: float | None = None


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""

    model: str
    provider: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_quality_score: float | None
    tokens_per_second: float
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "avg_quality_score": round(self.avg_quality_score, 3) if self.avg_quality_score else None,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "error_rate": f"{self.error_rate:.2%}",
        }


@dataclass
class UsagePattern:
    """Detected usage pattern."""

    pattern_type: str
    description: str
    frequency: int
    confidence: float
    time_range: tuple[datetime, datetime]
    affected_models: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "confidence": f"{self.confidence:.0%}",
            "time_range": (
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat(),
            ),
            "affected_models": self.affected_models,
        }


@dataclass
class UsageInsight:
    """An actionable insight from usage data."""

    insight_type: InsightType
    title: str
    description: str
    impact: str  # "high", "medium", "low"
    action: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.insight_type.value,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "action": self.action,
            "data": self.data,
        }


class UsageAnalytics:
    """
    Analyzes usage patterns and provides insights.

    Features:
    - Model performance tracking
    - Pattern detection
    - Actionable insights
    - Historical trend analysis

    Example:
        analytics = UsageAnalytics()

        # Record events
        analytics.record(
            model="gpt-4o",
            provider="openai",
            latency_ms=1500,
            input_tokens=500,
            output_tokens=200,
            success=True,
        )

        # Get insights
        insights = analytics.get_insights()
        performance = analytics.get_model_performance("gpt-4o")
    """

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self._events: list[UsageEvent] = []

    def _cleanup_old_events(self) -> None:
        """Remove events older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        self._events = [e for e in self._events if e.timestamp >= cutoff]

    def record(
        self,
        model: str,
        provider: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error_type: str | None = None,
        task_category: str | None = None,
        quality_score: float | None = None,
    ) -> UsageEvent:
        """Record a usage event."""
        event = UsageEvent(
            timestamp=datetime.now(timezone.utc),
            model=model,
            provider=provider,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error_type=error_type,
            task_category=task_category,
            quality_score=quality_score,
        )
        self._events.append(event)

        # Periodic cleanup
        if len(self._events) % 1000 == 0:
            self._cleanup_old_events()

        return event

    def _percentile(self, values: list[float], p: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def get_model_performance(
        self, model: str, hours: int = 24
    ) -> ModelPerformance | None:
        """Get performance metrics for a specific model."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events = [e for e in self._events if e.model == model and e.timestamp >= cutoff]

        if not events:
            return None

        latencies = [e.latency_ms for e in events if e.success]
        quality_scores = [e.quality_score for e in events if e.quality_score is not None]
        successful = [e for e in events if e.success]

        # Calculate tokens per second
        total_output_tokens = sum(e.output_tokens for e in successful)
        total_time_seconds = sum(e.latency_ms for e in successful) / 1000
        tps = total_output_tokens / total_time_seconds if total_time_seconds > 0 else 0

        return ModelPerformance(
            model=model,
            provider=events[0].provider,
            total_requests=len(events),
            successful_requests=len(successful),
            failed_requests=len(events) - len(successful),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency_ms=self._percentile(latencies, 50),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            avg_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else None,
            tokens_per_second=tps,
            error_rate=1 - (len(successful) / len(events)) if events else 0,
        )

    def get_all_model_performance(self, hours: int = 24) -> list[ModelPerformance]:
        """Get performance metrics for all models."""
        models = set(e.model for e in self._events)
        performances = []
        for model in models:
            perf = self.get_model_performance(model, hours)
            if perf:
                performances.append(perf)
        return sorted(performances, key=lambda p: p.total_requests, reverse=True)

    def detect_patterns(self) -> list[UsagePattern]:
        """Detect usage patterns in the data."""
        patterns = []
        now = datetime.now(timezone.utc)

        # Pattern 1: Peak usage hours
        hourly_counts: dict[int, int] = defaultdict(int)
        for event in self._events:
            hourly_counts[event.timestamp.hour] += 1

        if hourly_counts:
            peak_hour = max(hourly_counts, key=hourly_counts.get)  # type: ignore
            peak_count = hourly_counts[peak_hour]
            avg_count = sum(hourly_counts.values()) / len(hourly_counts)
            if peak_count > avg_count * 2:
                patterns.append(UsagePattern(
                    pattern_type="peak_usage",
                    description=f"Peak usage at hour {peak_hour}:00 UTC ({peak_count} requests)",
                    frequency=peak_count,
                    confidence=min(0.95, peak_count / (avg_count * 2)),
                    time_range=(now - timedelta(days=7), now),
                    affected_models=list(set(e.model for e in self._events)),
                ))

        # Pattern 2: Error spikes
        model_errors: dict[str, list[datetime]] = defaultdict(list)
        for event in self._events:
            if not event.success:
                model_errors[event.model].append(event.timestamp)

        for model, error_times in model_errors.items():
            if len(error_times) >= 5:
                # Check for clustering
                error_times.sort()
                for i in range(len(error_times) - 4):
                    window = error_times[i : i + 5]
                    time_span = (window[-1] - window[0]).total_seconds()
                    if time_span < 300:  # 5 errors in 5 minutes
                        patterns.append(UsagePattern(
                            pattern_type="error_spike",
                            description=f"Error spike detected for {model}",
                            frequency=5,
                            confidence=0.9,
                            time_range=(window[0], window[-1]),
                            affected_models=[model],
                        ))
                        break

        # Pattern 3: Model preference shifts
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)

        recent_models = [e.model for e in self._events if e.timestamp >= day_ago]
        older_models = [e.model for e in self._events if week_ago <= e.timestamp < day_ago]

        if recent_models and older_models:
            recent_counts = defaultdict(int)
            older_counts = defaultdict(int)
            for m in recent_models:
                recent_counts[m] += 1
            for m in older_models:
                older_counts[m] += 1

            for model in set(recent_counts.keys()) | set(older_counts.keys()):
                recent_pct = recent_counts[model] / len(recent_models)
                older_pct = older_counts[model] / len(older_models) if older_models else 0
                if abs(recent_pct - older_pct) > 0.2:
                    direction = "increased" if recent_pct > older_pct else "decreased"
                    patterns.append(UsagePattern(
                        pattern_type="model_shift",
                        description=f"{model} usage has {direction} significantly",
                        frequency=recent_counts[model],
                        confidence=abs(recent_pct - older_pct),
                        time_range=(day_ago, now),
                        affected_models=[model],
                    ))

        return patterns

    def get_insights(self) -> list[UsageInsight]:
        """Generate actionable insights from usage data."""
        insights = []
        performances = self.get_all_model_performance()

        # Insight 1: High error rates
        for perf in performances:
            if perf.error_rate > 0.1:
                insights.append(UsageInsight(
                    insight_type=InsightType.RELIABILITY,
                    title=f"High error rate for {perf.model}",
                    description=f"{perf.model} has a {perf.error_rate:.1%} error rate over the last 24 hours",
                    impact="high",
                    action=f"Consider switching to a fallback model or investigating {perf.provider} API status",
                    data={"model": perf.model, "error_rate": perf.error_rate},
                ))

        # Insight 2: Slow models
        for perf in performances:
            if perf.p95_latency_ms > 10000:  # 10 seconds
                insights.append(UsageInsight(
                    insight_type=InsightType.PERFORMANCE,
                    title=f"High latency for {perf.model}",
                    description=f"P95 latency is {perf.p95_latency_ms:.0f}ms",
                    impact="medium",
                    action="Consider using a faster model or enabling streaming",
                    data={"model": perf.model, "p95_latency_ms": perf.p95_latency_ms},
                ))

        # Insight 3: Quality issues
        for perf in performances:
            if perf.avg_quality_score and perf.avg_quality_score < 0.6:
                insights.append(UsageInsight(
                    insight_type=InsightType.PERFORMANCE,
                    title=f"Low quality scores for {perf.model}",
                    description=f"Average quality score is {perf.avg_quality_score:.2f}",
                    impact="high",
                    action="Consider switching to a more capable model for this use case",
                    data={"model": perf.model, "avg_quality": perf.avg_quality_score},
                ))

        # Insight 4: Model recommendations based on task
        task_performance: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for event in self._events:
            if event.task_category and event.success and event.quality_score:
                task_performance[event.task_category].append(
                    (event.model, event.quality_score)
                )

        for task, model_scores in task_performance.items():
            if len(model_scores) >= 10:
                model_avg: dict[str, list[float]] = defaultdict(list)
                for model, score in model_scores:
                    model_avg[model].append(score)

                best_model = max(
                    model_avg.items(),
                    key=lambda x: sum(x[1]) / len(x[1])
                )
                worst_model = min(
                    model_avg.items(),
                    key=lambda x: sum(x[1]) / len(x[1])
                )

                if best_model[0] != worst_model[0]:
                    best_avg = sum(best_model[1]) / len(best_model[1])
                    worst_avg = sum(worst_model[1]) / len(worst_model[1])
                    if best_avg - worst_avg > 0.15:
                        insights.append(UsageInsight(
                            insight_type=InsightType.RECOMMENDATION,
                            title=f"Model recommendation for {task}",
                            description=f"{best_model[0]} outperforms {worst_model[0]} for {task} tasks",
                            impact="medium",
                            action=f"Consider routing {task} tasks to {best_model[0]}",
                            data={
                                "task": task,
                                "best_model": best_model[0],
                                "best_score": best_avg,
                                "worst_model": worst_model[0],
                                "worst_score": worst_avg,
                            },
                        ))

        # Detect patterns and add insights
        patterns = self.detect_patterns()
        for pattern in patterns:
            if pattern.pattern_type == "error_spike":
                insights.append(UsageInsight(
                    insight_type=InsightType.RELIABILITY,
                    title="Error spike detected",
                    description=pattern.description,
                    impact="high",
                    action="Investigate API issues or implement circuit breaker",
                    data={"pattern": pattern.to_dict()},
                ))

        return insights

    def get_summary(self) -> dict[str, Any]:
        """Get overall usage summary."""
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        recent = [e for e in self._events if e.timestamp >= day_ago]

        return {
            "total_events_24h": len(recent),
            "success_rate": sum(1 for e in recent if e.success) / len(recent) if recent else 0,
            "unique_models": len(set(e.model for e in recent)),
            "avg_latency_ms": sum(e.latency_ms for e in recent) / len(recent) if recent else 0,
            "total_tokens": sum(e.input_tokens + e.output_tokens for e in recent),
            "model_distribution": dict(
                sorted(
                    [(m, sum(1 for e in recent if e.model == m))
                     for m in set(e.model for e in recent)],
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }
