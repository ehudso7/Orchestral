"""
Metrics collection and monitoring for Orchestral.

Provides application metrics for monitoring performance and usage.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    timestamp: datetime
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    error_type: str | None = None


@dataclass
class ProviderMetrics:
    """Aggregated metrics for a provider."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class Metrics:
    """
    Thread-safe metrics collector for Orchestral.

    Collects and aggregates metrics for all AI model requests.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of requests to keep in history
        """
        self._lock = Lock()
        self._max_history = max_history
        self._requests: list[RequestMetrics] = []
        self._provider_metrics: dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self._model_metrics: dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self._start_time = datetime.utcnow()

    def record_completion(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Record a successful completion request.

        Args:
            provider: Provider name
            model: Model ID
            latency_ms: Request latency in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        with self._lock:
            request = RequestMetrics(
                timestamp=datetime.utcnow(),
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )
            self._add_request(request)

            # Update provider metrics
            pm = self._provider_metrics[provider]
            pm.total_requests += 1
            pm.successful_requests += 1
            pm.total_latency_ms += latency_ms
            pm.total_input_tokens += input_tokens
            pm.total_output_tokens += output_tokens

            # Update model metrics
            mm = self._model_metrics[model]
            mm.total_requests += 1
            mm.successful_requests += 1
            mm.total_latency_ms += latency_ms
            mm.total_input_tokens += input_tokens
            mm.total_output_tokens += output_tokens

    def record_error(
        self,
        model: str,
        error_type: str,
        provider: str | None = None,
    ) -> None:
        """
        Record a failed request.

        Args:
            model: Model ID
            error_type: Type of error
            provider: Provider name (optional)
        """
        with self._lock:
            request = RequestMetrics(
                timestamp=datetime.utcnow(),
                provider=provider or "unknown",
                model=model,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_type=error_type,
            )
            self._add_request(request)

            if provider:
                pm = self._provider_metrics[provider]
                pm.total_requests += 1
                pm.failed_requests += 1
                pm.errors[error_type] += 1

            mm = self._model_metrics[model]
            mm.total_requests += 1
            mm.failed_requests += 1
            mm.errors[error_type] += 1

    def _add_request(self, request: RequestMetrics) -> None:
        """Add request to history, maintaining max size."""
        self._requests.append(request)
        if len(self._requests) > self._max_history:
            self._requests = self._requests[-self._max_history:]

    def get_summary(self) -> dict[str, Any]:
        """
        Get metrics summary.

        Returns:
            Dictionary with aggregated metrics
        """
        with self._lock:
            total_requests = sum(pm.total_requests for pm in self._provider_metrics.values())
            total_successful = sum(pm.successful_requests for pm in self._provider_metrics.values())
            total_tokens = sum(
                pm.total_input_tokens + pm.total_output_tokens
                for pm in self._provider_metrics.values()
            )

            uptime = (datetime.utcnow() - self._start_time).total_seconds()

            return {
                "uptime_seconds": uptime,
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_requests - total_successful,
                "success_rate": total_successful / total_requests if total_requests > 0 else 0,
                "total_tokens": total_tokens,
                "requests_per_minute": total_requests / (uptime / 60) if uptime > 0 else 0,
                "providers": {
                    name: {
                        "total_requests": pm.total_requests,
                        "success_rate": pm.success_rate,
                        "avg_latency_ms": pm.avg_latency_ms,
                        "total_tokens": pm.total_input_tokens + pm.total_output_tokens,
                        "errors": dict(pm.errors),
                    }
                    for name, pm in self._provider_metrics.items()
                },
                "models": {
                    name: {
                        "total_requests": mm.total_requests,
                        "success_rate": mm.success_rate,
                        "avg_latency_ms": mm.avg_latency_ms,
                    }
                    for name, mm in self._model_metrics.items()
                },
            }

    def get_recent_requests(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent request history.

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of request dictionaries
        """
        with self._lock:
            recent = self._requests[-limit:]
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "provider": r.provider,
                    "model": r.model,
                    "latency_ms": r.latency_ms,
                    "tokens": r.input_tokens + r.output_tokens,
                    "success": r.success,
                    "error": r.error_type,
                }
                for r in recent
            ]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._requests.clear()
            self._provider_metrics.clear()
            self._model_metrics.clear()
            self._start_time = datetime.utcnow()


# Global metrics instance
metrics = Metrics()
