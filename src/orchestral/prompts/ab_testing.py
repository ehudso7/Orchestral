"""
A/B testing framework for prompts and models.

Enables data-driven optimization of prompts and model selection
through controlled experiments with statistical significance tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import math

import structlog

logger = structlog.get_logger()


class ExperimentStatus(str, Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MetricType(str, Enum):
    """Types of metrics to track."""

    LATENCY = "latency"
    COST = "cost"
    TOKENS = "tokens"
    SUCCESS_RATE = "success_rate"
    QUALITY_SCORE = "quality_score"
    USER_RATING = "user_rating"
    CUSTOM = "custom"


@dataclass
class VariantMetrics:
    """Metrics for an experiment variant."""

    impressions: int = 0
    conversions: int = 0  # Success count

    # Performance metrics
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Quality metrics
    total_quality_score: float = 0.0
    quality_samples: int = 0

    # Custom metrics
    custom_metrics: dict[str, float] = field(default_factory=dict)
    custom_counts: dict[str, int] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Get conversion rate."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        if self.impressions == 0:
            return 0.0
        return self.total_latency_ms / self.impressions

    @property
    def avg_cost_usd(self) -> float:
        """Get average cost."""
        if self.impressions == 0:
            return 0.0
        return self.total_cost_usd / self.impressions

    @property
    def avg_quality_score(self) -> float:
        """Get average quality score."""
        if self.quality_samples == 0:
            return 0.0
        return self.total_quality_score / self.quality_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_rate": self.conversion_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_quality_score": self.avg_quality_score,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "custom_metrics": {
                k: v / self.custom_counts.get(k, 1)
                for k, v in self.custom_metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariantMetrics":
        """Create from dictionary."""
        return cls(
            impressions=data.get("impressions", 0),
            conversions=data.get("conversions", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_quality_score=data.get("total_quality_score", 0.0),
            quality_samples=data.get("quality_samples", 0),
            custom_metrics=data.get("custom_metrics", {}),
            custom_counts=data.get("custom_counts", {}),
        )


@dataclass
class Variant:
    """A variant in an A/B test."""

    variant_id: str
    name: str
    weight: float = 1.0  # Relative traffic weight

    # Variant configuration
    prompt_id: str | None = None
    prompt_version: int | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # Metrics
    metrics: VariantMetrics = field(default_factory=VariantMetrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "weight": self.weight,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "config": self.config,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Variant":
        """Create from dictionary."""
        variant = cls(
            variant_id=data["variant_id"],
            name=data["name"],
            weight=data.get("weight", 1.0),
            prompt_id=data.get("prompt_id"),
            prompt_version=data.get("prompt_version"),
            model=data.get("model"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            config=data.get("config", {}),
        )
        if "metrics" in data:
            variant.metrics = VariantMetrics.from_dict(data["metrics"])
        return variant


@dataclass
class ExperimentResult:
    """Statistical results of an experiment."""

    experiment_id: str
    winner: str | None  # variant_id of winner, None if inconclusive
    confidence: float  # Statistical confidence (0-1)
    lift: float  # Improvement percentage vs control
    p_value: float  # Statistical p-value
    sample_size: int
    analysis_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Per-variant results
    variant_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "winner": self.winner,
            "confidence": self.confidence,
            "lift": self.lift,
            "p_value": self.p_value,
            "sample_size": self.sample_size,
            "analysis_time": self.analysis_time.isoformat(),
            "variant_results": self.variant_results,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    experiment_id: str
    name: str
    description: str | None = None
    tenant_id: str = "global"
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Configuration
    variants: list[Variant] = field(default_factory=list)
    control_variant_id: str | None = None  # Baseline variant
    target_sample_size: int = 1000
    min_runtime_hours: int = 24
    confidence_threshold: float = 0.95

    # Primary metric to optimize
    primary_metric: MetricType = MetricType.SUCCESS_RATE

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Results
    result: ExperimentResult | None = None

    def get_variant(self, variant_id: str) -> Variant | None:
        """Get variant by ID."""
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        return None

    def get_control(self) -> Variant | None:
        """Get control variant."""
        if self.control_variant_id:
            return self.get_variant(self.control_variant_id)
        return self.variants[0] if self.variants else None

    @property
    def total_impressions(self) -> int:
        """Get total impressions across all variants."""
        return sum(v.metrics.impressions for v in self.variants)

    @property
    def is_complete(self) -> bool:
        """Check if experiment has enough data."""
        return self.total_impressions >= self.target_sample_size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "variants": [v.to_dict() for v in self.variants],
            "control_variant_id": self.control_variant_id,
            "target_sample_size": self.target_sample_size,
            "min_runtime_hours": self.min_runtime_hours,
            "confidence_threshold": self.confidence_threshold,
            "primary_metric": self.primary_metric.value,
            "metadata": self.metadata,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        exp = cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description"),
            tenant_id=data.get("tenant_id", "global"),
            status=ExperimentStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]),
            control_variant_id=data.get("control_variant_id"),
            target_sample_size=data.get("target_sample_size", 1000),
            min_runtime_hours=data.get("min_runtime_hours", 24),
            confidence_threshold=data.get("confidence_threshold", 0.95),
            primary_metric=MetricType(data.get("primary_metric", "success_rate")),
            metadata=data.get("metadata", {}),
        )
        if data.get("started_at"):
            exp.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("ended_at"):
            exp.ended_at = datetime.fromisoformat(data["ended_at"])
        exp.variants = [Variant.from_dict(v) for v in data.get("variants", [])]
        return exp


def calculate_z_score(p1: float, p2: float, n1: int, n2: int) -> float:
    """Calculate z-score for two proportions."""
    if n1 == 0 or n2 == 0:
        return 0.0

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return 0.0

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0

    return (p1 - p2) / se


def z_to_p_value(z: float) -> float:
    """Convert z-score to two-tailed p-value."""
    # Approximation of the normal CDF
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    return 2 * (1 - norm_cdf(abs(z)))


class ABTestingManager:
    """
    Manager for A/B testing experiments.

    Features:
    - Create and manage experiments
    - Traffic splitting with weighted variants
    - Statistical significance calculation
    - Auto-completion based on sample size and confidence
    """

    EXPERIMENT_PREFIX = "orch:experiment:"
    INDEX_PREFIX = "orch:experiments:idx:"

    def __init__(
        self,
        redis_client: Any | None = None,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self._enabled = enabled
        self._local_store: dict[str, Experiment] = {}

    async def create_experiment(
        self,
        name: str,
        variants: list[dict[str, Any]],
        tenant_id: str = "global",
        description: str | None = None,
        control_variant_id: str | None = None,
        target_sample_size: int = 1000,
        primary_metric: MetricType = MetricType.SUCCESS_RATE,
        metadata: dict[str, Any] | None = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            variants: List of variant configurations
            tenant_id: Tenant ID
            description: Experiment description
            control_variant_id: Control variant ID
            target_sample_size: Target sample size
            primary_metric: Primary metric to optimize
            metadata: Additional metadata

        Returns:
            Created Experiment
        """
        now = datetime.now(timezone.utc)

        experiment_id = hashlib.sha256(
            f"{tenant_id}:{name}:{now.timestamp()}".encode()
        ).hexdigest()[:16]

        # Create variants
        exp_variants = []
        for i, v in enumerate(variants):
            variant_id = v.get("variant_id") or f"var_{i}"
            exp_variants.append(
                Variant(
                    variant_id=variant_id,
                    name=v.get("name", f"Variant {i}"),
                    weight=v.get("weight", 1.0),
                    prompt_id=v.get("prompt_id"),
                    prompt_version=v.get("prompt_version"),
                    model=v.get("model"),
                    temperature=v.get("temperature"),
                    max_tokens=v.get("max_tokens"),
                    config=v.get("config", {}),
                )
            )

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            status=ExperimentStatus.DRAFT,
            created_at=now,
            variants=exp_variants,
            control_variant_id=control_variant_id or (
                exp_variants[0].variant_id if exp_variants else None
            ),
            target_sample_size=target_sample_size,
            primary_metric=primary_metric,
            metadata=metadata or {},
        )

        await self._store_experiment(experiment)

        logger.info(
            "Experiment created",
            experiment_id=experiment_id,
            name=name,
            variants=len(variants),
        )

        return experiment

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now(timezone.utc)

        await self._store_experiment(experiment)

        logger.info("Experiment started", experiment_id=experiment_id)
        return experiment

    async def stop_experiment(
        self,
        experiment_id: str,
        analyze: bool = True,
    ) -> Experiment:
        """Stop an experiment."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.now(timezone.utc)

        if analyze:
            experiment.result = await self.analyze_experiment(experiment_id)

        await self._store_experiment(experiment)

        logger.info(
            "Experiment stopped",
            experiment_id=experiment_id,
            winner=experiment.result.winner if experiment.result else None,
        )

        return experiment

    async def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Variant | None:
        """
        Get the assigned variant for a user.

        Uses consistent hashing to ensure same user always gets same variant.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier

        Returns:
            Assigned Variant or None if experiment not running
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        if not experiment.variants:
            return None

        # Consistent hash for user assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)

        # Calculate total weight
        total_weight = sum(v.weight for v in experiment.variants)

        # Weighted random selection using hash
        target = (hash_value % 10000) / 10000 * total_weight
        cumulative = 0.0

        for variant in experiment.variants:
            cumulative += variant.weight
            if target <= cumulative:
                return variant

        return experiment.variants[-1]

    async def record_impression(
        self,
        experiment_id: str,
        variant_id: str,
        success: bool = True,
        latency_ms: float = 0.0,
        tokens: int = 0,
        cost_usd: float = 0.0,
        quality_score: float | None = None,
        custom_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Record an impression (exposure) for a variant.

        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            success: Whether the request succeeded
            latency_ms: Request latency
            tokens: Tokens used
            cost_usd: Cost in USD
            quality_score: Quality score (0-1)
            custom_metrics: Custom metric values
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return

        variant = experiment.get_variant(variant_id)
        if not variant:
            return

        # Update metrics
        variant.metrics.impressions += 1
        if success:
            variant.metrics.conversions += 1
        variant.metrics.total_latency_ms += latency_ms
        variant.metrics.total_tokens += tokens
        variant.metrics.total_cost_usd += cost_usd

        if quality_score is not None:
            variant.metrics.total_quality_score += quality_score
            variant.metrics.quality_samples += 1

        if custom_metrics:
            for key, value in custom_metrics.items():
                variant.metrics.custom_metrics[key] = (
                    variant.metrics.custom_metrics.get(key, 0.0) + value
                )
                variant.metrics.custom_counts[key] = (
                    variant.metrics.custom_counts.get(key, 0) + 1
                )

        await self._store_experiment(experiment)

        # Check if experiment should auto-complete
        if (
            experiment.is_complete
            and experiment.status == ExperimentStatus.RUNNING
        ):
            result = await self.analyze_experiment(experiment_id)
            if result.confidence >= experiment.confidence_threshold:
                await self.stop_experiment(experiment_id, analyze=False)
                experiment.result = result

    async def analyze_experiment(
        self,
        experiment_id: str,
    ) -> ExperimentResult:
        """
        Analyze experiment results and determine winner.

        Uses Welch's t-test for statistical significance.
        """
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        control = experiment.get_control()
        if not control:
            raise ValueError("No control variant")

        best_variant = control
        best_lift = 0.0
        best_z = 0.0
        variant_results = {}

        # Get control rate based on primary metric
        control_rate = self._get_metric_rate(control, experiment.primary_metric)
        control_n = control.metrics.impressions

        for variant in experiment.variants:
            variant_rate = self._get_metric_rate(variant, experiment.primary_metric)
            variant_n = variant.metrics.impressions

            # Calculate z-score
            z = calculate_z_score(variant_rate, control_rate, variant_n, control_n)
            p_value = z_to_p_value(z)

            # Calculate lift
            lift = (
                (variant_rate - control_rate) / control_rate * 100
                if control_rate > 0
                else 0.0
            )

            variant_results[variant.variant_id] = {
                "rate": variant_rate,
                "lift_percent": lift,
                "z_score": z,
                "p_value": p_value,
                "impressions": variant_n,
                "metrics": variant.metrics.to_dict(),
            }

            # Track best variant (higher is better for most metrics)
            if variant.variant_id != control.variant_id:
                if lift > best_lift and z > best_z:
                    best_lift = lift
                    best_z = z
                    best_variant = variant

        # Determine winner
        best_p_value = variant_results.get(
            best_variant.variant_id, {}
        ).get("p_value", 1.0)
        confidence = 1 - best_p_value

        winner = None
        if (
            confidence >= experiment.confidence_threshold
            and best_variant.variant_id != control.variant_id
            and best_lift > 0
        ):
            winner = best_variant.variant_id

        return ExperimentResult(
            experiment_id=experiment_id,
            winner=winner,
            confidence=confidence,
            lift=best_lift,
            p_value=best_p_value,
            sample_size=experiment.total_impressions,
            variant_results=variant_results,
        )

    def _get_metric_rate(self, variant: Variant, metric: MetricType) -> float:
        """Get the rate for a metric."""
        m = variant.metrics
        if metric == MetricType.SUCCESS_RATE:
            return m.conversion_rate
        elif metric == MetricType.LATENCY:
            return 1 / m.avg_latency_ms if m.avg_latency_ms > 0 else 0  # Lower is better
        elif metric == MetricType.COST:
            return 1 / m.avg_cost_usd if m.avg_cost_usd > 0 else 0  # Lower is better
        elif metric == MetricType.QUALITY_SCORE:
            return m.avg_quality_score
        else:
            return m.conversion_rate

    async def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                data = await loop.run_in_executor(
                    None,
                    self._redis.get,
                    f"{self.EXPERIMENT_PREFIX}{experiment_id}",
                )
                if data:
                    return Experiment.from_dict(json.loads(data))
            except Exception as e:
                logger.warning("Failed to get experiment", error=str(e))
        else:
            return self._local_store.get(experiment_id)
        return None

    async def list_experiments(
        self,
        tenant_id: str = "global",
        status: ExperimentStatus | None = None,
        limit: int = 100,
    ) -> list[Experiment]:
        """List experiments for a tenant."""
        experiments = []

        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                pattern = f"{self.EXPERIMENT_PREFIX}*"
                keys = await loop.run_in_executor(
                    None, lambda: list(self._redis.scan_iter(pattern, count=1000))
                )
                for key in keys[:limit]:
                    data = await loop.run_in_executor(None, self._redis.get, key)
                    if data:
                        exp = Experiment.from_dict(json.loads(data))
                        if exp.tenant_id == tenant_id:
                            if status is None or exp.status == status:
                                experiments.append(exp)
            except Exception as e:
                logger.warning("Failed to list experiments", error=str(e))
        else:
            for exp in self._local_store.values():
                if exp.tenant_id == tenant_id:
                    if status is None or exp.status == status:
                        experiments.append(exp)

        return experiments[:limit]

    async def _store_experiment(self, experiment: Experiment) -> None:
        """Store an experiment."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.set(
                        f"{self.EXPERIMENT_PREFIX}{experiment.experiment_id}",
                        json.dumps(experiment.to_dict()),
                    ),
                )
            except Exception as e:
                logger.warning("Failed to store experiment", error=str(e))
        else:
            self._local_store[experiment.experiment_id] = experiment


# Global A/B testing manager
_ab_manager: ABTestingManager | None = None


def get_ab_manager() -> ABTestingManager:
    """Get the global A/B testing manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestingManager()
    return _ab_manager


def configure_ab_manager(
    redis_client: Any | None = None,
    enabled: bool = True,
) -> ABTestingManager:
    """Configure the global A/B testing manager."""
    global _ab_manager
    _ab_manager = ABTestingManager(redis_client=redis_client, enabled=enabled)
    return _ab_manager
