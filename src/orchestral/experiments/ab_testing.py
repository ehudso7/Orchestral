"""
A/B testing framework for comparing models and configurations.

Enables data-driven decisions about model selection, prompts, and settings.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(str, Enum):
    """Type of variant being tested."""

    MODEL = "model"
    PROMPT = "prompt"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "system_prompt"
    CUSTOM = "custom"


@dataclass
class Variant:
    """A variant in an A/B test."""

    name: str
    variant_type: VariantType
    config: dict[str, Any]
    weight: float = 1.0  # Relative weight for traffic allocation
    is_control: bool = False

    # Metrics
    impressions: int = 0
    conversions: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    total_quality_score: float = 0.0
    quality_samples: int = 0
    errors: int = 0

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.impressions if self.impressions > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.impressions if self.impressions > 0 else 0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.impressions if self.impressions > 0 else 0

    @property
    def avg_quality_score(self) -> float | None:
        return self.total_quality_score / self.quality_samples if self.quality_samples > 0 else None

    @property
    def error_rate(self) -> float:
        return self.errors / self.impressions if self.impressions > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant_type": self.variant_type.value,
            "config": self.config,
            "weight": self.weight,
            "is_control": self.is_control,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_rate": f"{self.conversion_rate:.2%}",
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_cost": f"${self.avg_cost:.6f}",
            "avg_quality_score": round(self.avg_quality_score, 3) if self.avg_quality_score else None,
            "error_rate": f"{self.error_rate:.2%}",
        }


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""

    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    hypothesis: str = Field(default="", description="What you're testing")

    # Traffic allocation
    traffic_percentage: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Percentage of traffic to include",
    )

    # Statistical settings
    min_samples_per_variant: int = Field(
        default=100,
        description="Minimum samples before results are significant",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Statistical confidence level",
    )

    # Stopping rules
    max_duration_hours: int | None = Field(
        default=None,
        description="Auto-stop after this many hours",
    )
    max_samples: int | None = Field(
        default=None,
        description="Auto-stop after this many total samples",
    )
    early_stopping: bool = Field(
        default=True,
        description="Stop early if results are statistically significant",
    )

    # Targeting
    user_segment: str | None = Field(
        default=None,
        description="Target specific user segment",
    )


@dataclass
class StatisticalResult:
    """Statistical analysis of experiment results."""

    is_significant: bool
    confidence: float
    winner: str | None
    lift: float | None  # Percentage improvement over control
    p_value: float
    samples_needed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_significant": self.is_significant,
            "confidence": f"{self.confidence:.1%}",
            "winner": self.winner,
            "lift": f"{self.lift:+.1%}" if self.lift else None,
            "p_value": round(self.p_value, 4),
            "samples_needed": self.samples_needed,
        }


@dataclass
class ExperimentResult:
    """Results of an experiment."""

    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    variants: list[Variant]
    started_at: datetime
    ended_at: datetime | None
    total_samples: int
    statistical_result: StatisticalResult | None
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_samples": self.total_samples,
            "statistical_result": self.statistical_result.to_dict() if self.statistical_result else None,
            "recommendation": self.recommendation,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    id: str
    config: ExperimentConfig
    variants: list[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def get_variant(self, user_id: str | None = None) -> Variant:
        """
        Get a variant for a user.

        Uses consistent hashing so the same user always gets the same variant.
        """
        if user_id:
            # Consistent assignment based on user ID
            hash_input = f"{self.id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            random.seed(hash_value)

        # Weighted random selection
        total_weight = sum(v.weight for v in self.variants)
        r = random.random() * total_weight

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant

        return self.variants[-1]


class ABTestRunner:
    """
    Manages A/B testing experiments.

    Features:
    - Multiple concurrent experiments
    - Consistent user assignment
    - Statistical significance calculation
    - Early stopping
    - Automatic winner detection

    Example:
        runner = ABTestRunner()

        # Create experiment
        exp = runner.create_experiment(
            config=ExperimentConfig(name="Model Comparison"),
            variants=[
                Variant(name="gpt-4o", variant_type=VariantType.MODEL,
                       config={"model": "gpt-4o"}, is_control=True),
                Variant(name="claude-sonnet", variant_type=VariantType.MODEL,
                       config={"model": "claude-3-5-sonnet-20241022"}),
            ],
        )

        # Run experiment
        runner.start_experiment(exp.id)
        variant = runner.get_variant(exp.id, user_id="user123")

        # Record result
        runner.record_result(exp.id, variant.name, success=True, latency_ms=1500)

        # Get results
        results = runner.get_results(exp.id)
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}
        self._counter = 0

    def create_experiment(
        self,
        config: ExperimentConfig,
        variants: list[Variant],
    ) -> Experiment:
        """Create a new experiment."""
        if len(variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")

        if not any(v.is_control for v in variants):
            # Mark first variant as control if none specified
            variants[0].is_control = True

        self._counter += 1
        exp_id = f"exp_{self._counter}_{config.name.lower().replace(' ', '_')}"

        experiment = Experiment(
            id=exp_id,
            config=config,
            variants=variants,
        )

        self._experiments[exp_id] = experiment
        return experiment

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now(timezone.utc)

    def pause_experiment(self, experiment_id: str) -> None:
        """Pause an experiment."""
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.PAUSED

    def stop_experiment(self, experiment_id: str) -> None:
        """Stop an experiment."""
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.COMPLETED
            exp.ended_at = datetime.now(timezone.utc)

    def get_variant(
        self,
        experiment_id: str,
        user_id: str | None = None,
    ) -> Variant | None:
        """Get the variant assignment for a user."""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None

        # Check traffic allocation
        if random.random() * 100 > exp.config.traffic_percentage:
            # User not in experiment
            return None

        return exp.get_variant(user_id)

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        success: bool,
        latency_ms: float = 0,
        cost: float = 0,
        quality_score: float | None = None,
        error: bool = False,
    ) -> None:
        """Record a result for a variant."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        variant = next((v for v in exp.variants if v.name == variant_name), None)
        if not variant:
            return

        variant.impressions += 1
        if success:
            variant.conversions += 1
        variant.total_latency_ms += latency_ms
        variant.total_cost += cost
        if quality_score is not None:
            variant.total_quality_score += quality_score
            variant.quality_samples += 1
        if error:
            variant.errors += 1

        # Check stopping rules
        self._check_stopping_rules(exp)

    def _check_stopping_rules(self, exp: Experiment) -> None:
        """Check if experiment should be stopped."""
        if exp.status != ExperimentStatus.RUNNING:
            return

        config = exp.config

        # Check max samples
        total_samples = sum(v.impressions for v in exp.variants)
        if config.max_samples and total_samples >= config.max_samples:
            self.stop_experiment(exp.id)
            return

        # Check duration
        if config.max_duration_hours and exp.started_at:
            hours_running = (datetime.now(timezone.utc) - exp.started_at).total_seconds() / 3600
            if hours_running >= config.max_duration_hours:
                self.stop_experiment(exp.id)
                return

        # Check early stopping
        if config.early_stopping:
            result = self._compute_statistical_result(exp)
            if result and result.is_significant:
                min_samples = config.min_samples_per_variant
                if all(v.impressions >= min_samples for v in exp.variants):
                    self.stop_experiment(exp.id)

    def _compute_statistical_result(self, exp: Experiment) -> StatisticalResult | None:
        """Compute statistical significance of results."""
        control = next((v for v in exp.variants if v.is_control), None)
        if not control or control.impressions < 10:
            return None

        best_treatment = None
        best_lift = -float("inf")

        for variant in exp.variants:
            if variant.is_control or variant.impressions < 10:
                continue

            # Use quality score if available, otherwise conversion rate
            if variant.avg_quality_score and control.avg_quality_score:
                control_metric = control.avg_quality_score
                variant_metric = variant.avg_quality_score
            else:
                control_metric = control.conversion_rate
                variant_metric = variant.conversion_rate

            if control_metric > 0:
                lift = (variant_metric - control_metric) / control_metric
                if lift > best_lift:
                    best_lift = lift
                    best_treatment = variant

        if not best_treatment:
            return StatisticalResult(
                is_significant=False,
                confidence=0.0,
                winner=None,
                lift=None,
                p_value=1.0,
                samples_needed=exp.config.min_samples_per_variant,
            )

        # Simplified statistical test (two-proportion z-test approximation)
        n1 = control.impressions
        n2 = best_treatment.impressions
        p1 = control.conversion_rate
        p2 = best_treatment.conversion_rate

        if p1 == p2:
            p_value = 1.0
        else:
            # Pooled proportion
            p_pool = (control.conversions + best_treatment.conversions) / (n1 + n2)
            if p_pool == 0 or p_pool == 1:
                p_value = 1.0
            else:
                se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                if se == 0:
                    p_value = 1.0
                else:
                    z = abs(p2 - p1) / se
                    # Approximate p-value from z-score
                    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

        is_significant = p_value < (1 - exp.config.confidence_level)

        winner = None
        if is_significant:
            if best_lift > 0:
                winner = best_treatment.name
            else:
                winner = control.name

        return StatisticalResult(
            is_significant=is_significant,
            confidence=1 - p_value,
            winner=winner,
            lift=best_lift if best_lift != -float("inf") else None,
            p_value=p_value,
            samples_needed=None if is_significant else exp.config.min_samples_per_variant,
        )

    def get_results(self, experiment_id: str) -> ExperimentResult | None:
        """Get experiment results."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        statistical_result = self._compute_statistical_result(exp)

        # Generate recommendation
        if statistical_result and statistical_result.is_significant:
            if statistical_result.winner:
                recommendation = f"Deploy {statistical_result.winner} (lift: {statistical_result.lift:+.1%})"
            else:
                recommendation = "No clear winner - continue testing or keep control"
        else:
            total_samples = sum(v.impressions for v in exp.variants)
            needed = exp.config.min_samples_per_variant * len(exp.variants)
            if total_samples < needed:
                recommendation = f"Need more data ({total_samples}/{needed} samples)"
            else:
                recommendation = "Results not yet statistically significant"

        return ExperimentResult(
            experiment_id=exp.id,
            experiment_name=exp.config.name,
            status=exp.status,
            variants=exp.variants,
            started_at=exp.started_at or exp.created_at,
            ended_at=exp.ended_at,
            total_samples=sum(v.impressions for v in exp.variants),
            statistical_result=statistical_result,
            recommendation=recommendation,
        )

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments."""
        return [
            {
                "id": exp.id,
                "name": exp.config.name,
                "status": exp.status.value,
                "variants": len(exp.variants),
                "total_samples": sum(v.impressions for v in exp.variants),
                "created_at": exp.created_at.isoformat(),
            }
            for exp in self._experiments.values()
        ]
