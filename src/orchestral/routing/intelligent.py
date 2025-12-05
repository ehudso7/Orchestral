"""
Intelligent routing with ML-based model selection.

Learns from performance data to make optimal routing decisions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum

from pydantic import BaseModel, Field

from orchestral.routing.classifier import TaskClassifier, TaskType, ClassificationResult


class OptimizationGoal(str, Enum):
    """Optimization goals for routing."""

    QUALITY = "quality"  # Maximize response quality
    SPEED = "speed"  # Minimize latency
    COST = "cost"  # Minimize cost
    BALANCED = "balanced"  # Balance all factors


class RouterConfig(BaseModel):
    """Configuration for intelligent router."""

    optimization_goal: OptimizationGoal = Field(
        default=OptimizationGoal.BALANCED,
        description="Primary optimization goal",
    )
    quality_weight: float = Field(default=0.4, ge=0, le=1)
    speed_weight: float = Field(default=0.3, ge=0, le=1)
    cost_weight: float = Field(default=0.3, ge=0, le=1)

    # Constraints
    max_latency_ms: int | None = Field(default=None, description="Max acceptable latency")
    max_cost_per_request: float | None = Field(default=None, description="Max cost per request")
    min_quality_score: float | None = Field(default=None, description="Minimum quality score")

    # Learning settings
    learning_rate: float = Field(default=0.1, description="How fast to adapt to new data")
    exploration_rate: float = Field(default=0.1, description="Probability of trying non-optimal model")
    min_samples_for_learning: int = Field(default=10, description="Min samples before using learned data")


@dataclass
class ModelScore:
    """Computed score for a model."""

    model: str
    total_score: float
    quality_score: float
    speed_score: float
    cost_score: float
    task_fit_score: float
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_score": round(self.total_score, 3),
            "quality_score": round(self.quality_score, 3),
            "speed_score": round(self.speed_score, 3),
            "cost_score": round(self.cost_score, 3),
            "task_fit_score": round(self.task_fit_score, 3),
            "confidence": f"{self.confidence:.0%}",
            "reasoning": self.reasoning,
        }


@dataclass
class RoutingDecision:
    """A routing decision with explanation."""

    selected_model: str
    scores: list[ModelScore]
    task_classification: ClassificationResult
    decision_reasoning: str
    was_exploration: bool
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "scores": [s.to_dict() for s in self.scores],
            "task_classification": self.task_classification.to_dict(),
            "decision_reasoning": self.decision_reasoning,
            "was_exploration": self.was_exploration,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ModelPerformanceData:
    """Learned performance data for a model."""

    model: str
    task_type: TaskType

    # Aggregated metrics
    total_requests: int = 0
    successful_requests: int = 0
    total_latency_ms: float = 0
    total_cost: float = 0
    total_quality_score: float = 0
    quality_samples: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.total_requests if self.total_requests > 0 else 0

    @property
    def avg_quality(self) -> float | None:
        return self.total_quality_score / self.quality_samples if self.quality_samples > 0 else None

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0


class IntelligentRouter:
    """
    ML-based intelligent router that learns from performance data.

    Features:
    - Automatic task classification
    - Performance-based model scoring
    - Multi-objective optimization (quality, speed, cost)
    - Exploration/exploitation balance
    - Constraint satisfaction
    - Explainable decisions

    Example:
        router = IntelligentRouter(
            available_models=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
            config=RouterConfig(optimization_goal=OptimizationGoal.BALANCED),
        )

        # Get routing decision
        decision = router.route("Write a Python function to sort a list")
        print(f"Selected: {decision.selected_model}")
        print(f"Reason: {decision.decision_reasoning}")

        # Record outcome for learning
        router.record_outcome(
            model="gpt-4o",
            task_type=TaskType.CODING,
            latency_ms=1500,
            cost=0.01,
            quality_score=0.9,
            success=True,
        )
    """

    # Default model characteristics (used before learning)
    MODEL_PRIORS = {
        # OpenAI
        "gpt-4o": {
            "quality": 0.9,
            "speed": 0.7,
            "cost": 0.5,
            "strengths": [TaskType.CODING, TaskType.REASONING, TaskType.ANALYSIS],
        },
        "gpt-4o-mini": {
            "quality": 0.75,
            "speed": 0.9,
            "cost": 0.9,
            "strengths": [TaskType.CONVERSATION, TaskType.SUMMARIZATION, TaskType.EXTRACTION],
        },
        "gpt-5.1": {
            "quality": 0.95,
            "speed": 0.6,
            "cost": 0.4,
            "strengths": [TaskType.REASONING, TaskType.MATH, TaskType.CODING],
        },
        "o1": {
            "quality": 0.98,
            "speed": 0.3,
            "cost": 0.2,
            "strengths": [TaskType.REASONING, TaskType.MATH, TaskType.ANALYSIS],
        },
        # Anthropic
        "claude-3-5-sonnet-20241022": {
            "quality": 0.9,
            "speed": 0.75,
            "cost": 0.6,
            "strengths": [TaskType.CODING, TaskType.CREATIVE, TaskType.ANALYSIS],
        },
        "claude-3-opus-20240229": {
            "quality": 0.95,
            "speed": 0.5,
            "cost": 0.3,
            "strengths": [TaskType.REASONING, TaskType.CREATIVE, TaskType.RESEARCH],
        },
        "claude-3-haiku-20240307": {
            "quality": 0.7,
            "speed": 0.95,
            "cost": 0.95,
            "strengths": [TaskType.CONVERSATION, TaskType.EXTRACTION, TaskType.CLASSIFICATION],
        },
        "claude-opus-4-5-20251101": {
            "quality": 0.97,
            "speed": 0.5,
            "cost": 0.25,
            "strengths": [TaskType.REASONING, TaskType.CODING, TaskType.ANALYSIS],
        },
        "claude-sonnet-4-5-20250929": {
            "quality": 0.92,
            "speed": 0.7,
            "cost": 0.55,
            "strengths": [TaskType.CODING, TaskType.CREATIVE, TaskType.ANALYSIS],
        },
        # Google
        "gemini-3-pro-preview": {
            "quality": 0.9,
            "speed": 0.7,
            "cost": 0.6,
            "strengths": [TaskType.REASONING, TaskType.ANALYSIS, TaskType.RESEARCH],
        },
        "gemini-2.5-pro": {
            "quality": 0.85,
            "speed": 0.8,
            "cost": 0.75,
            "strengths": [TaskType.ANALYSIS, TaskType.SUMMARIZATION, TaskType.CONVERSATION],
        },
        "gemini-2.5-flash": {
            "quality": 0.7,
            "speed": 0.95,
            "cost": 0.95,
            "strengths": [TaskType.CONVERSATION, TaskType.EXTRACTION, TaskType.CLASSIFICATION],
        },
    }

    def __init__(
        self,
        available_models: list[str],
        config: RouterConfig | None = None,
    ):
        self.available_models = available_models
        self.config = config or RouterConfig()
        self.classifier = TaskClassifier()

        # Learned performance data: (model, task_type) -> data
        self._performance_data: dict[tuple[str, TaskType], ModelPerformanceData] = {}

        # Decision history for analysis
        self._decision_history: list[RoutingDecision] = []

    def _get_prior_scores(self, model: str) -> dict[str, float]:
        """Get prior scores for a model."""
        if model in self.MODEL_PRIORS:
            return self.MODEL_PRIORS[model]
        # Default for unknown models
        return {"quality": 0.7, "speed": 0.7, "cost": 0.7, "strengths": []}

    def _get_learned_scores(
        self, model: str, task_type: TaskType
    ) -> dict[str, float] | None:
        """Get learned scores from performance data."""
        key = (model, task_type)
        data = self._performance_data.get(key)

        if not data or data.total_requests < self.config.min_samples_for_learning:
            return None

        # Normalize metrics to 0-1 scale
        # Lower latency is better (invert)
        # Assuming max latency of 30s = 30000ms
        speed_score = max(0, 1 - (data.avg_latency_ms / 30000))

        # Lower cost is better (invert)
        # Assuming max cost of $0.10 per request
        cost_score = max(0, 1 - (data.avg_cost / 0.10))

        # Quality is direct
        quality_score = data.avg_quality if data.avg_quality else 0.7

        # Success rate bonus
        reliability_bonus = data.success_rate * 0.1

        return {
            "quality": min(1.0, quality_score + reliability_bonus),
            "speed": speed_score,
            "cost": cost_score,
        }

    def _compute_task_fit(self, model: str, task_type: TaskType) -> float:
        """Compute how well a model fits a task type."""
        priors = self._get_prior_scores(model)
        strengths = priors.get("strengths", [])

        if task_type in strengths:
            return 1.0
        elif TaskType.GENERAL in strengths:
            return 0.7
        else:
            return 0.5

    def _score_model(
        self,
        model: str,
        task_classification: ClassificationResult,
    ) -> ModelScore:
        """Compute comprehensive score for a model."""
        task_type = task_classification.primary_type

        # Get scores (learned if available, otherwise priors)
        learned = self._get_learned_scores(model, task_type)
        priors = self._get_prior_scores(model)

        if learned:
            # Blend learned and prior with learning rate
            lr = self.config.learning_rate
            quality = learned["quality"] * lr + priors["quality"] * (1 - lr)
            speed = learned["speed"] * lr + priors["speed"] * (1 - lr)
            cost = learned["cost"] * lr + priors["cost"] * (1 - lr)
            confidence = min(0.95, 0.5 + self._performance_data[(model, task_type)].total_requests / 100)
            source = "learned"
        else:
            quality = priors["quality"]
            speed = priors["speed"]
            cost = priors["cost"]
            confidence = 0.5
            source = "prior"

        # Task fit
        task_fit = self._compute_task_fit(model, task_type)

        # Apply optimization goal weights
        if self.config.optimization_goal == OptimizationGoal.QUALITY:
            weights = (0.6, 0.2, 0.2)
        elif self.config.optimization_goal == OptimizationGoal.SPEED:
            weights = (0.2, 0.6, 0.2)
        elif self.config.optimization_goal == OptimizationGoal.COST:
            weights = (0.2, 0.2, 0.6)
        else:  # BALANCED
            weights = (
                self.config.quality_weight,
                self.config.speed_weight,
                self.config.cost_weight,
            )

        # Compute weighted score
        base_score = (
            quality * weights[0] +
            speed * weights[1] +
            cost * weights[2]
        )

        # Apply task fit multiplier
        total_score = base_score * (0.7 + task_fit * 0.3)

        # Build reasoning
        reasoning_parts = [f"Based on {source} data"]
        if task_fit >= 0.9:
            reasoning_parts.append(f"strong fit for {task_type.value}")
        if quality >= 0.9:
            reasoning_parts.append("high quality")
        if speed >= 0.9:
            reasoning_parts.append("fast")
        if cost >= 0.9:
            reasoning_parts.append("cost-effective")

        return ModelScore(
            model=model,
            total_score=total_score,
            quality_score=quality,
            speed_score=speed,
            cost_score=cost,
            task_fit_score=task_fit,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts),
        )

    def _apply_constraints(self, scores: list[ModelScore]) -> list[ModelScore]:
        """Filter scores based on constraints."""
        filtered = []

        for score in scores:
            # Check quality constraint
            if self.config.min_quality_score:
                if score.quality_score < self.config.min_quality_score:
                    continue

            # Note: latency and cost constraints would need actual values
            # which we don't have at routing time. These are handled
            # via the learned performance data influencing scores.

            filtered.append(score)

        return filtered if filtered else scores  # Don't filter all out

    def route(self, prompt: str) -> RoutingDecision:
        """
        Route a prompt to the optimal model.

        Args:
            prompt: The user's prompt

        Returns:
            RoutingDecision with selected model and explanation
        """
        # Classify the task
        classification = self.classifier.classify(prompt)

        # Score all available models
        scores = [
            self._score_model(model, classification)
            for model in self.available_models
        ]

        # Apply constraints
        scores = self._apply_constraints(scores)

        # Sort by score
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Exploration vs exploitation
        import random
        if random.random() < self.config.exploration_rate and len(scores) > 1:
            # Explore: pick randomly from top 3
            selected_idx = random.randint(0, min(2, len(scores) - 1))
            was_exploration = True
            reasoning = f"Exploration: trying {scores[selected_idx].model} instead of optimal"
        else:
            # Exploit: pick the best
            selected_idx = 0
            was_exploration = False
            reasoning = f"Selected {scores[0].model} ({scores[0].reasoning})"

        selected_model = scores[selected_idx].model

        decision = RoutingDecision(
            selected_model=selected_model,
            scores=scores,
            task_classification=classification,
            decision_reasoning=reasoning,
            was_exploration=was_exploration,
        )

        # Record for analysis
        self._decision_history.append(decision)

        return decision

    def record_outcome(
        self,
        model: str,
        task_type: TaskType,
        latency_ms: float,
        cost: float,
        quality_score: float | None = None,
        success: bool = True,
    ) -> None:
        """
        Record the outcome of a request for learning.

        Call this after each request to help the router learn.
        """
        key = (model, task_type)

        if key not in self._performance_data:
            self._performance_data[key] = ModelPerformanceData(
                model=model,
                task_type=task_type,
            )

        data = self._performance_data[key]
        data.total_requests += 1
        data.total_latency_ms += latency_ms
        data.total_cost += cost

        if success:
            data.successful_requests += 1

        if quality_score is not None:
            data.total_quality_score += quality_score
            data.quality_samples += 1

    def get_model_rankings(self, task_type: TaskType | None = None) -> list[dict[str, Any]]:
        """Get current model rankings, optionally for a specific task."""
        if task_type:
            classification = ClassificationResult(
                primary_type=task_type,
                confidence=1.0,
                secondary_types=[],
                detected_features=[],
                complexity="moderate",
                estimated_tokens=100,
            )
        else:
            # Use general task
            classification = ClassificationResult(
                primary_type=TaskType.GENERAL,
                confidence=1.0,
                secondary_types=[],
                detected_features=[],
                complexity="moderate",
                estimated_tokens=100,
            )

        scores = [
            self._score_model(model, classification)
            for model in self.available_models
        ]
        scores.sort(key=lambda s: s.total_score, reverse=True)

        return [s.to_dict() for s in scores]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of learned performance data."""
        summary = {}

        for (model, task_type), data in self._performance_data.items():
            if model not in summary:
                summary[model] = {}

            summary[model][task_type.value] = {
                "requests": data.total_requests,
                "success_rate": f"{data.success_rate:.1%}",
                "avg_latency_ms": round(data.avg_latency_ms, 2),
                "avg_cost": f"${data.avg_cost:.6f}",
                "avg_quality": round(data.avg_quality, 3) if data.avg_quality else None,
            }

        return summary

    def export_model(self) -> dict[str, Any]:
        """Export learned data for persistence."""
        return {
            "performance_data": {
                f"{model}|{task_type.value}": {
                    "total_requests": data.total_requests,
                    "successful_requests": data.successful_requests,
                    "total_latency_ms": data.total_latency_ms,
                    "total_cost": data.total_cost,
                    "total_quality_score": data.total_quality_score,
                    "quality_samples": data.quality_samples,
                }
                for (model, task_type), data in self._performance_data.items()
            },
            "config": self.config.model_dump(),
        }

    def import_model(self, data: dict[str, Any]) -> None:
        """Import learned data from export."""
        for key_str, perf_data in data.get("performance_data", {}).items():
            model, task_type_str = key_str.split("|")
            task_type = TaskType(task_type_str)

            self._performance_data[(model, task_type)] = ModelPerformanceData(
                model=model,
                task_type=task_type,
                total_requests=perf_data["total_requests"],
                successful_requests=perf_data["successful_requests"],
                total_latency_ms=perf_data["total_latency_ms"],
                total_cost=perf_data["total_cost"],
                total_quality_score=perf_data["total_quality_score"],
                quality_samples=perf_data["quality_samples"],
            )
