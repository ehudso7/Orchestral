"""
Intelligent model routing and cost optimization.

Provides ML-based routing that optimizes for cost, quality, and latency
based on task characteristics and historical performance data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class OptimizationStrategy(str, Enum):
    """Optimization strategy for routing."""

    COST = "cost"  # Minimize cost
    QUALITY = "quality"  # Maximize quality
    LATENCY = "latency"  # Minimize latency
    BALANCED = "balanced"  # Balance all factors
    CUSTOM = "custom"  # Custom scoring function


class TaskComplexity(str, Enum):
    """Estimated complexity of a task."""

    SIMPLE = "simple"  # Basic queries, short responses
    MODERATE = "moderate"  # Standard tasks
    COMPLEX = "complex"  # Multi-step reasoning, code generation
    EXPERT = "expert"  # Specialized knowledge, complex analysis


@dataclass
class ModelScore:
    """Scoring result for a model."""

    model_id: str
    provider: str
    total_score: float
    cost_score: float
    quality_score: float
    latency_score: float
    availability_score: float

    # Cost estimates
    estimated_cost_usd: float
    estimated_tokens: int
    estimated_latency_ms: float

    # Selection metadata
    selected: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "total_score": self.total_score,
            "cost_score": self.cost_score,
            "quality_score": self.quality_score,
            "latency_score": self.latency_score,
            "availability_score": self.availability_score,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_tokens": self.estimated_tokens,
            "estimated_latency_ms": self.estimated_latency_ms,
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass
class ModelPerformance:
    """Historical performance data for a model."""

    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Quality metrics from evaluations
    avg_quality_score: float = 0.8
    quality_samples: int = 0

    # Task-specific performance
    task_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    # Availability tracking
    last_failure: datetime | None = None
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 1000.0  # Default estimate
        return self.total_latency_ms / self.total_requests

    @property
    def avg_cost_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.01  # Default estimate
        return self.total_cost_usd / self.total_requests

    @property
    def availability_score(self) -> float:
        """Score based on recent availability."""
        if self.consecutive_failures > 5:
            return 0.0
        if self.consecutive_failures > 0:
            return max(0.5, 1.0 - (self.consecutive_failures * 0.1))
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_cost_per_request": self.avg_cost_per_request,
            "avg_quality_score": self.avg_quality_score,
            "availability_score": self.availability_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelPerformance":
        perf = cls(
            model_id=data["model_id"],
            total_requests=data.get("total_requests", 0),
            successful_requests=data.get("successful_requests", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            avg_quality_score=data.get("avg_quality_score", 0.8),
            quality_samples=data.get("quality_samples", 0),
            consecutive_failures=data.get("consecutive_failures", 0),
        )
        if data.get("last_failure"):
            perf.last_failure = datetime.fromisoformat(data["last_failure"])
        return perf


# Model characteristics for cost/quality estimation
MODEL_CHARACTERISTICS = {
    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "base_quality": 0.95,
        "base_latency_ms": 800,
        "complexity_bonus": {"complex": 0.1, "expert": 0.15},
        "strengths": ["reasoning", "coding", "analysis"],
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "base_quality": 0.85,
        "base_latency_ms": 400,
        "complexity_bonus": {},
        "strengths": ["conversation", "simple"],
    },
    # Anthropic models
    "claude-sonnet-4-5-20250929": {
        "provider": "anthropic",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "base_quality": 0.95,
        "base_latency_ms": 900,
        "complexity_bonus": {"complex": 0.1, "expert": 0.1},
        "strengths": ["coding", "reasoning", "creative"],
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.004,
        "base_quality": 0.82,
        "base_latency_ms": 350,
        "complexity_bonus": {},
        "strengths": ["conversation", "simple", "fast"],
    },
    # Google models
    "gemini-2.5-pro": {
        "provider": "google",
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.005,
        "base_quality": 0.92,
        "base_latency_ms": 700,
        "complexity_bonus": {"complex": 0.05},
        "strengths": ["multimodal", "reasoning"],
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
        "base_quality": 0.80,
        "base_latency_ms": 300,
        "complexity_bonus": {},
        "strengths": ["simple", "fast", "cheap"],
    },
}


def estimate_complexity(prompt: str, task_category: str | None = None) -> TaskComplexity:
    """
    Estimate task complexity based on prompt characteristics.

    Uses heuristics based on:
    - Prompt length
    - Question complexity indicators
    - Task category hints
    """
    prompt_lower = prompt.lower()
    length = len(prompt)

    # Expert indicators
    expert_terms = [
        "analyze in depth", "comprehensive", "detailed analysis",
        "expert", "advanced", "complex algorithm", "optimize",
        "architecture", "design pattern", "security audit",
    ]
    if any(term in prompt_lower for term in expert_terms):
        return TaskComplexity.EXPERT

    # Complex indicators
    complex_terms = [
        "explain", "compare", "implement", "write code",
        "step by step", "multiple", "debug", "refactor",
    ]
    if any(term in prompt_lower for term in complex_terms) or length > 1000:
        return TaskComplexity.COMPLEX

    # Moderate indicators
    moderate_terms = [
        "how to", "what is", "describe", "summarize",
        "list", "example", "help me",
    ]
    if any(term in prompt_lower for term in moderate_terms) or length > 200:
        return TaskComplexity.MODERATE

    return TaskComplexity.SIMPLE


def estimate_tokens(prompt: str, expected_response_ratio: float = 2.0) -> tuple[int, int]:
    """
    Estimate input and output tokens.

    Args:
        prompt: Input prompt
        expected_response_ratio: Expected output/input ratio

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    # Rough estimate: ~4 characters per token
    input_tokens = len(prompt) // 4 + 1
    output_tokens = int(input_tokens * expected_response_ratio)

    # Clamp to reasonable bounds
    input_tokens = max(10, min(input_tokens, 100000))
    output_tokens = max(50, min(output_tokens, 4000))

    return input_tokens, output_tokens


class SmartRouter:
    """
    Intelligent model router that optimizes for cost, quality, and latency.

    Features:
    - Task complexity analysis
    - Historical performance tracking
    - Multi-factor scoring
    - Automatic fallback selection
    - A/B testing integration
    """

    PERFORMANCE_PREFIX = "orch:perf:"

    def __init__(
        self,
        redis_client: Any | None = None,
        available_models: list[str] | None = None,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        cost_weight: float = 0.3,
        quality_weight: float = 0.4,
        latency_weight: float = 0.2,
        availability_weight: float = 0.1,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self._available_models = available_models or list(MODEL_CHARACTERISTICS.keys())
        self._strategy = strategy
        self._cost_weight = cost_weight
        self._quality_weight = quality_weight
        self._latency_weight = latency_weight
        self._availability_weight = availability_weight
        self._enabled = enabled

        # Local performance cache
        self._performance_cache: dict[str, ModelPerformance] = {}

    async def select_model(
        self,
        prompt: str,
        task_category: str | None = None,
        strategy: OptimizationStrategy | None = None,
        preferred_provider: str | None = None,
        max_cost_usd: float | None = None,
        max_latency_ms: float | None = None,
        min_quality: float | None = None,
        excluded_models: list[str] | None = None,
    ) -> tuple[str, list[ModelScore]]:
        """
        Select the optimal model for a request.

        Args:
            prompt: The input prompt
            task_category: Category of task (coding, creative, etc.)
            strategy: Override default optimization strategy
            preferred_provider: Prefer a specific provider
            max_cost_usd: Maximum acceptable cost
            max_latency_ms: Maximum acceptable latency
            min_quality: Minimum acceptable quality score
            excluded_models: Models to exclude

        Returns:
            Tuple of (selected_model_id, all_scores)
        """
        strategy = strategy or self._strategy
        excluded = set(excluded_models or [])

        # Analyze task
        complexity = estimate_complexity(prompt, task_category)
        input_tokens, output_tokens = estimate_tokens(prompt)

        # Score all available models
        scores: list[ModelScore] = []

        for model_id in self._available_models:
            if model_id in excluded:
                continue

            if model_id not in MODEL_CHARACTERISTICS:
                continue

            score = await self._score_model(
                model_id=model_id,
                complexity=complexity,
                task_category=task_category,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                strategy=strategy,
                preferred_provider=preferred_provider,
            )

            # Apply constraints
            if max_cost_usd and score.estimated_cost_usd > max_cost_usd:
                score.total_score = 0.0
                score.reason = "exceeds_cost_limit"
            if max_latency_ms and score.estimated_latency_ms > max_latency_ms:
                score.total_score = 0.0
                score.reason = "exceeds_latency_limit"
            if min_quality and score.quality_score < min_quality:
                score.total_score = 0.0
                score.reason = "below_quality_threshold"

            scores.append(score)

        # Sort by total score
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Select best model
        if scores and scores[0].total_score > 0:
            scores[0].selected = True
            scores[0].reason = "highest_score"
            selected_model = scores[0].model_id
        else:
            # Fallback to first available
            selected_model = self._available_models[0]
            if scores:
                scores[0].selected = True
                scores[0].reason = "fallback"

        logger.debug(
            "Model selected",
            model=selected_model,
            complexity=complexity.value,
            strategy=strategy.value,
            top_score=scores[0].total_score if scores else 0,
        )

        return selected_model, scores

    async def _score_model(
        self,
        model_id: str,
        complexity: TaskComplexity,
        task_category: str | None,
        input_tokens: int,
        output_tokens: int,
        strategy: OptimizationStrategy,
        preferred_provider: str | None,
    ) -> ModelScore:
        """Score a model for the given task."""
        chars = MODEL_CHARACTERISTICS.get(model_id, {})
        perf = await self._get_performance(model_id)

        # Calculate cost estimate
        cost_per_1k_in = chars.get("cost_per_1k_input", 0.01)
        cost_per_1k_out = chars.get("cost_per_1k_output", 0.03)
        estimated_cost = (
            (input_tokens / 1000) * cost_per_1k_in +
            (output_tokens / 1000) * cost_per_1k_out
        )

        # Calculate latency estimate
        base_latency = chars.get("base_latency_ms", 1000)
        estimated_latency = base_latency * (1 + output_tokens / 1000)
        if perf.total_requests > 10:
            # Blend with historical data
            estimated_latency = (estimated_latency + perf.avg_latency_ms) / 2

        # Calculate quality score
        base_quality = chars.get("base_quality", 0.8)
        complexity_bonus = chars.get("complexity_bonus", {}).get(complexity.value, 0)
        quality = base_quality + complexity_bonus

        # Task-specific bonus
        strengths = chars.get("strengths", [])
        if task_category and task_category.lower() in strengths:
            quality += 0.05

        # Blend with historical quality
        if perf.quality_samples > 5:
            quality = (quality + perf.avg_quality_score) / 2

        # Provider preference bonus
        provider = chars.get("provider", "unknown")
        provider_bonus = 0.05 if preferred_provider and provider == preferred_provider else 0

        # Normalize scores (0-1 scale, inverted for cost/latency)
        # Lower cost = higher score
        cost_score = 1.0 / (1.0 + estimated_cost * 100)
        # Lower latency = higher score
        latency_score = 1.0 / (1.0 + estimated_latency / 1000)
        # Quality already 0-1
        quality_score = min(1.0, quality)
        # Availability from performance data
        availability_score = perf.availability_score

        # Calculate total score based on strategy
        if strategy == OptimizationStrategy.COST:
            total_score = cost_score * 0.7 + quality_score * 0.2 + availability_score * 0.1
        elif strategy == OptimizationStrategy.QUALITY:
            total_score = quality_score * 0.7 + cost_score * 0.1 + latency_score * 0.1 + availability_score * 0.1
        elif strategy == OptimizationStrategy.LATENCY:
            total_score = latency_score * 0.7 + quality_score * 0.2 + availability_score * 0.1
        else:  # BALANCED
            total_score = (
                cost_score * self._cost_weight +
                quality_score * self._quality_weight +
                latency_score * self._latency_weight +
                availability_score * self._availability_weight
            )

        # Apply provider preference
        total_score += provider_bonus

        return ModelScore(
            model_id=model_id,
            provider=provider,
            total_score=total_score,
            cost_score=cost_score,
            quality_score=quality_score,
            latency_score=latency_score,
            availability_score=availability_score,
            estimated_cost_usd=estimated_cost,
            estimated_tokens=input_tokens + output_tokens,
            estimated_latency_ms=estimated_latency,
        )

    async def record_result(
        self,
        model_id: str,
        success: bool,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        quality_score: float | None = None,
    ) -> None:
        """Record the result of a model call for future optimization."""
        perf = await self._get_performance(model_id)

        perf.total_requests += 1
        if success:
            perf.successful_requests += 1
            perf.consecutive_failures = 0
        else:
            perf.consecutive_failures += 1
            perf.last_failure = datetime.now(timezone.utc)

        perf.total_latency_ms += latency_ms
        perf.total_tokens += input_tokens + output_tokens
        perf.total_cost_usd += cost_usd

        if quality_score is not None:
            n = perf.quality_samples
            perf.avg_quality_score = (perf.avg_quality_score * n + quality_score) / (n + 1)
            perf.quality_samples = n + 1

        await self._store_performance(model_id, perf)

    async def _get_performance(self, model_id: str) -> ModelPerformance:
        """Get performance data for a model."""
        if model_id in self._performance_cache:
            return self._performance_cache[model_id]

        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                data = await loop.run_in_executor(
                    None, self._redis.get, f"{self.PERFORMANCE_PREFIX}{model_id}"
                )
                if data:
                    perf = ModelPerformance.from_dict(json.loads(data))
                    self._performance_cache[model_id] = perf
                    return perf
            except Exception as e:
                logger.warning("Failed to get performance data", error=str(e))

        # Return default performance
        perf = ModelPerformance(model_id=model_id)
        self._performance_cache[model_id] = perf
        return perf

    async def _store_performance(self, model_id: str, perf: ModelPerformance) -> None:
        """Store performance data."""
        self._performance_cache[model_id] = perf

        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.set(
                        f"{self.PERFORMANCE_PREFIX}{model_id}",
                        json.dumps(perf.to_dict()),
                    ),
                )
            except Exception as e:
                logger.warning("Failed to store performance data", error=str(e))

    async def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        stats = {}
        for model_id in self._available_models:
            perf = await self._get_performance(model_id)
            stats[model_id] = perf.to_dict()
        return stats


# Global smart router
_router: SmartRouter | None = None


def get_smart_router() -> SmartRouter:
    """Get the global smart router."""
    global _router
    if _router is None:
        _router = SmartRouter()
    return _router


def configure_smart_router(
    redis_client: Any | None = None,
    available_models: list[str] | None = None,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    enabled: bool = True,
) -> SmartRouter:
    """Configure the global smart router."""
    global _router
    _router = SmartRouter(
        redis_client=redis_client,
        available_models=available_models,
        strategy=strategy,
        enabled=enabled,
    )
    return _router
