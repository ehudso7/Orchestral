"""
Quality scoring system for LLM responses.

Provides comprehensive evaluation across multiple dimensions
without requiring additional LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from orchestral.evaluation.metrics import (
    ResponseMetrics,
    compute_response_metrics,
)


class QualityDimension(str, Enum):
    """Dimensions of response quality."""

    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    SAFETY = "safety"
    READABILITY = "readability"
    SPECIFICITY = "specificity"


class EvaluationConfig(BaseModel):
    """Configuration for quality evaluation."""

    # Dimension weights (must sum to 1.0)
    weights: dict[str, float] = Field(
        default={
            "relevance": 0.25,
            "coherence": 0.15,
            "completeness": 0.15,
            "accuracy": 0.15,
            "safety": 0.10,
            "readability": 0.10,
            "specificity": 0.10,
        }
    )

    # Thresholds
    min_acceptable_score: float = Field(default=0.6)
    excellent_score_threshold: float = Field(default=0.85)

    # Response expectations
    min_word_count: int = Field(default=10)
    max_word_count: int = Field(default=5000)
    ideal_reading_level: tuple[float, float] = Field(default=(30.0, 70.0))


@dataclass
class QualityScore:
    """Quality score for a single dimension."""

    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    weight: float
    weighted_score: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 3),
            "weight": round(self.weight, 3),
            "weighted_score": round(self.weighted_score, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a response."""

    overall_score: float
    grade: str  # A, B, C, D, F
    dimension_scores: list[QualityScore]
    metrics: ResponseMetrics
    recommendations: list[str]
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_acceptable(self) -> bool:
        return self.overall_score >= 0.6

    @property
    def is_excellent(self) -> bool:
        return self.overall_score >= 0.85

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "grade": self.grade,
            "is_acceptable": self.is_acceptable,
            "is_excellent": self.is_excellent,
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "metrics": self.metrics.to_dict(),
            "recommendations": self.recommendations,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class QualityScorer:
    """
    Evaluates LLM response quality across multiple dimensions.

    Provides fast, heuristic-based scoring that doesn't require
    additional API calls, making it suitable for real-time evaluation.

    Example:
        scorer = QualityScorer()
        result = scorer.evaluate(
            query="Explain quantum computing",
            response="Quantum computing uses qubits...",
        )
        print(f"Score: {result.overall_score}, Grade: {result.grade}")
    """

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _evaluate_relevance(
        self, query: str, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """Evaluate response relevance to query."""
        # Simple keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for", "on", "what", "how", "why", "when", "where", "which", "who"}
        query_keywords = query_words - stop_words
        response_keywords = response_words - stop_words

        if query_keywords:
            overlap = len(query_keywords & response_keywords) / len(query_keywords)
        else:
            overlap = 0.5

        # Length appropriateness
        if metrics.word_count < self.config.min_word_count:
            length_score = 0.3
            length_reason = "too short"
        elif metrics.word_count > self.config.max_word_count:
            length_score = 0.7
            length_reason = "too long"
        else:
            length_score = 1.0
            length_reason = "appropriate"

        score = overlap * 0.6 + length_score * 0.4
        weight = self.config.weights.get("relevance", 0.25)

        reasoning = f"Keyword overlap: {overlap:.0%}, Length: {length_reason}"

        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reasoning,
        )

    def _evaluate_coherence(
        self, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """Evaluate response coherence and flow."""
        score = metrics.coherence_score

        # Penalize high repetition
        if metrics.repetition_score > 0.3:
            score *= 0.8
            reason = "High repetition detected"
        elif metrics.repetition_score > 0.15:
            score *= 0.9
            reason = "Some repetition detected"
        else:
            reason = "Good flow and minimal repetition"

        weight = self.config.weights.get("coherence", 0.15)

        return QualityScore(
            dimension=QualityDimension.COHERENCE,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reason,
        )

    def _evaluate_completeness(
        self, query: str, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """Evaluate response completeness."""
        # Check for question markers in query
        is_question = any(
            q in query.lower() for q in ["?", "what", "how", "why", "when", "where", "which", "who", "explain", "describe"]
        )

        score = 0.5  # Base score

        # Word count contribution
        if metrics.word_count >= 50:
            score += 0.2
        if metrics.word_count >= 100:
            score += 0.1

        # Structure indicates completeness
        if metrics.has_formatting:
            score += 0.1
        if metrics.list_item_count > 0:
            score += 0.05
        if metrics.header_count > 0:
            score += 0.05

        score = min(1.0, score)
        weight = self.config.weights.get("completeness", 0.15)

        if score >= 0.8:
            reason = "Response appears comprehensive"
        elif score >= 0.6:
            reason = "Response covers main points"
        else:
            reason = "Response may be incomplete"

        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reason,
        )

    def _evaluate_accuracy(
        self, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """
        Evaluate accuracy signals.

        Note: True accuracy requires ground truth, so this is a proxy
        based on linguistic signals that correlate with accuracy.
        """
        # Start with specificity as a proxy
        score = metrics.specificity_score * 0.6

        # Uncertainty signals reduce confidence in accuracy
        uncertainty_penalty = min(0.3, metrics.uncertainty_phrases * 0.05)
        score -= uncertainty_penalty

        # Hallucination signals are concerning
        if metrics.potential_hallucination_signals > 0:
            score *= 0.7
            reason = f"Contains {metrics.potential_hallucination_signals} potential hallucination signal(s)"
        elif metrics.uncertainty_phrases > 2:
            reason = "High uncertainty in response"
        else:
            score += 0.3  # Bonus for confident, specific response
            reason = "Response appears confident and specific"

        score = max(0.0, min(1.0, score))
        weight = self.config.weights.get("accuracy", 0.15)

        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reason,
        )

    def _evaluate_safety(
        self, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """Evaluate response safety."""
        score = 1.0

        # Check for concerning patterns
        concerning_patterns = [
            "i cannot help with",
            "i won't provide",
            "illegal",
            "harmful",
            "dangerous",
            "unethical",
        ]

        response_lower = response.lower()
        concerns = [p for p in concerning_patterns if p in response_lower]

        if concerns:
            score = 0.9  # Slight reduction - these might be appropriate refusals
            reason = "Contains safety-related language (may be appropriate)"
        else:
            reason = "No safety concerns detected"

        weight = self.config.weights.get("safety", 0.10)

        return QualityScore(
            dimension=QualityDimension.SAFETY,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reason,
        )

    def _evaluate_readability(
        self, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """Evaluate response readability."""
        ideal_min, ideal_max = self.config.ideal_reading_level
        fre = metrics.flesch_reading_ease

        if ideal_min <= fre <= ideal_max:
            score = 1.0
            reason = f"Reading level is ideal (FRE: {fre:.0f})"
        elif fre < ideal_min:
            # Too difficult
            distance = (ideal_min - fre) / ideal_min
            score = max(0.5, 1 - distance)
            reason = f"Text may be too complex (FRE: {fre:.0f})"
        else:
            # Too simple
            distance = (fre - ideal_max) / (100 - ideal_max)
            score = max(0.6, 1 - distance * 0.5)
            reason = f"Text may be too simple (FRE: {fre:.0f})"

        weight = self.config.weights.get("readability", 0.10)

        return QualityScore(
            dimension=QualityDimension.READABILITY,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reason,
        )

    def _evaluate_specificity(
        self, response: str, metrics: ResponseMetrics
    ) -> QualityScore:
        """Evaluate response specificity."""
        score = metrics.specificity_score

        # Bonus for code blocks in technical responses
        if metrics.code_block_count > 0:
            score = min(1.0, score + 0.1)

        weight = self.config.weights.get("specificity", 0.10)

        if score >= 0.8:
            reason = "Response is specific and detailed"
        elif score >= 0.5:
            reason = "Response has moderate specificity"
        else:
            reason = "Response may be too vague or generic"

        return QualityScore(
            dimension=QualityDimension.SPECIFICITY,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            reasoning=reason,
        )

    def _generate_recommendations(
        self, scores: list[QualityScore], metrics: ResponseMetrics
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        score_dict = {s.dimension: s.score for s in scores}

        if score_dict.get(QualityDimension.RELEVANCE, 1) < 0.7:
            recommendations.append("Improve relevance by addressing the query more directly")

        if score_dict.get(QualityDimension.COHERENCE, 1) < 0.7:
            recommendations.append("Improve coherence with better transitions and structure")

        if score_dict.get(QualityDimension.COMPLETENESS, 1) < 0.7:
            recommendations.append("Provide more comprehensive coverage of the topic")

        if score_dict.get(QualityDimension.SPECIFICITY, 1) < 0.6:
            recommendations.append("Add specific examples, numbers, or details")

        if metrics.repetition_score > 0.2:
            recommendations.append("Reduce repetitive content")

        if metrics.potential_hallucination_signals > 0:
            recommendations.append("Verify factual claims and reduce uncertainty")

        if not recommendations:
            recommendations.append("Response meets quality standards")

        return recommendations

    def evaluate(self, query: str, response: str) -> EvaluationResult:
        """
        Evaluate a response's quality.

        Args:
            query: The original query/prompt
            response: The LLM's response

        Returns:
            EvaluationResult with scores and recommendations
        """
        # Compute metrics
        metrics = compute_response_metrics(response)

        # Evaluate each dimension
        dimension_scores = [
            self._evaluate_relevance(query, response, metrics),
            self._evaluate_coherence(response, metrics),
            self._evaluate_completeness(query, response, metrics),
            self._evaluate_accuracy(response, metrics),
            self._evaluate_safety(response, metrics),
            self._evaluate_readability(response, metrics),
            self._evaluate_specificity(response, metrics),
        ]

        # Compute overall score
        overall_score = sum(s.weighted_score for s in dimension_scores)
        grade = self._score_to_grade(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, metrics)

        return EvaluationResult(
            overall_score=overall_score,
            grade=grade,
            dimension_scores=dimension_scores,
            metrics=metrics,
            recommendations=recommendations,
        )

    def compare(
        self, query: str, responses: dict[str, str]
    ) -> dict[str, EvaluationResult]:
        """
        Compare multiple responses for the same query.

        Args:
            query: The original query
            responses: Dict mapping model/source name to response text

        Returns:
            Dict mapping model/source name to EvaluationResult
        """
        return {name: self.evaluate(query, response) for name, response in responses.items()}

    def rank(
        self, query: str, responses: dict[str, str]
    ) -> list[tuple[str, EvaluationResult]]:
        """
        Rank responses by quality score.

        Returns:
            List of (name, result) tuples sorted by score descending
        """
        results = self.compare(query, responses)
        return sorted(results.items(), key=lambda x: x[1].overall_score, reverse=True)
