"""
Quality evaluation system for LLM outputs.

Provides automated evaluation metrics and scoring for:
- Relevance to prompt
- Coherence and fluency
- Factuality (basic)
- Toxicity detection
- Custom metrics
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class EvaluationMetric(str, Enum):
    """Types of evaluation metrics."""

    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    FACTUALITY = "factuality"
    TOXICITY = "toxicity"
    HELPFULNESS = "helpfulness"
    CONCISENESS = "conciseness"
    CUSTOM = "custom"


@dataclass
class EvaluationResult:
    """Result of an evaluation."""

    metric: EvaluationMetric
    score: float  # 0-1 normalized score
    passed: bool
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    evaluator_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
            "evaluator_name": self.evaluator_name,
            "timestamp": self.timestamp.isoformat(),
        }


class Evaluator(ABC):
    """Base class for evaluators."""

    def __init__(
        self,
        name: str,
        metric: EvaluationMetric,
        threshold: float = 0.7,
        enabled: bool = True,
    ):
        self.name = name
        self.metric = metric
        self.threshold = threshold
        self.enabled = enabled

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a response.

        Args:
            prompt: The input prompt
            response: The model response
            context: Additional context

        Returns:
            EvaluationResult
        """
        pass


class RelevanceEvaluator(Evaluator):
    """
    Evaluate relevance of response to prompt.

    Uses keyword overlap and semantic similarity heuristics.
    """

    def __init__(
        self,
        name: str = "relevance",
        threshold: float = 0.6,
        enabled: bool = True,
    ):
        super().__init__(name, EvaluationMetric.RELEVANCE, threshold, enabled)

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate relevance."""
        if not self.enabled:
            return EvaluationResult(
                metric=self.metric,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                evaluator_name=self.name,
            )

        # Extract key terms from prompt
        prompt_words = set(self._extract_keywords(prompt))
        response_words = set(self._extract_keywords(response))

        if not prompt_words:
            return EvaluationResult(
                metric=self.metric,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                evaluator_name=self.name,
                details={"reason": "no_keywords_in_prompt"},
            )

        # Calculate keyword overlap
        overlap = len(prompt_words & response_words)
        overlap_score = overlap / len(prompt_words)

        # Check for question answering
        is_question = "?" in prompt
        has_answer_indicators = any(
            phrase in response.lower()
            for phrase in ["is", "are", "the", "yes", "no", "because", "therefore"]
        )
        answer_bonus = 0.1 if is_question and has_answer_indicators else 0

        # Length ratio check (response shouldn't be too short)
        length_ratio = len(response) / max(len(prompt), 1)
        length_penalty = 0 if length_ratio > 0.3 else -0.2

        score = min(1.0, max(0.0, overlap_score + answer_bonus + length_penalty))

        return EvaluationResult(
            metric=self.metric,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            evaluator_name=self.name,
            details={
                "overlap_score": overlap_score,
                "prompt_keywords": list(prompt_words)[:10],
                "matching_keywords": list(prompt_words & response_words)[:10],
            },
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Remove punctuation and lowercase
        text = re.sub(r"[^\w\s]", " ", text.lower())
        words = text.split()

        # Filter stop words and short words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "this", "that", "these", "those", "i",
            "me", "my", "we", "our", "you", "your", "he", "him", "his",
            "she", "her", "it", "its", "they", "them", "their", "what",
            "which", "who", "whom", "this", "that", "am", "it's",
        }

        return [w for w in words if w not in stop_words and len(w) > 2]


class CoherenceEvaluator(Evaluator):
    """
    Evaluate coherence and fluency of response.

    Checks for:
    - Sentence structure
    - Logical flow
    - Grammar indicators
    """

    def __init__(
        self,
        name: str = "coherence",
        threshold: float = 0.7,
        enabled: bool = True,
    ):
        super().__init__(name, EvaluationMetric.COHERENCE, threshold, enabled)

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate coherence."""
        if not self.enabled or not response.strip():
            return EvaluationResult(
                metric=self.metric,
                score=1.0 if not response.strip() else 0.0,
                passed=True,
                threshold=self.threshold,
                evaluator_name=self.name,
            )

        scores = []
        details = {}

        # Check sentence structure
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Average sentence length (prefer 10-25 words)
            avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_words <= 25:
                sentence_score = 1.0
            elif 5 <= avg_words < 10 or 25 < avg_words <= 40:
                sentence_score = 0.8
            else:
                sentence_score = 0.5
            scores.append(sentence_score)
            details["avg_sentence_length"] = avg_words

        # Check for transition words (indicates logical flow)
        transitions = [
            "however", "therefore", "furthermore", "moreover", "additionally",
            "consequently", "meanwhile", "nevertheless", "otherwise", "thus",
            "first", "second", "finally", "next", "then", "also", "because",
        ]
        transition_count = sum(
            1 for t in transitions if t in response.lower()
        )
        transition_score = min(1.0, 0.5 + transition_count * 0.1)
        scores.append(transition_score)
        details["transition_words"] = transition_count

        # Check for repetition (bad)
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            repetition_score = unique_ratio
            scores.append(repetition_score)
            details["unique_word_ratio"] = unique_ratio

        # Check capitalization and basic structure
        has_capital_start = response[0].isupper() if response else False
        structure_score = 1.0 if has_capital_start else 0.7
        scores.append(structure_score)

        final_score = sum(scores) / len(scores) if scores else 0.5

        return EvaluationResult(
            metric=self.metric,
            score=final_score,
            passed=final_score >= self.threshold,
            threshold=self.threshold,
            evaluator_name=self.name,
            details=details,
        )


class FactualityEvaluator(Evaluator):
    """
    Basic factuality evaluation.

    Checks for:
    - Hedging language (indicates uncertainty)
    - Citation patterns
    - Claim confidence
    """

    HEDGING_PHRASES = [
        "i think", "i believe", "probably", "maybe", "perhaps",
        "might be", "could be", "possibly", "it seems", "appears to",
        "i'm not sure", "i don't know", "uncertain", "unclear",
    ]

    CONFIDENT_PHRASES = [
        "definitely", "certainly", "always", "never", "absolutely",
        "100%", "guaranteed", "proven", "fact", "undoubtedly",
    ]

    def __init__(
        self,
        name: str = "factuality",
        threshold: float = 0.6,
        enabled: bool = True,
    ):
        super().__init__(name, EvaluationMetric.FACTUALITY, threshold, enabled)

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate factuality indicators."""
        if not self.enabled:
            return EvaluationResult(
                metric=self.metric,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                evaluator_name=self.name,
            )

        response_lower = response.lower()

        # Count hedging (appropriate uncertainty is good)
        hedging_count = sum(
            1 for phrase in self.HEDGING_PHRASES
            if phrase in response_lower
        )

        # Count overconfidence (bad for factual claims)
        confident_count = sum(
            1 for phrase in self.CONFIDENT_PHRASES
            if phrase in response_lower
        )

        # Check for citations or references
        has_citations = bool(re.search(
            r"according to|source:|reference:|study|research|\[\d+\]",
            response_lower
        ))

        # Score calculation
        # Hedging is neutral to slightly good (shows appropriate uncertainty)
        # Overconfidence is bad
        # Citations are good

        base_score = 0.7
        hedging_modifier = min(0.1, hedging_count * 0.02)  # Small bonus
        confident_penalty = confident_count * 0.1  # Penalty for overconfidence
        citation_bonus = 0.15 if has_citations else 0

        score = min(1.0, max(0.0, base_score + hedging_modifier - confident_penalty + citation_bonus))

        return EvaluationResult(
            metric=self.metric,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            evaluator_name=self.name,
            details={
                "hedging_phrases": hedging_count,
                "confident_phrases": confident_count,
                "has_citations": has_citations,
            },
        )


class ToxicityEvaluator(Evaluator):
    """
    Evaluate response for toxic content.

    Inverse scoring - higher score means less toxic.
    """

    TOXIC_PATTERNS = [
        r"\b(hate|kill|die|stupid|idiot|dumb)\b",
        r"\b(racist|sexist|bigot)\b",
        r"(offensive|discriminat)",
    ]

    def __init__(
        self,
        name: str = "toxicity",
        threshold: float = 0.9,  # High threshold - want low toxicity
        enabled: bool = True,
    ):
        super().__init__(name, EvaluationMetric.TOXICITY, threshold, enabled)
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate toxicity (inverse - high score = low toxicity)."""
        if not self.enabled:
            return EvaluationResult(
                metric=self.metric,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                evaluator_name=self.name,
            )

        matches = []
        for pattern in self._patterns:
            found = pattern.findall(response)
            if found:
                matches.extend(found)

        # Inverse score - more matches = lower score
        if not matches:
            score = 1.0
        else:
            score = max(0.0, 1.0 - len(matches) * 0.2)

        return EvaluationResult(
            metric=self.metric,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            evaluator_name=self.name,
            details={
                "toxic_matches": matches[:5],
                "match_count": len(matches),
            },
        )


@dataclass
class AggregatedEvaluation:
    """Aggregated evaluation results."""

    overall_score: float
    passed: bool
    results: list[EvaluationResult]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp.isoformat(),
        }


class EvaluationPipeline:
    """
    Pipeline for running multiple evaluators.

    Aggregates results and calculates overall quality score.
    """

    def __init__(
        self,
        evaluators: list[Evaluator] | None = None,
        weights: dict[EvaluationMetric, float] | None = None,
        pass_threshold: float = 0.7,
        enabled: bool = True,
    ):
        self.evaluators = evaluators or [
            RelevanceEvaluator(),
            CoherenceEvaluator(),
            ToxicityEvaluator(),
        ]
        self.weights = weights or {
            EvaluationMetric.RELEVANCE: 0.35,
            EvaluationMetric.COHERENCE: 0.25,
            EvaluationMetric.FACTUALITY: 0.20,
            EvaluationMetric.TOXICITY: 0.20,
        }
        self.pass_threshold = pass_threshold
        self.enabled = enabled

    def add_evaluator(self, evaluator: Evaluator, weight: float = 0.25) -> None:
        """Add an evaluator to the pipeline."""
        self.evaluators.append(evaluator)
        self.weights[evaluator.metric] = weight

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> AggregatedEvaluation:
        """
        Run all evaluators and aggregate results.

        Args:
            prompt: Input prompt
            response: Model response
            context: Additional context

        Returns:
            AggregatedEvaluation with overall score
        """
        if not self.enabled:
            return AggregatedEvaluation(
                overall_score=1.0,
                passed=True,
                results=[],
            )

        results = []
        weighted_scores = []
        total_weight = 0.0

        for evaluator in self.evaluators:
            try:
                result = await evaluator.evaluate(prompt, response, context)
                results.append(result)

                weight = self.weights.get(evaluator.metric, 0.25)
                weighted_scores.append(result.score * weight)
                total_weight += weight
            except Exception as e:
                logger.warning(
                    "Evaluator failed",
                    evaluator=evaluator.name,
                    error=str(e),
                )

        # Calculate weighted average
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.5

        passed = overall_score >= self.pass_threshold and all(r.passed for r in results)

        return AggregatedEvaluation(
            overall_score=overall_score,
            passed=passed,
            results=results,
        )


# Global evaluation pipeline
_pipeline: EvaluationPipeline | None = None


def get_evaluation_pipeline() -> EvaluationPipeline:
    """Get the global evaluation pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = EvaluationPipeline()
    return _pipeline


def configure_evaluation_pipeline(
    evaluators: list[Evaluator] | None = None,
    weights: dict[EvaluationMetric, float] | None = None,
    pass_threshold: float = 0.7,
    enabled: bool = True,
) -> EvaluationPipeline:
    """Configure the global evaluation pipeline."""
    global _pipeline
    _pipeline = EvaluationPipeline(
        evaluators=evaluators,
        weights=weights,
        pass_threshold=pass_threshold,
        enabled=enabled,
    )
    return _pipeline
