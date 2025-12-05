"""
Evaluation module for Orchestral.

Provides quality scoring, automated evaluation, and feedback collection
for continuous improvement of LLM outputs.
"""

from orchestral.evaluation.evaluator import (
    EvaluationMetric,
    Evaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FactualityEvaluator,
    ToxicityEvaluator,
    AggregatedEvaluation,
    EvaluationPipeline,
    get_evaluation_pipeline,
    configure_evaluation_pipeline,
)

# Note: EvaluationResult from evaluator is intentionally not re-exported
# to avoid collision with scorer's EvaluationResult
from orchestral.evaluation.evaluator import EvaluationResult as MetricEvaluationResult

from orchestral.evaluation.scorer import (
    QualityScorer,
    QualityScore,
    EvaluationConfig,
    EvaluationResult,
)
from orchestral.evaluation.metrics import (
    ResponseMetrics,
    compute_readability,
    compute_coherence,
    detect_hallucination_signals,
)

__all__ = [
    # Evaluator exports
    "EvaluationMetric",
    "MetricEvaluationResult",
    "Evaluator",
    "RelevanceEvaluator",
    "CoherenceEvaluator",
    "FactualityEvaluator",
    "ToxicityEvaluator",
    "AggregatedEvaluation",
    "EvaluationPipeline",
    "get_evaluation_pipeline",
    "configure_evaluation_pipeline",
    # Scorer exports
    "QualityScorer",
    "QualityScore",
    "EvaluationConfig",
    "EvaluationResult",
    # Metrics exports
    "ResponseMetrics",
    "compute_readability",
    "compute_coherence",
    "detect_hallucination_signals",
]
