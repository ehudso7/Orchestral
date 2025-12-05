"""
Evaluation module for Orchestral.

Provides quality scoring, automated evaluation, and feedback collection
for continuous improvement of LLM outputs.
"""

from orchestral.evaluation.evaluator import (
    EvaluationMetric,
    EvaluationResult,
    Evaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FactualityEvaluator,
    ToxicityEvaluator,
    EvaluationPipeline,
    get_evaluation_pipeline,
)

__all__ = [
    "EvaluationMetric",
    "EvaluationResult",
    "Evaluator",
    "RelevanceEvaluator",
    "CoherenceEvaluator",
    "FactualityEvaluator",
    "ToxicityEvaluator",
    "EvaluationPipeline",
    "get_evaluation_pipeline",
Response evaluation and quality scoring module.

Provides automatic evaluation of LLM responses across multiple dimensions
including relevance, coherence, accuracy, and safety.
"""

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
    "QualityScorer",
    "QualityScore",
    "EvaluationConfig",
    "EvaluationResult",
    "ResponseMetrics",
    "compute_readability",
    "compute_coherence",
    "detect_hallucination_signals",
]
