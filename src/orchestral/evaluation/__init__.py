"""
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
