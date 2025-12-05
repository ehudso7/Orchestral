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
]
