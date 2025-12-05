"""
A/B testing and experimentation framework.

Run controlled experiments to compare models, prompts, and configurations.
"""

from orchestral.experiments.ab_testing import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    Variant,
    ABTestRunner,
)

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "Variant",
    "ABTestRunner",
]
