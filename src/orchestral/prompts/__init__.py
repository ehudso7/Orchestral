"""
Prompt management module for Orchestral.

Provides prompt versioning, templates, A/B testing, and management
capabilities for enterprise prompt engineering workflows.
"""

from orchestral.prompts.manager import (
    Prompt,
    PromptVersion,
    PromptTemplate,
    PromptManager,
    get_prompt_manager,
)
from orchestral.prompts.ab_testing import (
    Experiment,
    Variant,
    ExperimentResult,
    ABTestingManager,
    get_ab_manager,
)

__all__ = [
    "Prompt",
    "PromptVersion",
    "PromptTemplate",
    "PromptManager",
    "get_prompt_manager",
    "Experiment",
    "Variant",
    "ExperimentResult",
    "ABTestingManager",
    "get_ab_manager",
Prompt management and optimization module.

Provides versioned prompt templates, optimization, and A/B testing integration.
"""

from orchestral.prompts.manager import (
    PromptManager,
    PromptTemplate,
    PromptVersion,
    PromptConfig,
)
from orchestral.prompts.optimizer import (
    PromptOptimizer,
    OptimizationResult,
    OptimizationStrategy,
)

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "PromptVersion",
    "PromptConfig",
    "PromptOptimizer",
    "OptimizationResult",
    "OptimizationStrategy",
]
