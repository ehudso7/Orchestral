"""
Cost optimization module for Orchestral.

Provides intelligent routing, cost prediction, and optimization
features for minimizing costs while maintaining quality.
"""

from orchestral.optimization.router import (
    OptimizationStrategy,
    ModelScore,
    SmartRouter,
    get_smart_router,
)

__all__ = [
    "OptimizationStrategy",
    "ModelScore",
    "SmartRouter",
    "get_smart_router",
]
