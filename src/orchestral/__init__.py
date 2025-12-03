"""
Orchestral - AI Model Orchestration Platform

A unified interface for ChatGPT, Claude, and Gemini with intelligent routing,
parallel processing, and comprehensive comparison capabilities.
"""

__version__ = "1.0.0"
__author__ = "Orchestral Team"

from orchestral.core.orchestrator import Orchestrator
from orchestral.core.models import (
    ModelProvider,
    ModelConfig,
    Message,
    CompletionRequest,
    CompletionResponse,
    ComparisonResult,
)

__all__ = [
    "Orchestrator",
    "ModelProvider",
    "ModelConfig",
    "Message",
    "CompletionRequest",
    "CompletionResponse",
    "ComparisonResult",
]
