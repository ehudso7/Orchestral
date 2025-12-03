"""
Core data models for the Orchestral platform.

Defines unified types for multi-model orchestration, comparison, and routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class ModelProvider(str, Enum):
    """Supported AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelTier(str, Enum):
    """Model capability tiers for routing decisions."""

    FLAGSHIP = "flagship"      # Top-tier models (GPT-5.1, Claude Opus 4.5, Gemini 3 Ultra)
    STANDARD = "standard"      # Mid-tier (GPT-4o, Claude Sonnet 4.5, Gemini 3 Pro)
    FAST = "fast"              # Speed-optimized (GPT-4o-mini, Claude Haiku, Gemini Flash)


@dataclass
class ModelSpec:
    """Specification for a specific AI model."""

    provider: ModelProvider
    model_id: str
    tier: ModelTier
    context_window: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            "gpt-5.1": "ChatGPT 5.1",
            "gpt-5.1-chat-latest": "ChatGPT 5.1 Instant",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
            "claude-opus-4-5-20251101": "Claude Opus 4.5",
            "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
            "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
            "gemini-3-ultra": "Gemini 3 Ultra",
            "gemini-3-pro-preview": "Gemini 3 Pro",
            "gemini-3-flash": "Gemini 3 Flash",
        }
        return names.get(self.model_id, self.model_id)


# Model Registry with current specifications (November 2025)
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # OpenAI Models
    "gpt-5.1": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_id="gpt-5.1",
        tier=ModelTier.FLAGSHIP,
        context_window=272_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_audio=True,
        input_cost_per_million=1.25,
        output_cost_per_million=10.00,
    ),
    "gpt-4o": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        tier=ModelTier.STANDARD,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-mini": ModelSpec(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        tier=ModelTier.FAST,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    # Anthropic Models
    "claude-opus-4-5-20251101": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-5-20251101",
        tier=ModelTier.FLAGSHIP,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        input_cost_per_million=5.00,
        output_cost_per_million=25.00,
    ),
    "claude-sonnet-4-5-20250929": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250929",
        tier=ModelTier.STANDARD,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-haiku-4-5-20251001": ModelSpec(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        tier=ModelTier.FAST,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
    ),
    # Google Models
    "gemini-3-ultra": ModelSpec(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-3-ultra",
        tier=ModelTier.FLAGSHIP,
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "gemini-3-pro-preview": ModelSpec(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-3-pro-preview",
        tier=ModelTier.STANDARD,
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        input_cost_per_million=2.00,
        output_cost_per_million=12.00,
    ),
    "gemini-3-flash": ModelSpec(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-3-flash",
        tier=ModelTier.FAST,
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
}


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(str, Enum):
    """Types of content in messages."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class ContentBlock(BaseModel):
    """A block of content within a message."""

    model_config = ConfigDict(frozen=True)

    type: ContentType = ContentType.TEXT
    text: str | None = None
    media_url: str | None = None
    media_base64: str | None = None
    mime_type: str | None = None
    file_name: str | None = None


class Message(BaseModel):
    """A message in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str | list[ContentBlock]
    name: str | None = None

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)


class ModelConfig(BaseModel):
    """Configuration for a model request."""

    model_config = ConfigDict(frozen=True)

    model: str = "gpt-4o"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: list[str] | None = None

    @property
    def spec(self) -> ModelSpec | None:
        """Get the model specification."""
        return MODEL_REGISTRY.get(self.model)


class CompletionRequest(BaseModel):
    """Request for a completion from one or more models."""

    model_config = ConfigDict(frozen=True)

    messages: list[Message]
    config: ModelConfig = Field(default_factory=ModelConfig)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageStats(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on typical pricing."""
        # This is a rough estimate - actual costs vary by model
        return (self.input_tokens * 0.003 + self.output_tokens * 0.015) / 1000


class CompletionResponse(BaseModel):
    """Response from a model completion."""

    model_config = ConfigDict(frozen=True)

    id: str
    model: str
    provider: ModelProvider
    content: str
    finish_reason: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonMetrics(BaseModel):
    """Metrics for comparing model responses."""

    model_config = ConfigDict(frozen=True)

    response_length: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    estimated_cost: float = 0.0

    # Quality scores (0-100, computed by evaluator)
    relevance_score: float | None = None
    coherence_score: float | None = None
    accuracy_score: float | None = None
    creativity_score: float | None = None


class ModelResult(BaseModel):
    """Result from a single model in a comparison."""

    model_config = ConfigDict(frozen=True)

    model: str
    provider: ModelProvider
    response: CompletionResponse | None = None
    error: str | None = None
    metrics: ComparisonMetrics = Field(default_factory=ComparisonMetrics)
    success: bool = True


class ComparisonResult(BaseModel):
    """Result of comparing multiple models on the same prompt."""

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    results: list[ModelResult]
    winner: str | None = None  # Model ID of the "best" response
    comparison_notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def successful_results(self) -> list[ModelResult]:
        """Get only successful results."""
        return [r for r in self.results if r.success]

    @property
    def failed_results(self) -> list[ModelResult]:
        """Get only failed results."""
        return [r for r in self.results if not r.success]


class TaskCategory(str, Enum):
    """Categories of tasks for intelligent routing."""

    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    MULTIMODAL = "multimodal"
    CONVERSATION = "conversation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


class RoutingStrategy(str, Enum):
    """Strategies for routing requests to models."""

    SINGLE = "single"           # Use a single specified model
    FASTEST = "fastest"         # Use the fastest model
    CHEAPEST = "cheapest"       # Use the cheapest model
    BEST_FOR_TASK = "best"      # Use the best model for the task category
    COMPARE_ALL = "compare"     # Query all models and compare
    FALLBACK = "fallback"       # Try models in order until one succeeds
    CONSENSUS = "consensus"     # Query multiple models and find consensus


@dataclass
class RoutingConfig:
    """Configuration for request routing."""

    strategy: RoutingStrategy = RoutingStrategy.SINGLE
    models: list[str] = field(default_factory=lambda: ["gpt-4o"])
    task_category: TaskCategory | None = None
    max_parallel: int = 3
    timeout_seconds: float = 120.0
    fallback_on_error: bool = True
