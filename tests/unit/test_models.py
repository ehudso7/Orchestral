"""Tests for core data models."""

import pytest
from datetime import datetime

from orchestral.core.models import (
    Message,
    MessageRole,
    ModelConfig,
    CompletionRequest,
    CompletionResponse,
    UsageStats,
    ModelProvider,
    ModelSpec,
    ModelTier,
    MODEL_REGISTRY,
    ComparisonResult,
    ModelResult,
    ComparisonMetrics,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
)


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        msg = Message.user("Hello, world!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.name is None

    def test_create_assistant_message(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant."

    def test_message_with_name(self):
        msg = Message(role=MessageRole.USER, content="Test", name="Alice")
        assert msg.name == "Alice"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.top_p == 1.0

    def test_custom_values(self):
        config = ModelConfig(
            model="claude-opus-4-5-20251101",
            temperature=0.2,
            max_tokens=8192,
        )
        assert config.model == "claude-opus-4-5-20251101"
        assert config.temperature == 0.2
        assert config.max_tokens == 8192

    def test_temperature_validation(self):
        with pytest.raises(ValueError):
            ModelConfig(temperature=3.0)

        with pytest.raises(ValueError):
            ModelConfig(temperature=-0.5)

    def test_get_spec(self):
        config = ModelConfig(model="gpt-4o")
        spec = config.spec
        assert spec is not None
        assert spec.provider == ModelProvider.OPENAI


class TestUsageStats:
    """Tests for UsageStats."""

    def test_default_values(self):
        usage = UsageStats()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_estimated_cost(self):
        usage = UsageStats(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = usage.estimated_cost
        assert cost > 0


class TestCompletionResponse:
    """Tests for CompletionResponse."""

    def test_create_response(self):
        response = CompletionResponse(
            id="test-123",
            model="gpt-4o",
            provider=ModelProvider.OPENAI,
            content="Hello!",
            finish_reason="stop",
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
            latency_ms=150.5,
        )
        assert response.id == "test-123"
        assert response.model == "gpt-4o"
        assert response.provider == ModelProvider.OPENAI
        assert response.content == "Hello!"
        assert response.latency_ms == 150.5


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_openai_models_exist(self):
        assert "gpt-5.1" in MODEL_REGISTRY
        assert "gpt-4o" in MODEL_REGISTRY
        assert "gpt-4o-mini" in MODEL_REGISTRY

    def test_anthropic_models_exist(self):
        assert "claude-opus-4-5-20251101" in MODEL_REGISTRY
        assert "claude-sonnet-4-5-20250929" in MODEL_REGISTRY

    def test_google_models_exist(self):
        assert "gemini-3-pro-preview" in MODEL_REGISTRY
        assert "gemini-2.5-pro" in MODEL_REGISTRY
        assert "gemini-2.5-flash" in MODEL_REGISTRY

    def test_model_spec_properties(self):
        spec = MODEL_REGISTRY["gpt-5.1"]
        assert spec.provider == ModelProvider.OPENAI
        assert spec.tier == ModelTier.FLAGSHIP
        assert spec.context_window == 400_000
        assert spec.supports_vision is True

    def test_gemini_multimodal_support(self):
        spec = MODEL_REGISTRY["gemini-3-pro-preview"]
        assert spec.supports_vision is True
        assert spec.supports_audio is True
        assert spec.supports_video is True


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_successful_results(self):
        results = [
            ModelResult(
                model="gpt-4o",
                provider=ModelProvider.OPENAI,
                response=CompletionResponse(
                    id="1",
                    model="gpt-4o",
                    provider=ModelProvider.OPENAI,
                    content="Response 1",
                ),
                success=True,
            ),
            ModelResult(
                model="claude-sonnet-4-5-20250929",
                provider=ModelProvider.ANTHROPIC,
                error="Rate limited",
                success=False,
            ),
        ]

        comparison = ComparisonResult(
            id="cmp-123",
            prompt="Test prompt",
            results=results,
        )

        assert len(comparison.successful_results) == 1
        assert len(comparison.failed_results) == 1
        assert comparison.successful_results[0].model == "gpt-4o"
        assert comparison.failed_results[0].model == "claude-sonnet-4-5-20250929"


class TestRoutingConfig:
    """Tests for RoutingConfig."""

    def test_default_config(self):
        config = RoutingConfig()
        assert config.strategy == RoutingStrategy.SINGLE
        assert config.models == ["gpt-4o"]
        assert config.max_parallel == 3

    def test_custom_config(self):
        config = RoutingConfig(
            strategy=RoutingStrategy.BEST_FOR_TASK,
            task_category=TaskCategory.CODING,
            models=["claude-opus-4-5-20251101", "gpt-5.1"],
        )
        assert config.strategy == RoutingStrategy.BEST_FOR_TASK
        assert config.task_category == TaskCategory.CODING
        assert len(config.models) == 2
