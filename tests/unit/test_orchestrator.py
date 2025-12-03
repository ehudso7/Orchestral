"""Tests for the Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from orchestral.core.orchestrator import Orchestrator, TASK_MODEL_RECOMMENDATIONS
from orchestral.core.models import (
    Message,
    ModelConfig,
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    UsageStats,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
    ModelTier,
)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    return CompletionResponse(
        id="test-openai-123",
        model="gpt-4o",
        provider=ModelProvider.OPENAI,
        content="OpenAI response",
        finish_reason="stop",
        usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
        latency_ms=100.0,
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response."""
    return CompletionResponse(
        id="test-anthropic-123",
        model="claude-sonnet-4-5-20250929",
        provider=ModelProvider.ANTHROPIC,
        content="Claude response",
        finish_reason="end_turn",
        usage=UsageStats(input_tokens=15, output_tokens=25, total_tokens=40),
        latency_ms=150.0,
    )


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""

    def test_init_without_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            orch = Orchestrator(auto_configure=False)
            assert len(orch.providers.available_providers) == 0

    def test_init_with_openai_key(self):
        orch = Orchestrator(
            openai_api_key="test-key",
            auto_configure=False,
        )
        assert ModelProvider.OPENAI in orch.providers.available_providers

    def test_init_with_all_keys(self):
        orch = Orchestrator(
            openai_api_key="test-openai",
            anthropic_api_key="test-anthropic",
            google_api_key="test-google",
            auto_configure=False,
        )
        assert len(orch.providers.available_providers) == 3


class TestOrchestratorComplete:
    """Tests for Orchestrator.complete()."""

    @pytest.mark.asyncio
    async def test_complete_with_string(self, mock_openai_response):
        orch = Orchestrator(openai_api_key="test-key", auto_configure=False)

        with patch.object(
            orch.providers.openai,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            response = await orch.complete("Hello, world!")

            assert response.content == "OpenAI response"
            assert response.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_complete_with_messages(self, mock_openai_response):
        orch = Orchestrator(openai_api_key="test-key", auto_configure=False)

        messages = [
            Message.system("You are helpful."),
            Message.user("Hello!"),
        ]

        with patch.object(
            orch.providers.openai,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            response = await orch.complete(messages)

            assert response.content == "OpenAI response"

    @pytest.mark.asyncio
    async def test_complete_with_unavailable_model(self):
        orch = Orchestrator(openai_api_key="test-key", auto_configure=False)

        with pytest.raises(ValueError, match="No provider available"):
            await orch.complete("Hello", model="claude-opus-4-5-20251101")


class TestOrchestratorCompare:
    """Tests for Orchestrator.compare()."""

    @pytest.mark.asyncio
    async def test_compare_multiple_models(
        self, mock_openai_response, mock_anthropic_response
    ):
        orch = Orchestrator(
            openai_api_key="test-openai",
            anthropic_api_key="test-anthropic",
            auto_configure=False,
        )

        with patch.object(
            orch.providers.openai,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ), patch.object(
            orch.providers.anthropic,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_anthropic_response,
        ):
            comparison = await orch.compare(
                "Test prompt",
                models=["gpt-4o", "claude-sonnet-4-5-20250929"],
            )

            assert len(comparison.results) == 2
            assert len(comparison.successful_results) == 2
            assert comparison.prompt == "Test prompt"

    @pytest.mark.asyncio
    async def test_compare_with_failure(self, mock_openai_response):
        orch = Orchestrator(
            openai_api_key="test-openai",
            anthropic_api_key="test-anthropic",
            auto_configure=False,
        )

        with patch.object(
            orch.providers.openai,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ), patch.object(
            orch.providers.anthropic,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            comparison = await orch.compare(
                "Test prompt",
                models=["gpt-4o", "claude-sonnet-4-5-20250929"],
            )

            assert len(comparison.successful_results) == 1
            assert len(comparison.failed_results) == 1


class TestOrchestratorRouting:
    """Tests for Orchestrator routing strategies."""

    @pytest.mark.asyncio
    async def test_route_single_strategy(self, mock_openai_response):
        orch = Orchestrator(openai_api_key="test-key", auto_configure=False)

        routing = RoutingConfig(
            strategy=RoutingStrategy.SINGLE,
            models=["gpt-4o"],
        )

        with patch.object(
            orch.providers.openai,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            response = await orch.route("Test", routing=routing)

            assert response.content == "OpenAI response"

    @pytest.mark.asyncio
    async def test_route_compare_strategy(
        self, mock_openai_response, mock_anthropic_response
    ):
        orch = Orchestrator(
            openai_api_key="test-openai",
            anthropic_api_key="test-anthropic",
            auto_configure=False,
        )

        routing = RoutingConfig(
            strategy=RoutingStrategy.COMPARE_ALL,
            models=["gpt-4o", "claude-sonnet-4-5-20250929"],
        )

        with patch.object(
            orch.providers.openai,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ), patch.object(
            orch.providers.anthropic,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_anthropic_response,
        ):
            result = await orch.route("Test", routing=routing)

            # Should return a ComparisonResult
            assert hasattr(result, "results")
            assert len(result.results) == 2


class TestTaskRecommendations:
    """Tests for task-based model recommendations."""

    def test_coding_recommendations(self):
        recommendations = TASK_MODEL_RECOMMENDATIONS[TaskCategory.CODING]
        # Claude Opus should be first for coding (80.9% SWE-bench)
        assert "claude-opus-4-5-20251101" in recommendations

    def test_multimodal_recommendations(self):
        recommendations = TASK_MODEL_RECOMMENDATIONS[TaskCategory.MULTIMODAL]
        # Gemini should be first for multimodal
        assert "gemini-3-ultra" in recommendations

    def test_get_best_model_for_task(self):
        orch = Orchestrator(
            openai_api_key="test-key",
            anthropic_api_key="test-anthropic",
            auto_configure=False,
        )

        # Should get Claude for coding if available
        best = orch.get_best_model_for_task(TaskCategory.CODING)
        assert best == "claude-opus-4-5-20251101"

    def test_get_best_model_with_tier(self):
        orch = Orchestrator(
            openai_api_key="test-key",
            auto_configure=False,
        )

        # Should get fast tier model
        best = orch.get_best_model_for_task(TaskCategory.CONVERSATION, tier=ModelTier.FAST)
        assert best is None or "mini" in best or "flash" in best or "haiku" in best


class TestOrchestratorHealth:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        orch = Orchestrator(openai_api_key="test-key", auto_configure=False)

        with patch.object(
            orch.providers.openai,
            "health_check",
            new_callable=AsyncMock,
            return_value=True,
        ):
            health = await orch.health_check()
            assert "openai" in health
            assert health["openai"] is True
