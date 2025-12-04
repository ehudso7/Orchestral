"""Tests for provider implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestral.providers.base import (
    BaseProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ContextLengthError,
)
from orchestral.providers.openai_provider import OpenAIProvider
from orchestral.providers.anthropic_provider import AnthropicProvider
from orchestral.providers.google_provider import GoogleProvider
from orchestral.core.models import (
    Message,
    MessageRole,
    ModelConfig,
    CompletionRequest,
    ModelProvider,
)


class TestProviderErrors:
    """Tests for provider error classes."""

    def test_provider_error(self):
        error = ProviderError(
            "Test error",
            provider=ModelProvider.OPENAI,
            status_code=500,
            retryable=True,
        )
        assert str(error) == "Test error"
        assert error.provider == ModelProvider.OPENAI
        assert error.status_code == 500
        assert error.retryable is True

    def test_rate_limit_error(self):
        error = RateLimitError(
            "Rate limited",
            provider=ModelProvider.ANTHROPIC,
            retry_after=30.0,
        )
        assert error.status_code == 429
        assert error.retryable is True
        assert error.retry_after == 30.0

    def test_authentication_error(self):
        error = AuthenticationError(
            "Invalid API key",
            provider=ModelProvider.GOOGLE,
        )
        assert error.status_code == 401
        assert error.retryable is False

    def test_context_length_error(self):
        error = ContextLengthError(
            "Context too long",
            provider=ModelProvider.OPENAI,
            token_count=300000,
            max_tokens=272000,
        )
        assert error.status_code == 400
        assert error.retryable is False
        assert error.token_count == 300000


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_init(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider == ModelProvider.OPENAI
        assert provider.api_key == "test-key"

    def test_convert_messages(self):
        provider = OpenAIProvider(api_key="test-key")
        messages = [
            Message.system("Be helpful"),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]
        converted = provider._convert_messages(messages)

        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"
        assert converted[2]["role"] == "assistant"

    def test_count_tokens(self):
        provider = OpenAIProvider(api_key="test-key")
        count = provider.count_tokens("Hello, world!")
        assert count > 0

    def test_count_message_tokens(self):
        provider = OpenAIProvider(api_key="test-key")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi!"),
        ]
        count = provider.count_message_tokens(messages)
        assert count > 0


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_init(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.provider == ModelProvider.ANTHROPIC

    def test_extract_system_message(self):
        provider = AnthropicProvider(api_key="test-key")
        messages = [
            Message.system("Be helpful"),
            Message.user("Hello"),
        ]
        system, filtered = provider._extract_system_message(messages)

        assert system == "Be helpful"
        assert len(filtered) == 1
        assert filtered[0].role == MessageRole.USER

    def test_convert_messages(self):
        provider = AnthropicProvider(api_key="test-key")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi!"),
        ]
        converted = provider._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_count_tokens_approximation(self):
        provider = AnthropicProvider(api_key="test-key")
        # Test that approximation works (4 chars per token)
        text = "a" * 100
        count = provider.count_tokens(text)
        assert 20 <= count <= 30  # Approximately 25 tokens


class TestGoogleProvider:
    """Tests for Google provider."""

    def test_init(self):
        with patch("google.generativeai.configure"):
            provider = GoogleProvider(api_key="test-key")
            assert provider.provider == ModelProvider.GOOGLE

    def test_convert_messages(self):
        with patch("google.generativeai.configure"):
            provider = GoogleProvider(api_key="test-key")
            messages = [
                Message.system("Be helpful"),
                Message.user("Hello"),
                Message.assistant("Hi!"),
            ]
            history, system = provider._convert_messages(messages)

            assert system == "Be helpful"
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "model"


class TestProviderTokenCounting:
    """Tests for token counting across providers."""

    def test_openai_token_counting(self):
        provider = OpenAIProvider(api_key="test-key")

        # Simple text
        count1 = provider.count_tokens("Hello")
        count2 = provider.count_tokens("Hello, how are you today?")
        assert count2 > count1

    def test_anthropic_token_approximation(self):
        provider = AnthropicProvider(api_key="test-key")

        # Should approximate ~4 chars per token
        count = provider.count_tokens("a" * 400)
        assert 90 <= count <= 110  # Approximately 100 tokens


class TestProviderHealthCheck:
    """Tests for provider health check functionality."""

    def test_openai_health_check_model(self):
        """Test that OpenAI provider uses gpt-4o for health checks."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._health_check_model == "gpt-4o"

    def test_anthropic_health_check_model(self):
        """Test that Anthropic provider uses Claude Haiku for health checks."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider._health_check_model == "claude-haiku-4-5-20251001"

    def test_google_health_check_model(self):
        """Test that Google provider uses Gemini Flash for health checks."""
        with patch("google.generativeai.configure"):
            provider = GoogleProvider(api_key="test-key")
            assert provider._health_check_model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check returns True."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "OK"

        with patch.object(provider, "complete_async", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response
            result = await provider.health_check()

            assert result is True
            mock_complete.assert_called_once()
            # Verify the correct model was used
            call_args = mock_complete.call_args[0][0]
            assert call_args.config.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check returns False on exception."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(provider, "complete_async", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = Exception("API error")
            result = await provider.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_empty_response(self):
        """Test health check returns False on empty response."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = ""

        with patch.object(provider, "complete_async", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response
            result = await provider.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_anthropic_health_check_uses_correct_model(self):
        """Test that Anthropic health check uses the correct model."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "OK"

        with patch.object(provider, "complete_async", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response
            await provider.health_check()

            call_args = mock_complete.call_args[0][0]
            assert call_args.config.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_google_health_check_uses_correct_model(self):
        """Test that Google health check uses the correct model."""
        with patch("google.generativeai.configure"):
            provider = GoogleProvider(api_key="test-key")

            mock_response = MagicMock()
            mock_response.content = "OK"

            with patch.object(provider, "complete_async", new_callable=AsyncMock) as mock_complete:
                mock_complete.return_value = mock_response
                await provider.health_check()

                call_args = mock_complete.call_args[0][0]
                assert call_args.config.model == "gemini-2.5-flash"
