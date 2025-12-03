"""
Base provider interface for AI models.

All provider implementations must inherit from BaseProvider.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from datetime import datetime
import uuid

from orchestral.core.models import (
    Message,
    ModelConfig,
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    UsageStats,
)


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: ModelProvider | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: ModelProvider | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, provider, status_code=429, retryable=True)
        self.retry_after = retry_after


class ContextLengthError(ProviderError):
    """Raised when context length is exceeded."""

    def __init__(
        self,
        message: str,
        provider: ModelProvider | None = None,
        token_count: int | None = None,
        max_tokens: int | None = None,
    ):
        super().__init__(message, provider, status_code=400, retryable=False)
        self.token_count = token_count
        self.max_tokens = max_tokens


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""

    def __init__(self, message: str, provider: ModelProvider | None = None):
        super().__init__(message, provider, status_code=401, retryable=False)


class BaseProvider(ABC):
    """
    Abstract base class for AI model providers.

    All provider implementations must implement:
    - complete(): Synchronous completion
    - complete_async(): Asynchronous completion
    - stream(): Streaming completion
    - count_tokens(): Token counting
    """

    provider: ModelProvider

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """Initialize the provider with optional API key."""
        self.api_key = api_key
        self._initialized = False

    @abstractmethod
    async def complete_async(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """
        Generate a completion asynchronously.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with the generated content
        """
        ...

    @abstractmethod
    async def stream_async(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Stream a completion asynchronously.

        Args:
            request: The completion request

        Yields:
            String chunks of the response
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens in text for the given model.

        Args:
            text: The text to count tokens for
            model: Optional model identifier

        Returns:
            Number of tokens
        """
        ...

    @abstractmethod
    def count_message_tokens(
        self,
        messages: list[Message],
        model: str | None = None,
    ) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of messages
            model: Optional model identifier

        Returns:
            Total number of tokens
        """
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion synchronously.

        This is a convenience wrapper around complete_async().
        """
        return asyncio.run(self.complete_async(request))

    async def health_check(self) -> bool:
        """
        Check if the provider is available and configured correctly.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test request
            test_request = CompletionRequest(
                messages=[Message.user("Say 'OK'")],
                config=ModelConfig(max_tokens=10, temperature=0),
            )
            response = await self.complete_async(test_request)
            return response is not None and len(response.content) > 0
        except Exception:
            return False

    def _create_response_id(self) -> str:
        """Generate a unique response ID."""
        return f"orch-{uuid.uuid4().hex[:16]}"

    def _create_response(
        self,
        content: str,
        model: str,
        usage: UsageStats | None = None,
        finish_reason: str | None = None,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        """Create a standardized completion response."""
        return CompletionResponse(
            id=self._create_response_id(),
            model=model,
            provider=self.provider,
            content=content,
            finish_reason=finish_reason,
            usage=usage or UsageStats(),
            latency_ms=latency_ms,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
        )
