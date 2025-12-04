"""
Anthropic provider implementation for Claude models.

Supports Claude Opus 4.5, Sonnet 4.5, and Haiku with full async support.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Any

from anthropic import AsyncAnthropic, APIError, RateLimitError as AnthropicRateLimitError

from orchestral.core.models import (
    Message,
    MessageRole,
    ModelConfig,
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    UsageStats,
    ContentBlock,
    ContentType,
)
from orchestral.providers.base import (
    BaseProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ContextLengthError,
)


class AnthropicProvider(BaseProvider):
    """
    Anthropic provider for Claude models.

    Supports:
    - Claude Opus 4.5 (flagship, 80.9% SWE-bench)
    - Claude Sonnet 4.5 (balanced performance)
    - Claude Haiku 4.5 (fast and efficient)
    """

    provider = ModelProvider.ANTHROPIC

    # Approximate tokens per character for Claude
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(api_key, **kwargs)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )

    def _extract_system_message(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[Message]]:
        """Extract system message from message list (Claude handles it separately)."""
        system_content = None
        filtered_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                content = msg.content if isinstance(msg.content, str) else " ".join(
                    b.text or "" for b in msg.content if b.text
                )
                system_content = content
            else:
                filtered_messages.append(msg)

        return system_content, filtered_messages

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert Orchestral messages to Anthropic format."""
        result = []
        for msg in messages:
            converted: dict[str, Any] = {"role": msg.role.value}

            if isinstance(msg.content, str):
                converted["content"] = msg.content
            else:
                # Handle multimodal content
                content_parts = []
                for block in msg.content:
                    if block.type == ContentType.TEXT and block.text:
                        content_parts.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif block.type == ContentType.IMAGE:
                        if block.media_base64:
                            content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.mime_type or "image/png",
                                    "data": block.media_base64,
                                },
                            })
                        elif block.media_url:
                            # Claude prefers base64, but we can pass URL
                            content_parts.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": block.media_url,
                                },
                            })
                converted["content"] = content_parts

            result.append(converted)
        return result

    async def complete_async(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a completion using Claude."""
        start_time = time.perf_counter()
        model = request.config.model

        system_content, filtered_messages = self._extract_system_message(request.messages)

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": self._convert_messages(filtered_messages),
                "max_tokens": request.config.max_tokens,
                "temperature": request.config.temperature,
                "top_p": request.config.top_p,
            }

            if system_content:
                kwargs["system"] = system_content

            if request.config.stop:
                kwargs["stop_sequences"] = request.config.stop

            response = await self.client.messages.create(**kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract text content from response
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            usage = UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            return self._create_response(
                content=content,
                model=model,
                usage=usage,
                finish_reason=response.stop_reason,
                latency_ms=latency_ms,
                metadata={
                    "anthropic_id": response.id,
                    "model_version": response.model,
                },
            )

        except AnthropicRateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.provider,
            )
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(str(e), provider=self.provider)
            if e.status_code == 400 and "context" in str(e).lower():
                raise ContextLengthError(str(e), provider=self.provider)
            raise ProviderError(
                str(e),
                provider=self.provider,
                status_code=e.status_code,
                retryable=e.status_code >= 500,
            )

    async def stream_async(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[str]:
        """Stream a completion from Claude."""
        model = request.config.model
        system_content, filtered_messages = self._extract_system_message(request.messages)

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": self._convert_messages(filtered_messages),
                "max_tokens": request.config.max_tokens,
                "temperature": request.config.temperature,
                "top_p": request.config.top_p,
            }

            if system_content:
                kwargs["system"] = system_content

            if request.config.stop:
                kwargs["stop_sequences"] = request.config.stop

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e), provider=self.provider)
        except APIError as e:
            raise ProviderError(
                str(e),
                provider=self.provider,
                status_code=e.status_code,
                retryable=e.status_code >= 500,
            )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Approximate token count for Claude.

        Note: Anthropic doesn't provide a public tokenizer.
        This is an approximation based on ~4 characters per token.
        """
        return len(text) // self.CHARS_PER_TOKEN + 1

    def count_message_tokens(
        self,
        messages: list[Message],
        model: str | None = None,
    ) -> int:
        """Approximate token count for messages."""
        total = 0
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else " ".join(
                b.text or "" for b in msg.content if b.text
            )
            total += self.count_tokens(content)
            total += 4  # Overhead for message structure
        return total

    async def health_check(self) -> bool:
        """
        Check if the Anthropic provider is available and configured correctly.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Use Claude Haiku for health checks (fastest/cheapest)
            test_request = CompletionRequest(
                messages=[Message.user("Say 'OK'")],
                config=ModelConfig(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=10,
                    temperature=0,
                ),
            )
            response = await self.complete_async(test_request)
            return response is not None and len(response.content) > 0
        except Exception:
            return False
