"""
OpenAI provider implementation for ChatGPT models.

Supports GPT-5.1, GPT-4o, and other OpenAI models with full async support.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Any

import tiktoken
from openai import AsyncOpenAI, APIError, RateLimitError as OpenAIRateLimitError

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


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider for ChatGPT models.

    Supports:
    - GPT-5.1 (flagship with adaptive reasoning)
    - GPT-4o (standard multimodal)
    - GPT-4o-mini (fast and cost-effective)
    """

    provider = ModelProvider.OPENAI

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
        )
        self._tokenizers: dict[str, tiktoken.Encoding] = {}

    def _get_tokenizer(self, model: str) -> tiktoken.Encoding:
        """Get or create tokenizer for model."""
        if model not in self._tokenizers:
            try:
                self._tokenizers[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for newer models
                self._tokenizers[model] = tiktoken.get_encoding("cl100k_base")
        return self._tokenizers[model]

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert Orchestral messages to OpenAI format."""
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
                        if block.media_url:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": block.media_url},
                            })
                        elif block.media_base64:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.mime_type or 'image/png'};base64,{block.media_base64}"
                                },
                            })
                converted["content"] = content_parts

            if msg.name:
                converted["name"] = msg.name

            result.append(converted)
        return result

    async def complete_async(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a completion using OpenAI."""
        start_time = time.perf_counter()
        model = request.config.model

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=self._convert_messages(request.messages),
                temperature=request.config.temperature,
                max_tokens=request.config.max_tokens,
                top_p=request.config.top_p,
                frequency_penalty=request.config.frequency_penalty,
                presence_penalty=request.config.presence_penalty,
                stop=request.config.stop,
                stream=False,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            usage = UsageStats(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return self._create_response(
                content=response.choices[0].message.content or "",
                model=model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms,
                metadata={
                    "openai_id": response.id,
                    "system_fingerprint": response.system_fingerprint,
                },
            )

        except OpenAIRateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.provider,
                retry_after=float(e.response.headers.get("retry-after", 60)),
            )
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(str(e), provider=self.provider)
            if e.status_code == 400 and "context_length" in str(e).lower():
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
        """Stream a completion from OpenAI."""
        model = request.config.model

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=self._convert_messages(request.messages),
                temperature=request.config.temperature,
                max_tokens=request.config.max_tokens,
                top_p=request.config.top_p,
                frequency_penalty=request.config.frequency_penalty,
                presence_penalty=request.config.presence_penalty,
                stop=request.config.stop,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e), provider=self.provider)
        except APIError as e:
            raise ProviderError(
                str(e),
                provider=self.provider,
                status_code=e.status_code,
                retryable=e.status_code >= 500,
            )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in text."""
        model = model or "gpt-4o"
        tokenizer = self._get_tokenizer(model)
        return len(tokenizer.encode(text))

    def count_message_tokens(
        self,
        messages: list[Message],
        model: str | None = None,
    ) -> int:
        """Count tokens in messages following OpenAI's format."""
        model = model or "gpt-4o"
        tokenizer = self._get_tokenizer(model)

        # Token overhead per message (role, separators, etc.)
        # Per OpenAI's tiktoken docs: 3 tokens per message + 1 if name is included
        tokens_per_message = 3

        total = 0
        for msg in messages:
            total += tokens_per_message
            content = msg.content if isinstance(msg.content, str) else " ".join(
                b.text or "" for b in msg.content if b.text
            )
            total += len(tokenizer.encode(content))
            total += len(tokenizer.encode(msg.role.value))
            if msg.name:
                total += len(tokenizer.encode(msg.name)) + 1

        total += 3  # Reply priming tokens
        return total
