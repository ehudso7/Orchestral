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
    - GPT-4o (standard multimodal)
    - GPT-4o-mini (fast and cost-effective)
    - o1, o3 reasoning models
    - o1/o3 series (reasoning models)
    """

    provider = ModelProvider.OPENAI

    # Models that require max_completion_tokens instead of max_tokens
    # These models don't support temperature, top_p, frequency_penalty, etc.
    COMPLETION_TOKEN_MODELS = frozenset({
        "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o3-preview",
        "gpt-5.1", "gpt-5",  # Newer flagship models
    })

    def _uses_completion_tokens(self, model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens."""
        model_lower = model.lower()
        # Check exact prefix matches for reasoning/new models
        return any(model_lower.startswith(rm) for rm in self.COMPLETION_TOKEN_MODELS)
    REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "gpt-5.1"}

    # Map fictional/placeholder models to real ones
    MODEL_ALIASES = {
        "gpt-5.1": "gpt-4o",  # Fallback to gpt-4o for fictional model
    }

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

    def _resolve_model(self, model: str) -> str:
        """Resolve model aliases to actual API model names."""
        return self.MODEL_ALIASES.get(model, model)

    def _needs_completion_tokens(self, model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens."""
        return any(model.startswith(rm) for rm in self.REASONING_MODELS)

    async def complete_async(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a completion using OpenAI."""
        start_time = time.perf_counter()
        model = request.config.model
        is_reasoning = self._uses_completion_tokens(model)

        try:
            # Build params - reasoning models use different parameters
            params: dict[str, Any] = {
                "model": model,
                "messages": self._convert_messages(request.messages),
                "stream": False,
            }

            if is_reasoning:
                # Reasoning models (o1, o3) use max_completion_tokens
                # and don't support temperature, top_p, etc.
                params["max_completion_tokens"] = request.config.max_tokens
            else:
                # Standard models use max_tokens and support all params
                params["max_tokens"] = request.config.max_tokens
                params["temperature"] = request.config.temperature
                params["top_p"] = request.config.top_p
                params["frequency_penalty"] = request.config.frequency_penalty
                params["presence_penalty"] = request.config.presence_penalty
                params["stop"] = request.config.stop

        model = self._resolve_model(request.config.model)

        # Build base parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(request.messages),
            "stream": False,
        }

        # Handle max tokens - newer models use max_completion_tokens
        if self._needs_completion_tokens(model):
            params["max_completion_tokens"] = request.config.max_tokens
        else:
            params["max_tokens"] = request.config.max_tokens
            params["temperature"] = request.config.temperature
            params["top_p"] = request.config.top_p
            params["frequency_penalty"] = request.config.frequency_penalty
            params["presence_penalty"] = request.config.presence_penalty
            if request.config.stop:
                params["stop"] = request.config.stop

        try:
            response = await self.client.chat.completions.create(**params)

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
        is_reasoning = self._uses_completion_tokens(model)

        try:
            # Build params - reasoning models use different parameters
            params: dict[str, Any] = {
                "model": model,
                "messages": self._convert_messages(request.messages),
                "stream": True,
            }

            if is_reasoning:
                # Reasoning models (o1, o3) use max_completion_tokens
                params["max_completion_tokens"] = request.config.max_tokens
            else:
                # Standard models use max_tokens and support all params
                params["max_tokens"] = request.config.max_tokens
                params["temperature"] = request.config.temperature
                params["top_p"] = request.config.top_p
                params["frequency_penalty"] = request.config.frequency_penalty
                params["presence_penalty"] = request.config.presence_penalty
                params["stop"] = request.config.stop

        model = self._resolve_model(request.config.model)

        # Build base parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(request.messages),
            "stream": True,
        }

        # Handle max tokens - newer models use max_completion_tokens
        if self._needs_completion_tokens(model):
            params["max_completion_tokens"] = request.config.max_tokens
        else:
            params["max_tokens"] = request.config.max_tokens
            params["temperature"] = request.config.temperature
            params["top_p"] = request.config.top_p
            params["frequency_penalty"] = request.config.frequency_penalty
            params["presence_penalty"] = request.config.presence_penalty
            if request.config.stop:
                params["stop"] = request.config.stop

        try:
            stream = await self.client.chat.completions.create(**params)

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
