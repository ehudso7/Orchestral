"""
Google provider implementation for Gemini models.

Supports Gemini 3 Ultra, Pro, and Flash with full multimodal capabilities.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

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


class GoogleProvider(BaseProvider):
    """
    Google provider for Gemini models.

    Supports:
    - Gemini 3 Ultra (flagship multimodal)
    - Gemini 3 Pro (balanced, ~1M context)
    - Gemini 3 Flash (fast and efficient)
    """

    provider = ModelProvider.GOOGLE

    # Approximate tokens per character for Gemini
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str = "us-central1",
        **kwargs: Any,
    ):
        super().__init__(api_key, **kwargs)
        self.project_id = project_id
        self.location = location

        if api_key:
            genai.configure(api_key=api_key)

        self._models: dict[str, genai.GenerativeModel] = {}

    def _get_model(self, model_id: str) -> genai.GenerativeModel:
        """Get or create a GenerativeModel instance."""
        if model_id not in self._models:
            self._models[model_id] = genai.GenerativeModel(model_id)
        return self._models[model_id]

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert Orchestral messages to Gemini format."""
        history = []
        system_instruction = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                content = msg.content if isinstance(msg.content, str) else " ".join(
                    b.text or "" for b in msg.content if b.text
                )
                system_instruction = content
                continue

            role = "user" if msg.role == MessageRole.USER else "model"

            if isinstance(msg.content, str):
                history.append({
                    "role": role,
                    "parts": [msg.content],
                })
            else:
                parts = []
                for block in msg.content:
                    if block.type == ContentType.TEXT and block.text:
                        parts.append(block.text)
                    elif block.type == ContentType.IMAGE:
                        if block.media_base64:
                            parts.append({
                                "inline_data": {
                                    "mime_type": block.mime_type or "image/png",
                                    "data": block.media_base64,
                                }
                            })
                        elif block.media_url:
                            # For URLs, we'd need to fetch and encode
                            parts.append(f"[Image: {block.media_url}]")
                    elif block.type == ContentType.VIDEO:
                        if block.media_base64:
                            parts.append({
                                "inline_data": {
                                    "mime_type": block.mime_type or "video/mp4",
                                    "data": block.media_base64,
                                }
                            })
                history.append({
                    "role": role,
                    "parts": parts,
                })

        return history, system_instruction

    def _create_generation_config(
        self,
        config: ModelConfig,
    ) -> GenerationConfig:
        """Create Gemini generation configuration."""
        return GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            stop_sequences=config.stop or None,
        )

    async def complete_async(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate a completion using Gemini."""
        start_time = time.perf_counter()
        model_id = request.config.model

        history, system_instruction = self._convert_messages(request.messages)

        try:
            # Create model with system instruction if present
            if system_instruction:
                model = genai.GenerativeModel(
                    model_id,
                    system_instruction=system_instruction,
                )
            else:
                model = self._get_model(model_id)

            # Get the last user message as the current prompt
            current_prompt = None
            chat_history = []

            if history:
                current_prompt = history[-1]["parts"]
                chat_history = history[:-1]

            # Start chat with history
            chat = model.start_chat(history=chat_history)

            # Generate response
            response = await chat.send_message_async(
                current_prompt or "",
                generation_config=self._create_generation_config(request.config),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract usage if available
            usage = UsageStats()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = UsageStats(
                    input_tokens=response.usage_metadata.prompt_token_count or 0,
                    output_tokens=response.usage_metadata.candidates_token_count or 0,
                    total_tokens=response.usage_metadata.total_token_count or 0,
                )

            return self._create_response(
                content=response.text,
                model=model_id,
                usage=usage,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                latency_ms=latency_ms,
                metadata={
                    "google_model": model_id,
                },
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise RateLimitError(str(e), provider=self.provider)
            if "auth" in error_str or "api key" in error_str:
                raise AuthenticationError(str(e), provider=self.provider)
            if "context" in error_str or "token" in error_str:
                raise ContextLengthError(str(e), provider=self.provider)
            raise ProviderError(
                str(e),
                provider=self.provider,
                retryable=True,
            )

    async def stream_async(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[str]:
        """Stream a completion from Gemini."""
        model_id = request.config.model
        history, system_instruction = self._convert_messages(request.messages)

        try:
            if system_instruction:
                model = genai.GenerativeModel(
                    model_id,
                    system_instruction=system_instruction,
                )
            else:
                model = self._get_model(model_id)

            current_prompt = None
            chat_history = []

            if history:
                current_prompt = history[-1]["parts"]
                chat_history = history[:-1]

            chat = model.start_chat(history=chat_history)

            response = await chat.send_message_async(
                current_prompt or "",
                generation_config=self._create_generation_config(request.config),
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise RateLimitError(str(e), provider=self.provider)
            raise ProviderError(str(e), provider=self.provider, retryable=True)

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens using Gemini's tokenizer.

        Falls back to approximation if API fails.
        """
        model_id = model or "gemini-3-pro-preview"
        try:
            model_instance = self._get_model(model_id)
            result = model_instance.count_tokens(text)
            return result.total_tokens
        except Exception:
            return len(text) // self.CHARS_PER_TOKEN + 1

    def count_message_tokens(
        self,
        messages: list[Message],
        model: str | None = None,
    ) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else " ".join(
                b.text or "" for b in msg.content if b.text
            )
            total += self.count_tokens(content, model)
            total += 4  # Message overhead
        return total
