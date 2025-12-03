"""
Core Orchestrator - The heart of the multi-model AI platform.

Provides unified interface for querying multiple AI models with:
- Parallel execution
- Intelligent routing
- Comparison and consensus
- Fallback handling
- Cost optimization
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator

import structlog

from orchestral.core.models import (
    Message,
    ModelConfig,
    ModelProvider,
    CompletionRequest,
    CompletionResponse,
    ComparisonResult,
    ComparisonMetrics,
    ModelResult,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
    MODEL_REGISTRY,
    ModelTier,
)
from orchestral.core.config import get_provider_settings, get_settings
from orchestral.providers.base import BaseProvider, ProviderError, RateLimitError
from orchestral.providers.openai_provider import OpenAIProvider
from orchestral.providers.anthropic_provider import AnthropicProvider
from orchestral.providers.google_provider import GoogleProvider

logger = structlog.get_logger()


# Task to model recommendations based on benchmarks
TASK_MODEL_RECOMMENDATIONS: dict[TaskCategory, list[str]] = {
    TaskCategory.CODING: [
        "claude-opus-4-5-20251101",  # 80.9% SWE-bench
        "gpt-5.1",
        "gemini-3-pro-preview",
    ],
    TaskCategory.REASONING: [
        "gpt-5.1",  # Adaptive reasoning
        "claude-opus-4-5-20251101",
        "gemini-3-ultra",
    ],
    TaskCategory.CREATIVE: [
        "gpt-5.1",
        "gemini-3-pro-preview",
        "claude-sonnet-4-5-20250929",
    ],
    TaskCategory.MULTIMODAL: [
        "gemini-3-ultra",  # Best multimodal (text, image, video, audio)
        "gemini-3-pro-preview",
        "gpt-5.1",
    ],
    TaskCategory.ANALYSIS: [
        "claude-opus-4-5-20251101",  # Long context coherence
        "gemini-3-pro-preview",  # 1M context
        "gpt-5.1",
    ],
    TaskCategory.CONVERSATION: [
        "gpt-4o",
        "claude-sonnet-4-5-20250929",
        "gemini-3-flash",
    ],
    TaskCategory.SUMMARIZATION: [
        "claude-opus-4-5-20251101",
        "gemini-3-pro-preview",
        "gpt-4o",
    ],
    TaskCategory.TRANSLATION: [
        "gpt-4o",
        "gemini-3-pro-preview",
        "claude-sonnet-4-5-20250929",
    ],
}


@dataclass
class ProviderRegistry:
    """Registry of available providers."""

    openai: OpenAIProvider | None = None
    anthropic: AnthropicProvider | None = None
    google: GoogleProvider | None = None

    def get_provider(self, provider: ModelProvider) -> BaseProvider | None:
        """Get provider by type."""
        mapping = {
            ModelProvider.OPENAI: self.openai,
            ModelProvider.ANTHROPIC: self.anthropic,
            ModelProvider.GOOGLE: self.google,
        }
        return mapping.get(provider)

    def get_provider_for_model(self, model_id: str) -> BaseProvider | None:
        """Get provider for a specific model."""
        spec = MODEL_REGISTRY.get(model_id)
        if spec:
            return self.get_provider(spec.provider)
        return None

    @property
    def available_providers(self) -> list[ModelProvider]:
        """List available providers."""
        providers = []
        if self.openai:
            providers.append(ModelProvider.OPENAI)
        if self.anthropic:
            providers.append(ModelProvider.ANTHROPIC)
        if self.google:
            providers.append(ModelProvider.GOOGLE)
        return providers


class Orchestrator:
    """
    Multi-model AI orchestrator.

    Provides a unified interface for:
    - Single model queries
    - Parallel multi-model comparison
    - Intelligent task-based routing
    - Fallback and retry handling
    - Cost-optimized selection
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        auto_configure: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            openai_api_key: OpenAI API key (or from env)
            anthropic_api_key: Anthropic API key (or from env)
            google_api_key: Google API key (or from env)
            auto_configure: Auto-load from environment if True
        """
        self.settings = get_settings()
        self.providers = ProviderRegistry()

        if auto_configure:
            provider_settings = get_provider_settings()
            openai_api_key = openai_api_key or (
                provider_settings.openai_api_key.get_secret_value()
                if provider_settings.openai_api_key else None
            )
            anthropic_api_key = anthropic_api_key or (
                provider_settings.anthropic_api_key.get_secret_value()
                if provider_settings.anthropic_api_key else None
            )
            google_api_key = google_api_key or (
                provider_settings.google_api_key.get_secret_value()
                if provider_settings.google_api_key else None
            )

        # Initialize available providers
        if openai_api_key:
            self.providers.openai = OpenAIProvider(api_key=openai_api_key)
            logger.info("OpenAI provider initialized")

        if anthropic_api_key:
            self.providers.anthropic = AnthropicProvider(api_key=anthropic_api_key)
            logger.info("Anthropic provider initialized")

        if google_api_key:
            self.providers.google = GoogleProvider(api_key=google_api_key)
            logger.info("Google provider initialized")

        logger.info(
            "Orchestrator initialized",
            available_providers=[p.value for p in self.providers.available_providers],
        )

    async def complete(
        self,
        messages: list[Message] | str,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion from a single model.

        Args:
            messages: List of messages or a single prompt string
            model: Model ID to use
            **kwargs: Additional ModelConfig parameters

        Returns:
            CompletionResponse from the model
        """
        if isinstance(messages, str):
            messages = [Message.user(messages)]

        config = ModelConfig(model=model, **kwargs)
        request = CompletionRequest(messages=messages, config=config)

        provider = self.providers.get_provider_for_model(model)
        if not provider:
            raise ValueError(f"No provider available for model: {model}")

        return await provider.complete_async(request)

    async def stream(
        self,
        messages: list[Message] | str,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion from a single model.

        Args:
            messages: List of messages or a single prompt string
            model: Model ID to use
            **kwargs: Additional ModelConfig parameters

        Yields:
            String chunks of the response
        """
        if isinstance(messages, str):
            messages = [Message.user(messages)]

        config = ModelConfig(model=model, **kwargs)
        request = CompletionRequest(messages=messages, config=config, stream=True)

        provider = self.providers.get_provider_for_model(model)
        if not provider:
            raise ValueError(f"No provider available for model: {model}")

        async for chunk in provider.stream_async(request):
            yield chunk

    async def compare(
        self,
        messages: list[Message] | str,
        models: list[str] | None = None,
        **kwargs: Any,
    ) -> ComparisonResult:
        """
        Query multiple models in parallel and compare results.

        Args:
            messages: List of messages or a single prompt string
            models: List of model IDs (defaults to one from each provider)
            **kwargs: Additional ModelConfig parameters

        Returns:
            ComparisonResult with all model responses
        """
        if isinstance(messages, str):
            prompt = messages
            messages = [Message.user(messages)]
        else:
            prompt = messages[-1].content if messages else ""
            if not isinstance(prompt, str):
                prompt = " ".join(b.text or "" for b in prompt if b.text)

        # Default to best model from each available provider
        if models is None:
            models = []
            if self.providers.openai:
                models.append("gpt-4o")
            if self.providers.anthropic:
                models.append("claude-sonnet-4-5-20250929")
            if self.providers.google:
                models.append("gemini-3-pro-preview")

        # Query all models in parallel
        tasks = []
        for model in models:
            config = ModelConfig(model=model, **kwargs)
            request = CompletionRequest(messages=messages, config=config)
            tasks.append(self._query_model(model, request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        model_results = []
        for model, result in zip(models, results):
            spec = MODEL_REGISTRY.get(model)
            provider = spec.provider if spec else ModelProvider.OPENAI

            if isinstance(result, Exception):
                model_results.append(ModelResult(
                    model=model,
                    provider=provider,
                    error=str(result),
                    success=False,
                ))
            else:
                metrics = ComparisonMetrics(
                    response_length=len(result.content),
                    latency_ms=result.latency_ms,
                    tokens_per_second=(
                        result.usage.output_tokens / (result.latency_ms / 1000)
                        if result.latency_ms > 0 else 0
                    ),
                    estimated_cost=result.usage.estimated_cost,
                )
                model_results.append(ModelResult(
                    model=model,
                    provider=provider,
                    response=result,
                    metrics=metrics,
                    success=True,
                ))

        return ComparisonResult(
            id=f"cmp-{uuid.uuid4().hex[:16]}",
            prompt=prompt,
            results=model_results,
            created_at=datetime.utcnow(),
        )

    async def route(
        self,
        messages: list[Message] | str,
        routing: RoutingConfig | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | ComparisonResult:
        """
        Route request based on routing strategy.

        Args:
            messages: List of messages or a single prompt string
            routing: Routing configuration
            **kwargs: Additional ModelConfig parameters

        Returns:
            CompletionResponse or ComparisonResult based on strategy
        """
        routing = routing or RoutingConfig()

        if routing.strategy == RoutingStrategy.SINGLE:
            return await self.complete(messages, model=routing.models[0], **kwargs)

        elif routing.strategy == RoutingStrategy.FASTEST:
            # Use fastest model tier
            fast_models = [
                m for m in routing.models
                if MODEL_REGISTRY.get(m) and MODEL_REGISTRY[m].tier == ModelTier.FAST
            ]
            model = fast_models[0] if fast_models else routing.models[0]
            return await self.complete(messages, model=model, **kwargs)

        elif routing.strategy == RoutingStrategy.CHEAPEST:
            # Sort by cost and use cheapest
            sorted_models = sorted(
                routing.models,
                key=lambda m: (
                    MODEL_REGISTRY.get(m).input_cost_per_million
                    if MODEL_REGISTRY.get(m) else float('inf')
                ),
            )
            return await self.complete(messages, model=sorted_models[0], **kwargs)

        elif routing.strategy == RoutingStrategy.BEST_FOR_TASK:
            # Use task-based recommendations
            if routing.task_category:
                recommended = TASK_MODEL_RECOMMENDATIONS.get(
                    routing.task_category, routing.models
                )
                # Find first available recommended model
                for model in recommended:
                    if self.providers.get_provider_for_model(model):
                        return await self.complete(messages, model=model, **kwargs)
            return await self.complete(messages, model=routing.models[0], **kwargs)

        elif routing.strategy == RoutingStrategy.COMPARE_ALL:
            return await self.compare(messages, models=routing.models, **kwargs)

        elif routing.strategy == RoutingStrategy.FALLBACK:
            # Try models in order until one succeeds
            last_error = None
            for model in routing.models:
                try:
                    return await self.complete(messages, model=model, **kwargs)
                except ProviderError as e:
                    last_error = e
                    logger.warning(f"Model {model} failed, trying next", error=str(e))
                    continue
            raise last_error or ValueError("All models failed")

        elif routing.strategy == RoutingStrategy.CONSENSUS:
            # Query multiple models and find consensus
            comparison = await self.compare(messages, models=routing.models, **kwargs)
            # For now, return the comparison - consensus logic can be added
            return comparison

        else:
            raise ValueError(f"Unknown routing strategy: {routing.strategy}")

    async def _query_model(
        self,
        model: str,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Query a single model with error handling."""
        provider = self.providers.get_provider_for_model(model)
        if not provider:
            raise ValueError(f"No provider for model: {model}")
        return await provider.complete_async(request)

    def get_best_model_for_task(
        self,
        task: TaskCategory,
        tier: ModelTier | None = None,
    ) -> str | None:
        """
        Get the best available model for a task category.

        Args:
            task: The task category
            tier: Optional tier constraint

        Returns:
            Model ID or None if no suitable model is available
        """
        recommended = TASK_MODEL_RECOMMENDATIONS.get(task, [])

        for model in recommended:
            spec = MODEL_REGISTRY.get(model)
            if not spec:
                continue
            if tier and spec.tier != tier:
                continue
            if self.providers.get_provider_for_model(model):
                return model

        return None

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all configured providers.

        Returns:
            Dict mapping provider names to health status
        """
        results = {}

        if self.providers.openai:
            results["openai"] = await self.providers.openai.health_check()
        if self.providers.anthropic:
            results["anthropic"] = await self.providers.anthropic.health_check()
        if self.providers.google:
            results["google"] = await self.providers.google.health_check()

        return results

    @property
    def available_models(self) -> list[str]:
        """List all available models based on configured providers."""
        models = []
        for model_id, spec in MODEL_REGISTRY.items():
            if self.providers.get_provider(spec.provider):
                models.append(model_id)
        return models
