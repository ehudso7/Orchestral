#!/usr/bin/env python3
"""
Advanced usage examples for Orchestral.

This file demonstrates advanced patterns for production use
including async parallelism, custom routing, and monitoring.
"""

import asyncio
from datetime import datetime
from typing import Any

from orchestral import Orchestrator, Message
from orchestral.core.models import (
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
    MODEL_REGISTRY,
    ModelTier,
)
from orchestral.utils.metrics import metrics
from orchestral.utils.retry import RetryConfig, retry_async
from orchestral.providers.base import RateLimitError


async def parallel_processing():
    """Process multiple prompts in parallel."""
    print("\n=== Parallel Processing ===\n")

    orch = Orchestrator()

    prompts = [
        "Explain Python decorators",
        "What is a closure in JavaScript?",
        "How does garbage collection work?",
        "What is the difference between a process and a thread?",
        "Explain the concept of polymorphism",
    ]

    # Process all prompts in parallel
    start_time = datetime.now()

    tasks = [
        orch.complete(prompt, model="gpt-4o-mini", max_tokens=200)
        for prompt in prompts
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"Processed {len(prompts)} prompts in {elapsed:.2f}s\n")

    for prompt, response in zip(prompts, responses):
        if isinstance(response, Exception):
            print(f"Q: {prompt[:40]}... -> ERROR: {response}")
        else:
            print(f"Q: {prompt[:40]}... -> {len(response.content)} chars")


async def batch_comparison():
    """Compare models on a batch of prompts."""
    print("\n=== Batch Comparison ===\n")

    orch = Orchestrator()

    test_cases = [
        ("Code generation", "Write a Python function to reverse a string"),
        ("Creative writing", "Write a haiku about coffee"),
        ("Reasoning", "If A > B and B > C, what is the relationship between A and C?"),
        ("Analysis", "What are the pros and cons of microservices?"),
    ]

    results: dict[str, dict[str, Any]] = {}

    for category, prompt in test_cases:
        print(f"\nTesting: {category}")
        comparison = await orch.compare(
            prompt,
            models=["gpt-4o", "claude-sonnet-4-5-20250929"],
            max_tokens=200,
        )

        results[category] = {
            "prompt": prompt,
            "responses": {
                r.model: {
                    "length": len(r.response.content) if r.success else 0,
                    "latency": r.metrics.latency_ms,
                    "success": r.success,
                }
                for r in comparison.results
            }
        }

    # Print summary
    print("\n--- Summary ---")
    for category, data in results.items():
        print(f"\n{category}:")
        for model, stats in data["responses"].items():
            status = "OK" if stats["success"] else "FAIL"
            print(f"  {model}: {stats['length']} chars, {stats['latency']:.0f}ms [{status}]")


async def custom_routing_logic():
    """Implement custom routing based on prompt characteristics."""
    print("\n=== Custom Routing ===\n")

    orch = Orchestrator()

    def analyze_prompt(prompt: str) -> tuple[str, TaskCategory]:
        """Analyze prompt to determine best model and task category."""
        prompt_lower = prompt.lower()

        # Code-related keywords
        code_keywords = ["code", "function", "debug", "implement", "python", "javascript"]
        if any(kw in prompt_lower for kw in code_keywords):
            return "claude-opus-4-5-20251101", TaskCategory.CODING

        # Creative keywords
        creative_keywords = ["write", "story", "poem", "creative", "imagine"]
        if any(kw in prompt_lower for kw in creative_keywords):
            return "gpt-5.1", TaskCategory.CREATIVE

        # Analysis keywords
        analysis_keywords = ["analyze", "compare", "summarize", "review"]
        if any(kw in prompt_lower for kw in analysis_keywords):
            return "claude-opus-4-5-20251101", TaskCategory.ANALYSIS

        # Default
        return "gpt-4o", TaskCategory.CONVERSATION

    prompts = [
        "Write a Python function to sort a list",
        "Write a short poem about autumn",
        "Analyze the impact of AI on healthcare",
        "What's the weather like today?",
    ]

    for prompt in prompts:
        model, category = analyze_prompt(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Selected: {model} (category: {category.value})")

        # Use fallback in case the selected model isn't available
        routing = RoutingConfig(
            strategy=RoutingStrategy.FALLBACK,
            models=[model, "gpt-4o"],  # Fallback to GPT-4o
            task_category=category,
        )

        try:
            response = await orch.route(prompt, routing=routing, max_tokens=150)
            print(f"Response: {response.content[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


async def with_retries():
    """Handle failures with retry logic."""
    print("\n=== Retry Handling ===\n")

    orch = Orchestrator()

    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        exponential_base=2.0,
        retryable_exceptions=(RateLimitError, ConnectionError),
    )

    async def make_request(prompt: str) -> str:
        response = await orch.complete(prompt, model="gpt-4o", max_tokens=100)
        return response.content

    try:
        result = await retry_async(
            make_request,
            "Explain the theory of relativity briefly",
            config=retry_config,
        )
        print(f"Success: {result[:100]}...")
    except Exception as e:
        print(f"Failed after retries: {e}")


async def monitoring_example():
    """Demonstrate metrics and monitoring."""
    print("\n=== Monitoring ===\n")

    orch = Orchestrator()

    # Reset metrics for this example
    metrics.reset()

    # Make several requests
    prompts = [
        "Hello",
        "What is Python?",
        "Explain machine learning",
    ]

    for prompt in prompts:
        try:
            response = await orch.complete(prompt, model="gpt-4o-mini", max_tokens=50)
            metrics.record_completion(
                provider="openai",
                model="gpt-4o-mini",
                latency_ms=response.latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except Exception as e:
            metrics.record_error("gpt-4o-mini", type(e).__name__, "openai")

    # Get metrics summary
    summary = metrics.get_summary()

    print("Metrics Summary:")
    print(f"  Total requests: {summary['total_requests']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total tokens: {summary['total_tokens']}")

    if "openai" in summary["providers"]:
        provider_stats = summary["providers"]["openai"]
        print(f"\nOpenAI Provider:")
        print(f"  Requests: {provider_stats['total_requests']}")
        print(f"  Avg latency: {provider_stats['avg_latency_ms']:.0f}ms")


async def model_selection_by_budget():
    """Select models based on cost budget."""
    print("\n=== Budget-Based Selection ===\n")

    def get_models_under_budget(
        max_cost_per_million_input: float,
        tier: ModelTier | None = None,
    ) -> list[str]:
        """Get models under a cost threshold."""
        models = []
        for model_id, spec in MODEL_REGISTRY.items():
            if spec.input_cost_per_million <= max_cost_per_million_input:
                if tier is None or spec.tier == tier:
                    models.append(model_id)
        return sorted(models, key=lambda m: MODEL_REGISTRY[m].input_cost_per_million)

    # Find models under $2/million tokens
    affordable = get_models_under_budget(2.0)
    print(f"Models under $2/M input tokens: {affordable}")

    # Find fast models under $1/million
    fast_affordable = get_models_under_budget(1.0, ModelTier.FAST)
    print(f"Fast models under $1/M input: {fast_affordable}")


async def multimodal_preparation():
    """Prepare for multimodal requests (image/video analysis)."""
    print("\n=== Multimodal Capabilities ===\n")

    # Show which models support what
    print("Model Capabilities:\n")
    print(f"{'Model':<35} {'Vision':<8} {'Audio':<8} {'Video':<8}")
    print("-" * 60)

    for model_id, spec in MODEL_REGISTRY.items():
        vision = "Yes" if spec.supports_vision else "-"
        audio = "Yes" if spec.supports_audio else "-"
        video = "Yes" if spec.supports_video else "-"
        print(f"{spec.display_name:<35} {vision:<8} {audio:<8} {video:<8}")

    print("\nFor true multimodal (all three), use Gemini 3 models.")


async def main():
    """Run all advanced examples."""
    examples = [
        ("Parallel Processing", parallel_processing),
        ("Batch Comparison", batch_comparison),
        ("Custom Routing", custom_routing_logic),
        ("Retry Handling", with_retries),
        ("Monitoring", monitoring_example),
        ("Budget Selection", model_selection_by_budget),
        ("Multimodal Prep", multimodal_preparation),
    ]

    for name, func in examples:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print("=" * 60)
        try:
            await func()
        except Exception as e:
            print(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
