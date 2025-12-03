#!/usr/bin/env python3
"""
Basic usage examples for Orchestral.

This file demonstrates the fundamental operations you can perform
with the Orchestral AI orchestration platform.
"""

import asyncio
from orchestral import Orchestrator, Message
from orchestral.core.models import RoutingConfig, RoutingStrategy, TaskCategory


async def simple_completion():
    """Send a simple prompt to a model."""
    print("\n=== Simple Completion ===\n")

    orch = Orchestrator()

    # Using a string prompt
    response = await orch.complete(
        "What are the three laws of robotics?",
        model="gpt-4o",
        temperature=0.7,
    )

    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Tokens: {response.usage.total_tokens}")


async def conversation():
    """Have a multi-turn conversation."""
    print("\n=== Multi-turn Conversation ===\n")

    orch = Orchestrator()

    messages = [
        Message.system("You are a helpful assistant that explains things simply."),
        Message.user("What is machine learning?"),
    ]

    # First response
    response = await orch.complete(messages, model="claude-sonnet-4-5-20250929")
    print(f"Assistant: {response.content}\n")

    # Continue the conversation
    messages.append(Message.assistant(response.content))
    messages.append(Message.user("Can you give me a simple example?"))

    response = await orch.complete(messages, model="claude-sonnet-4-5-20250929")
    print(f"Assistant: {response.content}")


async def compare_models():
    """Compare responses from multiple models."""
    print("\n=== Model Comparison ===\n")

    orch = Orchestrator()

    comparison = await orch.compare(
        "Write a one-paragraph explanation of blockchain technology.",
        models=["gpt-4o", "claude-sonnet-4-5-20250929", "gemini-3-pro-preview"],
        temperature=0.7,
        max_tokens=300,
    )

    print(f"Prompt: {comparison.prompt}\n")

    for result in comparison.results:
        if result.success:
            print(f"--- {result.model} ({result.provider.value}) ---")
            print(f"{result.response.content[:200]}...")
            print(f"Latency: {result.metrics.latency_ms:.0f}ms")
            print(f"Cost: ${result.metrics.estimated_cost:.4f}\n")
        else:
            print(f"--- {result.model} (FAILED) ---")
            print(f"Error: {result.error}\n")


async def intelligent_routing():
    """Route requests to the best model for the task."""
    print("\n=== Intelligent Routing ===\n")

    orch = Orchestrator()

    # Route for coding task
    routing = RoutingConfig(
        strategy=RoutingStrategy.BEST_FOR_TASK,
        task_category=TaskCategory.CODING,
    )

    response = await orch.route(
        "Write a Python function to check if a number is prime.",
        routing=routing,
    )

    print(f"Task: Coding")
    print(f"Selected Model: {response.model}")
    print(f"Response:\n{response.content}")


async def streaming():
    """Stream a response token by token."""
    print("\n=== Streaming Response ===\n")

    orch = Orchestrator()

    print("Response: ", end="", flush=True)
    async for chunk in orch.stream(
        "Tell me a short joke about programming.",
        model="gpt-4o",
    ):
        print(chunk, end="", flush=True)
    print("\n")


async def cost_optimized():
    """Use the cheapest model for simple tasks."""
    print("\n=== Cost Optimization ===\n")

    orch = Orchestrator()

    routing = RoutingConfig(
        strategy=RoutingStrategy.CHEAPEST,
        models=["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini-3-flash"],
    )

    response = await orch.route(
        "What is 2 + 2?",
        routing=routing,
    )

    print(f"Used cheapest model: {response.model}")
    print(f"Response: {response.content}")


async def fallback_strategy():
    """Try multiple models with fallback on failure."""
    print("\n=== Fallback Strategy ===\n")

    orch = Orchestrator()

    routing = RoutingConfig(
        strategy=RoutingStrategy.FALLBACK,
        models=["gpt-4o", "claude-sonnet-4-5-20250929", "gemini-3-pro-preview"],
    )

    response = await orch.route(
        "Summarize the key features of Python 3.12",
        routing=routing,
    )

    print(f"Successfully used: {response.model}")
    print(f"Response: {response.content[:200]}...")


async def main():
    """Run all examples."""
    try:
        await simple_completion()
    except Exception as e:
        print(f"Simple completion failed: {e}")

    try:
        await conversation()
    except Exception as e:
        print(f"Conversation failed: {e}")

    try:
        await compare_models()
    except Exception as e:
        print(f"Comparison failed: {e}")

    try:
        await intelligent_routing()
    except Exception as e:
        print(f"Routing failed: {e}")

    try:
        await streaming()
    except Exception as e:
        print(f"Streaming failed: {e}")

    try:
        await cost_optimized()
    except Exception as e:
        print(f"Cost optimization failed: {e}")

    try:
        await fallback_strategy()
    except Exception as e:
        print(f"Fallback failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
