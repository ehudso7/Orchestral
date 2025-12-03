"""
Rich CLI interface for Orchestral.

Provides a beautiful command-line interface for AI model orchestration.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.live import Live

from orchestral import __version__
from orchestral.core.orchestrator import Orchestrator
from orchestral.core.models import (
    MODEL_REGISTRY,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
)
from orchestral.core.config import get_settings

app = typer.Typer(
    name="orchestral",
    help="AI Model Orchestration Platform - Unified interface for ChatGPT, Claude, and Gemini",
    no_args_is_help=True,
)
console = Console()


def get_orchestrator() -> Orchestrator:
    """Get orchestrator instance."""
    return Orchestrator()


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold cyan]Orchestral[/bold cyan] v{__version__}")


@app.command()
def models():
    """List all available AI models."""
    orch = get_orchestrator()
    available = orch.available_models

    table = Table(title="Available AI Models", show_header=True, header_style="bold magenta")
    table.add_column("Model ID", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Tier", style="blue")
    table.add_column("Context", justify="right")
    table.add_column("Cost (in/out)", justify="right")

    for model_id in sorted(available):
        spec = MODEL_REGISTRY.get(model_id)
        if spec:
            table.add_row(
                model_id,
                spec.display_name,
                spec.provider.value,
                spec.tier.value,
                f"{spec.context_window:,}",
                f"${spec.input_cost_per_million:.2f}/${spec.output_cost_per_million:.2f}",
            )

    console.print(table)
    console.print(f"\n[dim]Total: {len(available)} models available[/dim]")


@app.command()
def providers():
    """Show configured providers and their status."""
    orch = get_orchestrator()

    table = Table(title="Provider Status", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Models", justify="right")

    providers_list = ["openai", "anthropic", "google"]
    for provider_name in providers_list:
        is_configured = provider_name in [p.value for p in orch.providers.available_providers]
        status = "[green]Configured[/green]" if is_configured else "[red]Not configured[/red]"

        model_count = len([
            m for m, s in MODEL_REGISTRY.items()
            if s.provider.value == provider_name and m in orch.available_models
        ])

        table.add_row(provider_name.upper(), status, str(model_count) if is_configured else "-")

    console.print(table)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The prompt to send"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model to use"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature (0.0-2.0)"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Maximum tokens"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
):
    """Ask a question to an AI model."""
    orch = get_orchestrator()

    if model not in orch.available_models:
        console.print(f"[red]Model '{model}' is not available.[/red]")
        console.print(f"Available models: {', '.join(orch.available_models)}")
        raise typer.Exit(1)

    spec = MODEL_REGISTRY.get(model)

    async def run():
        if stream:
            console.print(f"\n[bold cyan]{spec.display_name}:[/bold cyan]")
            async for chunk in orch.stream(prompt, model=model, temperature=temperature, max_tokens=max_tokens):
                console.print(chunk, end="")
            console.print("\n")
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"Querying {spec.display_name}...", total=None)
                response = await orch.complete(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            console.print(Panel(
                Markdown(response.content),
                title=f"[bold cyan]{spec.display_name}[/bold cyan]",
                subtitle=f"[dim]{response.latency_ms:.0f}ms | {response.usage.total_tokens} tokens[/dim]",
            ))

    asyncio.run(run())


@app.command()
def compare(
    prompt: str = typer.Argument(..., help="The prompt to compare"),
    models_str: str = typer.Option(None, "--models", "-m", help="Comma-separated model IDs"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Maximum tokens"),
):
    """Compare responses from multiple AI models."""
    orch = get_orchestrator()

    model_list = models_str.split(",") if models_str else None

    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Querying all models in parallel...", total=None)
            comparison = await orch.compare(
                prompt,
                models=model_list,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        console.print(f"\n[bold]Comparison ID:[/bold] {comparison.id}\n")

        for result in comparison.results:
            spec = MODEL_REGISTRY.get(result.model)
            display_name = spec.display_name if spec else result.model

            if result.success and result.response:
                console.print(Panel(
                    Markdown(result.response.content),
                    title=f"[bold cyan]{display_name}[/bold cyan] ({result.provider.value})",
                    subtitle=f"[dim]{result.metrics.latency_ms:.0f}ms | "
                             f"${result.metrics.estimated_cost:.4f}[/dim]",
                ))
            else:
                console.print(Panel(
                    f"[red]Error: {result.error}[/red]",
                    title=f"[bold red]{display_name}[/bold red]",
                ))
            console.print()

        # Summary table
        table = Table(title="Performance Summary", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Latency", justify="right")
        table.add_column("Length", justify="right")
        table.add_column("Speed", justify="right")
        table.add_column("Cost", justify="right")

        for result in comparison.successful_results:
            spec = MODEL_REGISTRY.get(result.model)
            table.add_row(
                spec.display_name if spec else result.model,
                f"{result.metrics.latency_ms:.0f}ms",
                f"{result.metrics.response_length:,} chars",
                f"{result.metrics.tokens_per_second:.0f} tok/s",
                f"${result.metrics.estimated_cost:.4f}",
            )

        console.print(table)

    asyncio.run(run())


@app.command()
def route(
    prompt: str = typer.Argument(..., help="The prompt to route"),
    strategy: str = typer.Option("best", "--strategy", "-s",
        help="Routing strategy: single, fastest, cheapest, best, compare, fallback"),
    task: str = typer.Option(None, "--task", "-t",
        help="Task category: coding, reasoning, creative, analysis, multimodal, conversation"),
    temperature: float = typer.Option(0.7, "--temp", help="Temperature"),
):
    """Route request to optimal model based on strategy."""
    orch = get_orchestrator()

    strategy_map = {
        "single": RoutingStrategy.SINGLE,
        "fastest": RoutingStrategy.FASTEST,
        "cheapest": RoutingStrategy.CHEAPEST,
        "best": RoutingStrategy.BEST_FOR_TASK,
        "compare": RoutingStrategy.COMPARE_ALL,
        "fallback": RoutingStrategy.FALLBACK,
    }

    category_map = {
        "coding": TaskCategory.CODING,
        "reasoning": TaskCategory.REASONING,
        "creative": TaskCategory.CREATIVE,
        "analysis": TaskCategory.ANALYSIS,
        "multimodal": TaskCategory.MULTIMODAL,
        "conversation": TaskCategory.CONVERSATION,
    }

    routing = RoutingConfig(
        strategy=strategy_map.get(strategy, RoutingStrategy.BEST_FOR_TASK),
        task_category=category_map.get(task) if task else None,
    )

    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Routing with '{strategy}' strategy...", total=None)
            result = await orch.route(prompt, routing=routing, temperature=temperature)

        if hasattr(result, "content"):
            spec = MODEL_REGISTRY.get(result.model)
            console.print(Panel(
                Markdown(result.content),
                title=f"[bold cyan]{spec.display_name if spec else result.model}[/bold cyan]",
                subtitle=f"[dim]Strategy: {strategy} | {result.latency_ms:.0f}ms[/dim]",
            ))
        else:
            console.print(f"[bold]Comparison result with {len(result.results)} models[/bold]")

    asyncio.run(run())


@app.command()
def chat(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model to use"),
):
    """Start an interactive chat session."""
    from orchestral.core.models import Message

    orch = get_orchestrator()
    spec = MODEL_REGISTRY.get(model)
    display_name = spec.display_name if spec else model

    console.print(Panel(
        f"Starting chat with [bold cyan]{display_name}[/bold cyan]\n"
        "Type 'exit' or 'quit' to end the session.\n"
        "Type '/switch <model>' to change models.\n"
        "Type '/compare' to compare last response with other models.",
        title="Orchestral Chat",
    ))

    messages: list[Message] = []
    current_model = model

    async def run():
        nonlocal current_model, messages

        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.startswith("/switch "):
                new_model = user_input[8:].strip()
                if new_model in orch.available_models:
                    current_model = new_model
                    new_spec = MODEL_REGISTRY.get(new_model)
                    console.print(f"[yellow]Switched to {new_spec.display_name if new_spec else new_model}[/yellow]")
                else:
                    console.print(f"[red]Model not available: {new_model}[/red]")
                continue

            messages.append(Message.user(user_input))

            spec = MODEL_REGISTRY.get(current_model)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(f"Thinking...", total=None)
                response = await orch.complete(messages, model=current_model)

            messages.append(Message.assistant(response.content))

            console.print(f"\n[bold cyan]{spec.display_name if spec else current_model}:[/bold cyan]")
            console.print(Markdown(response.content))
            console.print(f"[dim]{response.latency_ms:.0f}ms | {response.usage.total_tokens} tokens[/dim]")

    asyncio.run(run())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the API server."""
    from orchestral.api.server import run_server

    console.print(Panel(
        f"Starting Orchestral API server\n"
        f"Host: [cyan]{host}[/cyan]\n"
        f"Port: [cyan]{port}[/cyan]\n"
        f"Docs: [link]http://{host}:{port}/docs[/link]",
        title="Orchestral Server",
    ))

    run_server(host=host, port=port, reload=reload)


@app.command()
def mcp():
    """Start the MCP server for Claude Code integration."""
    from orchestral.mcp.server import run_mcp_server

    console.print("[dim]Starting MCP server over stdio...[/dim]", file=sys.stderr)
    asyncio.run(run_mcp_server())


@app.command()
def config():
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Orchestral Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Environment", settings.environment)
    table.add_row("Debug", str(settings.debug))
    table.add_row("Log Level", settings.orchestrator.log_level)
    table.add_row("Default Timeout", f"{settings.orchestrator.default_timeout}s")
    table.add_row("Max Retries", str(settings.orchestrator.max_retries))
    table.add_row("Server Host", settings.server.host)
    table.add_row("Server Port", str(settings.server.port))
    table.add_row("Auth Required", str(settings.server.require_auth))

    console.print(table)

    # Show configured providers
    console.print("\n[bold]Configured Providers:[/bold]")
    for provider in settings.providers.available_providers:
        console.print(f"  [green]✓[/green] {provider}")


@app.command()
def benchmark(
    prompt: str = typer.Argument("Explain quantum computing in simple terms.", help="Prompt to benchmark"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of iterations"),
):
    """Benchmark all available models."""
    orch = get_orchestrator()

    async def run():
        results = []

        for model in orch.available_models:
            spec = MODEL_REGISTRY.get(model)
            latencies = []

            console.print(f"[dim]Benchmarking {spec.display_name if spec else model}...[/dim]")

            for _ in range(iterations):
                try:
                    response = await orch.complete(prompt, model=model, max_tokens=500)
                    latencies.append(response.latency_ms)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                results.append((model, spec, avg_latency, min(latencies), max(latencies)))

        # Display results
        table = Table(title=f"Benchmark Results ({iterations} iterations)", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for model, spec, avg, min_lat, max_lat in sorted(results, key=lambda x: x[2]):
            table.add_row(
                spec.display_name if spec else model,
                f"{avg:.0f}ms",
                f"{min_lat:.0f}ms",
                f"{max_lat:.0f}ms",
            )

        console.print(table)

    asyncio.run(run())


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
