# Orchestral

**AI Model Orchestration Platform** - A unified interface for ChatGPT, Claude, and Gemini with intelligent routing, parallel processing, and comprehensive comparison capabilities.

## Features

- **Multi-Model Support**: Seamlessly integrate ChatGPT 5.1, Claude Opus 4.5, and Gemini 3 Pro
- **Parallel Comparison**: Query multiple models simultaneously and compare responses
- **Intelligent Routing**: Automatically route requests to the best model for each task
- **MCP Integration**: Model Context Protocol support for Claude Code integration
- **REST API**: Production-ready FastAPI server with OpenAPI documentation
- **Rich CLI**: Beautiful command-line interface with streaming support
- **Async First**: Built for high-performance with full async/await support
- **Observability**: Built-in metrics, logging, and monitoring

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ehudso7/Orchestral.git
cd Orchestral

# Install with pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Or create a `.env` file:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

### Python Usage

```python
import asyncio
from orchestral import Orchestrator

async def main():
    # Initialize orchestrator
    orch = Orchestrator()

    # Simple completion
    response = await orch.complete(
        "Explain quantum computing in simple terms",
        model="gpt-4o",
    )
    print(response.content)

    # Compare multiple models
    comparison = await orch.compare(
        "Write a haiku about programming",
        models=["gpt-4o", "claude-sonnet-4-5-20250929", "gemini-3-pro-preview"],
    )

    for result in comparison.results:
        print(f"\n{result.model}:")
        print(result.response.content)
        print(f"Latency: {result.metrics.latency_ms:.0f}ms")

asyncio.run(main())
```

### CLI Usage

```bash
# Ask a question
orchestral ask "What is the capital of France?" --model gpt-4o

# Compare models
orchestral compare "Explain recursion" --models gpt-4o,claude-sonnet-4-5-20250929

# Interactive chat
orchestral chat --model claude-opus-4-5-20251101

# Start API server
orchestral serve --port 8000

# List available models
orchestral models
```

### API Server

Start the server:

```bash
orchestral serve
# Or with uvicorn
uvicorn orchestral.api.server:app --reload
```

Make requests:

```bash
# Simple completion
curl -X POST http://localhost:8000/v1/simple \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "model": "gpt-4o"}'

# Compare models
curl -X POST http://localhost:8000/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain AI", "models": ["gpt-4o", "claude-sonnet-4-5-20250929"]}'

# Intelligent routing
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review this code", "strategy": "best", "task_category": "coding"}'
```

## Supported Models

### OpenAI
| Model | Tier | Context | Strengths |
|-------|------|---------|-----------|
| GPT-5.1 | Flagship | 272K | Adaptive reasoning, complex tasks |
| GPT-4o | Standard | 128K | Balanced performance |
| GPT-4o Mini | Fast | 128K | Cost-effective |

### Anthropic
| Model | Tier | Context | Strengths |
|-------|------|---------|-----------|
| Claude Opus 4.5 | Flagship | 200K | Best coding (80.9% SWE-bench) |
| Claude Sonnet 4.5 | Standard | 200K | Balanced performance |
| Claude Haiku 4.5 | Fast | 200K | Speed-optimized |

### Google
| Model | Tier | Context | Strengths |
|-------|------|---------|-----------|
| Gemini 3 Pro | Flagship | 1M | Best multimodal |
| Gemini 2.5 Pro | Standard | 1M | Long context |
| Gemini 2.5 Flash | Fast | 1M | Speed-optimized |

## Routing Strategies

- **`single`**: Use a specific model
- **`fastest`**: Use the fastest available model
- **`cheapest`**: Use the most cost-effective model
- **`best`**: Route to the best model for the task category
- **`compare`**: Query all specified models and compare
- **`fallback`**: Try models in order until one succeeds
- **`consensus`**: Query multiple models and find agreement

## Task Categories

The intelligent router optimizes model selection for:

- **coding**: Code generation, debugging, refactoring
- **reasoning**: Logic, math, complex analysis
- **creative**: Writing, brainstorming, storytelling
- **analysis**: Document review, summarization
- **multimodal**: Image, video, audio understanding
- **conversation**: Chat, dialogue, Q&A
- **summarization**: Text compression, key points
- **translation**: Language translation

## MCP Integration

Use Orchestral as an MCP server for Claude Code:

```json
{
  "mcpServers": {
    "orchestral": {
      "command": "orchestral",
      "args": ["mcp"]
    }
  }
}
```

Available MCP tools:
- `orchestral_complete`: Generate completion from any model
- `orchestral_compare`: Compare responses across models
- `orchestral_route`: Intelligently route to best model
- `orchestral_analyze`: Multi-model analysis

## Architecture

```
orchestral/
├── core/
│   ├── orchestrator.py   # Main orchestration engine
│   ├── models.py         # Data models and types
│   └── config.py         # Configuration management
├── providers/
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── google_provider.py
├── api/
│   └── server.py         # FastAPI server
├── cli/
│   └── main.py           # Typer CLI
├── mcp/
│   ├── server.py         # MCP server
│   └── tools.py          # MCP tool definitions
└── utils/
    ├── logging.py        # Structured logging
    ├── metrics.py        # Observability
    └── retry.py          # Retry logic
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/orchestral

# Type checking
mypy src/orchestral

# Linting
ruff check src/orchestral
```

## API Reference

See the full API documentation at `/docs` when running the server.

## License

MIT License - see [LICENSE](LICENSE) for details.
