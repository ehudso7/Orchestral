"""
FastAPI server for Orchestral.

Provides REST API endpoints for multi-model AI orchestration.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from orchestral.core.orchestrator import Orchestrator
from orchestral.core.models import (
    Message,
    MessageRole,
    ModelConfig,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
    MODEL_REGISTRY,
    CompletionResponse,
    ComparisonResult,
)
from orchestral.core.config import get_settings, Settings
from orchestral.utils.logging import setup_logging
from orchestral.utils.metrics import metrics

logger = structlog.get_logger()

# Global orchestrator instance
orchestrator: Orchestrator | None = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    Limits requests per IP address within a time window.
    For production, consider using Redis-based rate limiting.
    """

    def __init__(self, app: FastAPI, requests_limit: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (for reverse proxy setups)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Clean old requests
            self._requests[client_ip] = [
                ts for ts in self._requests[client_ip] if ts > window_start
            ]

            # Check limit
            if len(self._requests[client_ip]) >= self.requests_limit:
                return True

            # Record this request
            self._requests[client_ip].append(now)
            return False

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if self._is_rate_limited(client_ip):
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={"Retry-After": str(self.window_seconds)},
            )

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    global orchestrator
    setup_logging()
    logger.info("Starting Orchestral API server")
    orchestrator = Orchestrator()
    yield
    logger.info("Shutting down Orchestral API server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Orchestral API",
        description="Multi-model AI orchestration platform for ChatGPT, Claude, and Gemini",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting middleware (applied first, so it runs last)
    app.add_middleware(
        RateLimitMiddleware,
        requests_limit=settings.server.rate_limit_requests,
        window_seconds=settings.server.rate_limit_window_seconds,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# Request/Response Models

class MessageRequest(BaseModel):
    """A message in a conversation."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class CompletionRequest(BaseModel):
    """Request for a completion."""

    messages: list[MessageRequest] = Field(..., description="Conversation messages")
    model: str = Field(default="gpt-4o", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    stream: bool = Field(default=False, description="Enable streaming")


class SimplePromptRequest(BaseModel):
    """Simple prompt request."""

    prompt: str = Field(..., description="The prompt to send")
    model: str = Field(default="gpt-4o", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class CompareRequest(BaseModel):
    """Request to compare multiple models."""

    prompt: str = Field(..., description="Prompt to compare across models")
    models: list[str] | None = Field(default=None, description="Models to compare")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class RouteRequest(BaseModel):
    """Request for intelligent routing."""

    prompt: str = Field(..., description="Prompt to route")
    strategy: str = Field(default="best", description="Routing strategy")
    task_category: str | None = Field(default=None, description="Task category for routing")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class CompletionResponseModel(BaseModel):
    """Completion response."""

    id: str
    model: str
    provider: str
    content: str
    finish_reason: str | None = None
    usage: dict[str, int]
    latency_ms: float


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    provider: str
    tier: str
    context_window: int
    max_output_tokens: int
    supports_vision: bool
    supports_audio: bool
    supports_video: bool
    input_cost_per_million: float
    output_cost_per_million: float
    display_name: str


# Dependencies

def get_orchestrator() -> Orchestrator:
    """Get the orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


async def verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> bool:
    """Verify API key if authentication is enabled."""
    settings = get_settings()
    if not settings.server.require_auth:
        return True

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    # Use constant-time comparison to prevent timing attacks
    # Compare against ALL keys without short-circuiting to avoid leaking
    # timing information about which key position matches
    # Use bitwise OR to eliminate conditional branching (branch prediction side-channel)
    is_valid = False
    for key in settings.server.api_keys:
        is_valid |= secrets.compare_digest(x_api_key, key)

    if not is_valid:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


# Routes

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "Orchestral API",
        "version": "1.0.0",
        "description": "Multi-model AI orchestration platform",
    }


@app.get("/health")
async def health_check(
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """Health check endpoint."""
    provider_health = await orch.health_check()
    return {
        "status": "healthy" if any(provider_health.values()) else "degraded",
        "providers": provider_health,
    }


@app.get("/models", response_model=list[ModelInfo])
async def list_models(
    orch: Orchestrator = Depends(get_orchestrator),
    _: bool = Depends(verify_api_key),
) -> list[ModelInfo]:
    """List available models."""
    available = orch.available_models
    models = []

    for model_id in available:
        spec = MODEL_REGISTRY.get(model_id)
        if spec:
            models.append(ModelInfo(
                id=model_id,
                provider=spec.provider.value,
                tier=spec.tier.value,
                context_window=spec.context_window,
                max_output_tokens=spec.max_output_tokens,
                supports_vision=spec.supports_vision,
                supports_audio=spec.supports_audio,
                supports_video=spec.supports_video,
                input_cost_per_million=spec.input_cost_per_million,
                output_cost_per_million=spec.output_cost_per_million,
                display_name=spec.display_name,
            ))

    return models


@app.post("/v1/completions", response_model=CompletionResponseModel)
async def create_completion(
    request: CompletionRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    _: bool = Depends(verify_api_key),
) -> CompletionResponseModel | StreamingResponse:
    """Create a completion from a model."""
    start_time = time.perf_counter()

    # Convert messages
    messages = [
        Message(role=MessageRole(m.role), content=m.content)
        for m in request.messages
    ]

    try:
        if request.stream:
            async def generate():
                async for chunk in orch.stream(
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    yield {"data": chunk}

            return EventSourceResponse(generate())

        response = await orch.complete(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        metrics.record_completion(
            provider=response.provider.value,
            model=request.model,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return CompletionResponseModel(
            id=response.id,
            model=response.model,
            provider=response.provider.value,
            content=response.content,
            finish_reason=response.finish_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            latency_ms=response.latency_ms,
        )

    except Exception as e:
        logger.exception("Completion failed", model=request.model)
        metrics.record_error(request.model, str(type(e).__name__))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/simple")
async def simple_completion(
    request: SimplePromptRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    _: bool = Depends(verify_api_key),
) -> CompletionResponseModel:
    """Simple completion with just a prompt string."""
    try:
        response = await orch.complete(
            messages=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return CompletionResponseModel(
            id=response.id,
            model=response.model,
            provider=response.provider.value,
            content=response.content,
            finish_reason=response.finish_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            latency_ms=response.latency_ms,
        )

    except Exception as e:
        logger.exception("Simple completion failed", model=request.model)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/compare")
async def compare_models(
    request: CompareRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    _: bool = Depends(verify_api_key),
) -> dict[str, Any]:
    """Compare responses from multiple models."""
    try:
        comparison = await orch.compare(
            messages=request.prompt,
            models=request.models,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        results = []
        for r in comparison.results:
            result_data: dict[str, Any] = {
                "model": r.model,
                "provider": r.provider.value,
                "success": r.success,
            }
            if r.success and r.response:
                result_data["content"] = r.response.content
                result_data["metrics"] = {
                    "latency_ms": r.metrics.latency_ms,
                    "response_length": r.metrics.response_length,
                    "tokens_per_second": r.metrics.tokens_per_second,
                    "estimated_cost": r.metrics.estimated_cost,
                }
            else:
                result_data["error"] = r.error
            results.append(result_data)

        return {
            "comparison_id": comparison.id,
            "prompt": comparison.prompt,
            "results": results,
            "successful_count": len(comparison.successful_results),
            "failed_count": len(comparison.failed_results),
        }

    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/route")
async def route_request(
    request: RouteRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    _: bool = Depends(verify_api_key),
) -> dict[str, Any]:
    """Route request to optimal model based on strategy."""
    strategy_map = {
        "single": RoutingStrategy.SINGLE,
        "fastest": RoutingStrategy.FASTEST,
        "cheapest": RoutingStrategy.CHEAPEST,
        "best": RoutingStrategy.BEST_FOR_TASK,
        "compare": RoutingStrategy.COMPARE_ALL,
        "fallback": RoutingStrategy.FALLBACK,
        "consensus": RoutingStrategy.CONSENSUS,
    }

    category_map = {
        "coding": TaskCategory.CODING,
        "reasoning": TaskCategory.REASONING,
        "creative": TaskCategory.CREATIVE,
        "analysis": TaskCategory.ANALYSIS,
        "multimodal": TaskCategory.MULTIMODAL,
        "conversation": TaskCategory.CONVERSATION,
        "summarization": TaskCategory.SUMMARIZATION,
        "translation": TaskCategory.TRANSLATION,
    }

    try:
        routing = RoutingConfig(
            strategy=strategy_map.get(request.strategy, RoutingStrategy.BEST_FOR_TASK),
            task_category=category_map.get(request.task_category) if request.task_category else None,
        )

        result = await orch.route(
            messages=request.prompt,
            routing=routing,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        if isinstance(result, CompletionResponse):
            return {
                "type": "completion",
                "id": result.id,
                "model": result.model,
                "provider": result.provider.value,
                "content": result.content,
                "latency_ms": result.latency_ms,
            }
        else:
            return {
                "type": "comparison",
                "comparison_id": result.id,
                "results_count": len(result.results),
                "successful_count": len(result.successful_results),
            }

    except Exception as e:
        logger.exception("Routing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/recommend")
async def recommend_model(
    task: str,
    orch: Orchestrator = Depends(get_orchestrator),
    _: bool = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get model recommendation for a task category."""
    category_map = {
        "coding": TaskCategory.CODING,
        "reasoning": TaskCategory.REASONING,
        "creative": TaskCategory.CREATIVE,
        "analysis": TaskCategory.ANALYSIS,
        "multimodal": TaskCategory.MULTIMODAL,
        "conversation": TaskCategory.CONVERSATION,
        "summarization": TaskCategory.SUMMARIZATION,
        "translation": TaskCategory.TRANSLATION,
    }

    category = category_map.get(task.lower())
    if not category:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task category: {task}. Valid options: {list(category_map.keys())}",
        )

    recommended = orch.get_best_model_for_task(category)

    if not recommended:
        raise HTTPException(
            status_code=404,
            detail="No suitable model available for this task",
        )

    spec = MODEL_REGISTRY.get(recommended)
    return {
        "task": task,
        "recommended_model": recommended,
        "display_name": spec.display_name if spec else recommended,
        "provider": spec.provider.value if spec else "unknown",
        "tier": spec.tier.value if spec else "unknown",
    }


@app.get("/metrics")
async def get_metrics(
    _: bool = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get application metrics. Requires authentication."""
    return metrics.get_summary()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "orchestral.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
