"""
FastAPI server for Orchestral.

Provides REST API endpoints for multi-model AI orchestration with
commercial-grade billing, rate limiting, and usage tracking.
"""

from __future__ import annotations

import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from orchestral.core.orchestrator import Orchestrator
from orchestral.core.models import (
    Message,
    MessageRole,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
    MODEL_REGISTRY,
    CompletionResponse,
)
from orchestral.core.config import get_settings
from orchestral.utils.logging import setup_logging
from orchestral.utils.metrics import metrics
from orchestral.billing.api_keys import APIKeyManager, APIKey, KeyTier, TIER_LIMITS
from orchestral.billing.rate_limiter import RateLimiter
from orchestral.billing.usage import UsageTracker
from orchestral.billing.cache import ResponseCache

logger = structlog.get_logger()

# Global instances
orchestrator: Orchestrator | None = None
api_key_manager: APIKeyManager | None = None
rate_limiter: RateLimiter | None = None
usage_tracker: UsageTracker | None = None
response_cache: ResponseCache | None = None
redis_client: Any = None


def get_redis_client() -> Any:
    """Get or create Redis client."""
    global redis_client
    if redis_client is not None:
        return redis_client

    settings = get_settings()
    if not settings.redis.is_configured:
        return None

    try:
        import redis

        if settings.redis.url:
            redis_client = redis.from_url(
                settings.redis.url,
                decode_responses=False,
            )
        else:
            redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password.get_secret_value() if settings.redis.password else None,
                db=settings.redis.db,
                ssl=settings.redis.ssl,
                socket_timeout=settings.redis.socket_timeout,
                max_connections=settings.redis.max_connections,
            )

        # Test connection
        redis_client.ping()
        logger.info("Redis connected successfully")
        return redis_client

    except Exception as e:
        logger.warning("Redis connection failed, using in-memory fallback", error=str(e))
        return None


class CommercialRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Commercial-grade rate limiting middleware.

    Uses Redis for distributed rate limiting with per-API-key limits.
    Falls back to in-memory when Redis is unavailable.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for non-API routes
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        settings = get_settings()

        # Get API key from header
        api_key_header = request.headers.get("x-api-key", "")

        # Skip if rate limiter not initialized
        if not rate_limiter:
            return await call_next(request)

        if settings.server.rate_limit_by_key and api_key_header and api_key_manager:
            # Rate limit by API key
            api_key = api_key_manager.validate_key(api_key_header)
            if api_key:
                limits = api_key.limits
                result = await rate_limiter.check(
                    identifier=api_key.key_id,
                    limit=limits.requests_per_minute,
                    window=60,
                )

                if not result.allowed:
                    logger.warning(
                        "Rate limit exceeded for API key",
                        key_id=api_key.key_id,
                        tier=api_key.tier.value,
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded",
                            "tier": api_key.tier.value,
                            "limit": result.limit,
                            "retry_after": result.retry_after,
                        },
                        headers=result.headers,
                    )

                # Add rate limit headers to response
                response = await call_next(request)
                for key, value in result.headers.items():
                    response.headers[key] = value
                return response

        # Fallback to IP-based rate limiting
        forwarded = request.headers.get("x-forwarded-for")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else "unknown"
        )

        result = await rate_limiter.check(
            identifier=f"ip:{client_ip}",
            limit=settings.server.rate_limit_requests,
            window=settings.server.rate_limit_window_seconds,
        )

        if not result.allowed:
            logger.warning("Rate limit exceeded for IP", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers=result.headers,
            )

        response = await call_next(request)
        for key, value in result.headers.items():
            response.headers[key] = value
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    global orchestrator, api_key_manager, rate_limiter, usage_tracker, response_cache

    setup_logging()
    logger.info("Starting Orchestral API server (Commercial Edition)")

    settings = get_settings()
    redis = get_redis_client()

    # Initialize services
    orchestrator = Orchestrator()

    # Get secret key from config if available
    secret_key = None
    if settings.billing.api_key_secret:
        secret_key = bytes.fromhex(settings.billing.api_key_secret.get_secret_value())

    api_key_manager = APIKeyManager(redis_client=redis, secret_key=secret_key)
    rate_limiter = RateLimiter(
        redis_client=redis,
        default_limit=settings.server.rate_limit_requests,
        default_window=settings.server.rate_limit_window_seconds,
    )
    usage_tracker = UsageTracker(redis_client=redis)
    response_cache = ResponseCache(
        redis_client=redis,
        default_ttl_seconds=settings.billing.cache_ttl_seconds,
        max_entries=settings.billing.cache_max_entries,
        enabled=settings.billing.cache_enabled,
    )

    logger.info(
        "Commercial services initialized",
        redis_connected=redis is not None,
        cache_enabled=settings.billing.cache_enabled,
    )

    yield

    logger.info("Shutting down Orchestral API server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Orchestral API",
        description="Commercial Multi-Model AI Orchestration Platform",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Commercial rate limiting middleware
    app.add_middleware(CommercialRateLimitMiddleware)

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
    use_cache: bool = Field(default=True, description="Use response cache")


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
    cached: bool = False
    cost_usd: float = 0.0


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


class APIKeyCreate(BaseModel):
    """Request to create an API key."""
    name: str = Field(..., description="Key name")
    tier: str = Field(default="starter", description="Key tier")
    owner_id: str = Field(..., description="Owner identifier")
    expires_in_days: int | None = Field(default=None, description="Days until expiration")
    monthly_budget_usd: float | None = Field(default=None, description="Monthly budget override")


class APIKeyResponse(BaseModel):
    """API key response."""
    key_id: str
    name: str
    tier: str
    owner_id: str
    created_at: str
    expires_at: str | None
    is_active: bool
    limits: dict[str, Any]


# Dependencies

def get_orchestrator() -> Orchestrator:
    """Get the orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


async def verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> APIKey | None:
    """Verify API key and return key metadata."""
    settings = get_settings()

    if not settings.server.require_auth:
        return None

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    # Check managed API keys first
    if api_key_manager:
        api_key = api_key_manager.validate_key(x_api_key)
        if api_key:
            return api_key

    # Fall back to static API keys
    is_valid = False
    for key in settings.server.api_keys:
        is_valid |= secrets.compare_digest(x_api_key, key)

    if not is_valid:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return None


async def verify_admin_key(
    x_admin_key: str | None = Header(default=None),
) -> bool:
    """Verify admin API key."""
    settings = get_settings()

    if not settings.server.admin_api_enabled:
        raise HTTPException(status_code=404, detail="Admin API not enabled")

    if not settings.server.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API key not configured")

    if not x_admin_key:
        raise HTTPException(status_code=401, detail="Admin API key required")

    if not secrets.compare_digest(
        x_admin_key,
        settings.server.admin_api_key.get_secret_value(),
    ):
        raise HTTPException(status_code=403, detail="Invalid admin API key")

    return True


async def check_budget(api_key: APIKey | None) -> None:
    """Check if API key is within budget."""
    if not api_key or not usage_tracker:
        return

    settings = get_settings()
    if not settings.billing.hard_budget_limit:
        return

    within_budget, current_spend, _remaining = await usage_tracker.check_budget(
        api_key.key_id,
        api_key.effective_monthly_budget,
    )

    if not within_budget:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly budget exceeded: ${current_spend:.2f} of ${api_key.effective_monthly_budget:.2f} used",
        )


# Routes

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "Orchestral API",
        "version": "2.0.0",
        "edition": "Commercial",
        "description": "Multi-model AI orchestration platform",
    }


@app.get("/health")
async def health_check(
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """Health check endpoint."""
    provider_health = await orch.health_check()
    redis_healthy = redis_client is not None

    cache_stats = None
    if response_cache:
        cache_stats = (await response_cache.get_stats()).to_dict()

    return {
        "status": "healthy" if any(provider_health.values()) else "degraded",
        "providers": provider_health,
        "redis": redis_healthy,
        "cache": cache_stats,
    }


@app.get("/models", response_model=list[ModelInfo])
async def list_models(
    orch: Orchestrator = Depends(get_orchestrator),
    api_key: APIKey | None = Depends(verify_api_key),
) -> list[ModelInfo]:
    """List available models."""
    available = orch.available_models

    # Filter by tier if using managed key
    if api_key and api_key.limits.allowed_models:
        available = [m for m in available if m in api_key.limits.allowed_models]

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
    api_key: APIKey | None = Depends(verify_api_key),
) -> CompletionResponseModel:
    """Create a completion from a model."""
    await check_budget(api_key)

    # Check model access for tier
    if api_key and api_key.limits.allowed_models:
        if request.model not in api_key.limits.allowed_models:
            raise HTTPException(
                status_code=403,
                detail=f"Model {request.model} not available on {api_key.tier.value} tier",
            )

    request_id = f"req-{uuid.uuid4().hex[:16]}"
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

        cost_usd = usage_tracker.calculate_cost(
            request.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        ) if usage_tracker else 0.0

        # Record usage
        if usage_tracker and api_key:
            await usage_tracker.record(
                key_id=api_key.key_id,
                model=request.model,
                provider=response.provider.value,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=response.latency_ms,
                request_id=request_id,
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
            cost_usd=cost_usd,
        )

    except Exception as e:
        logger.exception("Completion failed", model=request.model)
        if usage_tracker and api_key:
            await usage_tracker.record(
                key_id=api_key.key_id,
                model=request.model,
                provider="unknown",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                request_id=request_id,
                success=False,
                error=str(e),
            )
        metrics.record_error(request.model, str(type(e).__name__))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/simple")
async def simple_completion(
    request: SimplePromptRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    api_key: APIKey | None = Depends(verify_api_key),
) -> CompletionResponseModel:
    """Simple completion with just a prompt string."""
    await check_budget(api_key)
    request_id = f"req-{uuid.uuid4().hex[:16]}"

    # Try cache first (tenant-isolated)
    tenant_id = api_key.key_id if api_key else None
    if request.use_cache and response_cache:
        cached = await response_cache.get(
            request.prompt,
            request.model,
            request.temperature,
            request.max_tokens,
            tenant_id=tenant_id,
        )
        if cached:
            # Use actual token counts from cache for accurate cost savings
            usage_data = cached.response_data.get("usage", {})
            input_tokens = usage_data.get("input_tokens", 100)  # Fallback for old entries
            output_tokens = usage_data.get("output_tokens", len(cached.response_content) // 4)
            cost_saved = usage_tracker.calculate_cost(
                request.model,
                input_tokens,
                output_tokens,
            ) if usage_tracker else 0.0
            await response_cache.record_cost_saved(cost_saved)

            # Ensure total_tokens is present for consistency with non-cached responses
            usage_response = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage_data.get("total_tokens", input_tokens + output_tokens),
            }

            return CompletionResponseModel(
                id=f"cached-{cached.cache_key[:16]}",
                model=request.model,
                provider=cached.response_data.get("provider", "cached"),
                content=cached.response_content,
                finish_reason=cached.response_data.get("finish_reason", "stop"),
                usage=usage_response,
                latency_ms=0,
                cached=True,
                cost_usd=0.0,
            )

    try:
        response = await orch.complete(
            messages=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        cost_usd = usage_tracker.calculate_cost(
            request.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        ) if usage_tracker else 0.0

        # Cache the response (tenant-isolated)
        if request.use_cache and response_cache:
            await response_cache.set(
                request.prompt,
                request.model,
                response.content,
                {
                    "provider": response.provider.value,
                    "finish_reason": response.finish_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                },
                request.temperature,
                request.max_tokens,
                cost_usd=cost_usd,
                tenant_id=tenant_id,
            )

        # Record usage
        if usage_tracker and api_key:
            await usage_tracker.record(
                key_id=api_key.key_id,
                model=request.model,
                provider=response.provider.value,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=response.latency_ms,
                request_id=request_id,
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
            cost_usd=cost_usd,
        )

    except Exception as e:
        logger.exception("Simple completion failed", model=request.model)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/compare")
async def compare_models(
    request: CompareRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    api_key: APIKey | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Compare responses from multiple models."""
    await check_budget(api_key)

    try:
        comparison = await orch.compare(
            messages=request.prompt,
            models=request.models,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        results = []
        total_cost = 0.0
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
                total_cost += r.metrics.estimated_cost
            else:
                result_data["error"] = r.error
            results.append(result_data)

        return {
            "comparison_id": comparison.id,
            "prompt": comparison.prompt,
            "results": results,
            "successful_count": len(comparison.successful_results),
            "failed_count": len(comparison.failed_results),
            "total_cost_usd": total_cost,
        }

    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/route")
async def route_request(
    request: RouteRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    api_key: APIKey | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Route request to optimal model based on strategy."""
    await check_budget(api_key)

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
    _api_key: APIKey | None = Depends(verify_api_key),  # Auth required but key not used
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


# Usage and Billing Endpoints

@app.get("/v1/usage")
async def get_usage(
    api_key: APIKey | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get usage for the current API key."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required for usage data")

    if not usage_tracker:
        raise HTTPException(status_code=503, detail="Usage tracking not available")

    daily = await usage_tracker.get_daily_summary(api_key.key_id)
    monthly = await usage_tracker.get_monthly_summary(api_key.key_id)

    return {
        "key_id": api_key.key_id,
        "tier": api_key.tier.value,
        "limits": {
            "requests_per_minute": api_key.limits.requests_per_minute,
            "requests_per_day": api_key.limits.requests_per_day,
            "tokens_per_month": api_key.limits.tokens_per_month,
            "monthly_budget_usd": api_key.effective_monthly_budget,
        },
        "daily": daily.to_dict(),
        "monthly": monthly.to_dict(),
        "budget_remaining_usd": api_key.effective_monthly_budget - monthly.total_cost_usd,
    }


@app.get("/v1/usage/history")
async def get_usage_history(
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return"),
    api_key: APIKey | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get recent usage history."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")

    if not usage_tracker:
        raise HTTPException(status_code=503, detail="Usage tracking not available")

    records = await usage_tracker.get_usage(api_key.key_id, limit=limit)

    return {
        "key_id": api_key.key_id,
        "records": [r.to_dict() for r in records],
    }


# Admin Endpoints

@app.post("/admin/keys", response_model=dict[str, Any])
async def create_api_key(
    request: APIKeyCreate,
    _: bool = Depends(verify_admin_key),
) -> dict[str, Any]:
    """Create a new API key (admin only)."""
    if not api_key_manager:
        raise HTTPException(status_code=503, detail="API key management not available")

    tier_map = {
        "free": KeyTier.FREE,
        "starter": KeyTier.STARTER,
        "pro": KeyTier.PRO,
        "enterprise": KeyTier.ENTERPRISE,
    }

    tier = tier_map.get(request.tier.lower())
    if not tier:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier: {request.tier}. Valid options: {list(tier_map.keys())}",
        )

    raw_key, api_key = api_key_manager.generate_key(
        name=request.name,
        tier=tier,
        owner_id=request.owner_id,
        expires_in_days=request.expires_in_days,
        monthly_budget_usd=request.monthly_budget_usd,
    )

    return {
        "key": raw_key,  # Only returned once!
        "warning": "This is the only time the API key will be displayed. Copy and store it securely.",
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier.value,
        "owner_id": api_key.owner_id,
        "created_at": api_key.created_at.isoformat(),
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "limits": {
            "requests_per_minute": api_key.limits.requests_per_minute,
            "requests_per_day": api_key.limits.requests_per_day,
            "tokens_per_month": api_key.limits.tokens_per_month,
            "monthly_budget_usd": api_key.effective_monthly_budget,
        },
    }


@app.get("/admin/keys")
async def list_api_keys(
    owner_id: str | None = None,
    _: bool = Depends(verify_admin_key),
) -> dict[str, Any]:
    """List all API keys (admin only)."""
    if not api_key_manager:
        raise HTTPException(status_code=503, detail="API key management not available")

    keys = api_key_manager.list_keys(owner_id=owner_id)

    return {
        "keys": [
            {
                "key_id": k.key_id,
                "name": k.name,
                "tier": k.tier.value,
                "owner_id": k.owner_id,
                "is_active": k.is_active,
                "created_at": k.created_at.isoformat(),
                "total_requests": k.total_requests,
                "total_cost_usd": k.total_cost_usd,
            }
            for k in keys
        ],
    }


@app.delete("/admin/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    _: bool = Depends(verify_admin_key),
) -> dict[str, Any]:
    """Revoke an API key (admin only)."""
    if not api_key_manager:
        raise HTTPException(status_code=503, detail="API key management not available")

    if api_key_manager.revoke_key(key_id):
        return {"status": "revoked", "key_id": key_id}
    else:
        raise HTTPException(status_code=404, detail="API key not found")


@app.get("/admin/tiers")
async def list_tiers(
    _: bool = Depends(verify_admin_key),
) -> dict[str, Any]:
    """List available tiers and limits (admin only)."""
    return {
        "tiers": {
            tier.value: {
                "requests_per_minute": limits.requests_per_minute,
                "requests_per_day": limits.requests_per_day,
                "tokens_per_month": limits.tokens_per_month,
                "max_concurrent_requests": limits.max_concurrent_requests,
                "monthly_budget_usd": limits.monthly_budget_usd,
                "allowed_models": limits.allowed_models,
                "priority": limits.priority,
            }
            for tier, limits in TIER_LIMITS.items()
        }
    }


@app.get("/metrics")
async def get_metrics(
    _api_key: APIKey | None = Depends(verify_api_key),  # Auth required but key not used
) -> dict[str, Any]:
    """Get application metrics."""
    summary = metrics.get_summary()

    if response_cache:
        summary["cache"] = (await response_cache.get_stats()).to_dict()

    return summary


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
