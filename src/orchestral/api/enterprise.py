"""
Enterprise API endpoints for Orchestral.

Provides REST API for all premium features:
- Semantic caching
- Prompt management
- A/B testing
- Smart routing
- Guardrails
- Evaluation
- Audit logs
- Observability
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Depends, Query, Header
from pydantic import BaseModel, Field

from orchestral.billing.api_keys import APIKey, get_api_key_manager


async def get_current_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> APIKey | None:
    """
    Extract and validate API key from request header.

    Returns None if no key provided (allows global tenant fallback).
    In production, this should be stricter.
    """
    if not x_api_key:
        return None

    manager = get_api_key_manager()
    api_key = manager.validate_key(x_api_key)  # validate_key is synchronous
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


from orchestral.prompts.manager import (
    Prompt,
    PromptManager,
    PromptTemplate,
    get_prompt_manager,
    PromptStatus,
)
from orchestral.prompts.ab_testing import (
    Experiment,
    ExperimentStatus,
    ABTestingManager,
    MetricType,
    get_ab_manager,
)
from orchestral.safety.guardrails import (
    GuardrailPipeline,
    GuardrailAction,
    get_guardrail_pipeline,
)
from orchestral.optimization.router import (
    SmartRouter,
    OptimizationStrategy,
    get_smart_router,
)
from orchestral.evaluation.evaluator import (
    EvaluationPipeline,
    get_evaluation_pipeline,
)
from orchestral.observability.audit import (
    AuditLogger,
    AuditAction,
    get_audit_logger,
)
from orchestral.observability.tracing import get_tracer, SpanKind
from orchestral.observability.events import EventType, get_event_emitter

import structlog

logger = structlog.get_logger()

# Create enterprise router
enterprise_router = APIRouter(prefix="/enterprise", tags=["Enterprise"])


# ============================================================================
# Request/Response Models
# ============================================================================

class PromptCreateRequest(BaseModel):
    """Request to create a prompt."""

    name: str = Field(..., description="Prompt name")
    content: str = Field(..., description="Prompt template content")
    description: str | None = Field(None, description="Description")
    tags: list[str] = Field(default_factory=list, description="Tags")


class PromptVersionRequest(BaseModel):
    """Request to add a prompt version."""

    content: str = Field(..., description="New version content")
    description: str | None = Field(None, description="Version description")
    activate: bool = Field(False, description="Activate immediately")


class PromptRenderRequest(BaseModel):
    """Request to render a prompt."""

    prompt_id: str = Field(..., description="Prompt ID")
    variables: dict[str, Any] = Field(..., description="Template variables")
    version: int | None = Field(None, description="Specific version")


class ExperimentCreateRequest(BaseModel):
    """Request to create an experiment."""

    name: str = Field(..., description="Experiment name")
    description: str | None = Field(None, description="Description")
    variants: list[dict[str, Any]] = Field(..., description="Variant configurations")
    target_sample_size: int = Field(1000, description="Target sample size")
    primary_metric: str = Field("success_rate", description="Primary metric")


class SmartRouteRequest(BaseModel):
    """Request for smart model routing."""

    prompt: str = Field(..., description="Input prompt")
    task_category: str | None = Field(None, description="Task category hint")
    strategy: str = Field("balanced", description="Optimization strategy")
    max_cost_usd: float | None = Field(None, description="Max cost constraint")
    max_latency_ms: float | None = Field(None, description="Max latency constraint")
    min_quality: float | None = Field(None, description="Min quality requirement")


class GuardrailCheckRequest(BaseModel):
    """Request to check content against guardrails."""

    content: str = Field(..., description="Content to check")
    check_type: str = Field("input", description="input or output")


class EvaluateRequest(BaseModel):
    """Request to evaluate a response."""

    prompt: str = Field(..., description="Original prompt")
    response: str = Field(..., description="Response to evaluate")


class AuditQueryRequest(BaseModel):
    """Request to query audit logs."""

    action: str | None = Field(None, description="Filter by action")
    start_time: datetime | None = Field(None, description="Start time")
    end_time: datetime | None = Field(None, description="End time")
    actor_id: str | None = Field(None, description="Filter by actor")
    limit: int = Field(100, ge=1, le=1000, description="Max results")


# ============================================================================
# Prompt Management Endpoints
# ============================================================================

@enterprise_router.post("/prompts")
async def create_prompt(
    request: PromptCreateRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Create a new managed prompt."""
    manager = get_prompt_manager()
    tenant_id = api_key.key_id if api_key else "global"

    prompt = await manager.create_prompt(
        name=request.name,
        content=request.content,
        tenant_id=tenant_id,
        description=request.description,
        tags=request.tags,
    )

    # Audit log
    audit = get_audit_logger()
    await audit.log(
        action=AuditAction.CONFIG_CHANGED,
        actor_type="api_key",
        actor_id=tenant_id,
        resource_type="prompt",
        resource_id=prompt.prompt_id,
        tenant_id=tenant_id,
        details={"action": "created", "name": request.name},
    )

    return {
        "prompt_id": prompt.prompt_id,
        "name": prompt.name,
        "active_version": prompt.active_version,
        "created_at": prompt.created_at.isoformat(),
    }


@enterprise_router.get("/prompts")
async def list_prompts(
    tags: str | None = Query(None, description="Filter by tags (comma-separated)"),
    limit: int = Query(100, ge=1, le=1000),
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """List prompts for the tenant."""
    manager = get_prompt_manager()
    tenant_id = api_key.key_id if api_key else "global"

    tag_list = tags.split(",") if tags else None
    prompts = await manager.list_prompts(tenant_id=tenant_id, tags=tag_list, limit=limit)

    return {
        "prompts": [
            {
                "prompt_id": p.prompt_id,
                "name": p.name,
                "description": p.description,
                "active_version": p.active_version,
                "version_count": len(p.versions),
                "tags": p.tags,
            }
            for p in prompts
        ],
    }


@enterprise_router.get("/prompts/{prompt_id}")
async def get_prompt(
    prompt_id: str,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Get a prompt by ID."""
    manager = get_prompt_manager()
    prompt = await manager.get_prompt(prompt_id)

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return prompt.to_dict()


@enterprise_router.post("/prompts/{prompt_id}/versions")
async def add_prompt_version(
    prompt_id: str,
    request: PromptVersionRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Add a new version to a prompt."""
    manager = get_prompt_manager()

    try:
        version = await manager.add_version(
            prompt_id=prompt_id,
            content=request.content,
            description=request.description,
            activate=request.activate,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "version_id": version.version_id,
        "version": version.version,
        "status": version.status.value,
        "variables": version.variables,
    }


@enterprise_router.post("/prompts/{prompt_id}/activate/{version}")
async def activate_prompt_version(
    prompt_id: str,
    version: int,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Activate a specific prompt version."""
    manager = get_prompt_manager()

    try:
        pv = await manager.activate_version(prompt_id, version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "version_id": pv.version_id,
        "version": pv.version,
        "status": pv.status.value,
        "activated": True,
    }


@enterprise_router.post("/prompts/render")
async def render_prompt(
    request: PromptRenderRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Render a prompt template with variables."""
    manager = get_prompt_manager()

    try:
        rendered = await manager.render(
            prompt_id=request.prompt_id,
            variables=request.variables,
            version=request.version,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "rendered": rendered,
        "prompt_id": request.prompt_id,
        "version": request.version,
    }


# ============================================================================
# A/B Testing Endpoints
# ============================================================================

@enterprise_router.post("/experiments")
async def create_experiment(
    request: ExperimentCreateRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Create a new A/B test experiment."""
    ab_manager = get_ab_manager()
    tenant_id = api_key.key_id if api_key else "global"

    try:
        metric = MetricType(request.primary_metric)
    except ValueError:
        metric = MetricType.SUCCESS_RATE

    experiment = await ab_manager.create_experiment(
        name=request.name,
        description=request.description,
        variants=request.variants,
        tenant_id=tenant_id,
        target_sample_size=request.target_sample_size,
        primary_metric=metric,
    )

    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "status": experiment.status.value,
        "variants": [v.variant_id for v in experiment.variants],
    }


@enterprise_router.get("/experiments")
async def list_experiments(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """List experiments for the tenant."""
    ab_manager = get_ab_manager()
    tenant_id = api_key.key_id if api_key else "global"

    exp_status = None
    if status:
        try:
            exp_status = ExperimentStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid values: {[s.value for s in ExperimentStatus]}"
            ) from None
    experiments = await ab_manager.list_experiments(
        tenant_id=tenant_id,
        status=exp_status,
        limit=limit,
    )

    return {
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "status": e.status.value,
                "total_impressions": e.total_impressions,
                "variants": len(e.variants),
            }
            for e in experiments
        ],
    }


@enterprise_router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Start an experiment."""
    ab_manager = get_ab_manager()

    try:
        experiment = await ab_manager.start_experiment(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "experiment_id": experiment.experiment_id,
        "status": experiment.status.value,
        "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
    }


@enterprise_router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Stop an experiment and analyze results."""
    ab_manager = get_ab_manager()

    try:
        experiment = await ab_manager.stop_experiment(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = experiment.result
    return {
        "experiment_id": experiment.experiment_id,
        "status": experiment.status.value,
        "result": result.to_dict() if result else None,
    }


@enterprise_router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Get experiment details."""
    ab_manager = get_ab_manager()

    experiment = await ab_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiment.to_dict()


@enterprise_router.get("/experiments/{experiment_id}/variant")
async def get_experiment_variant(
    experiment_id: str,
    user_id: str = Query(..., description="User ID for assignment"),
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Get the assigned variant for a user."""
    ab_manager = get_ab_manager()

    variant = await ab_manager.get_variant_for_user(experiment_id, user_id)
    if not variant:
        raise HTTPException(status_code=404, detail="Experiment not found or not running")

    return {
        "variant_id": variant.variant_id,
        "name": variant.name,
        "config": variant.config,
        "model": variant.model,
        "prompt_id": variant.prompt_id,
    }


# ============================================================================
# Smart Routing Endpoints
# ============================================================================

@enterprise_router.post("/route")
async def smart_route(
    request: SmartRouteRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Get optimal model routing recommendation."""
    router = get_smart_router()

    try:
        strategy = OptimizationStrategy(request.strategy)
    except ValueError:
        strategy = OptimizationStrategy.BALANCED

    selected_model, scores = await router.select_model(
        prompt=request.prompt,
        task_category=request.task_category,
        strategy=strategy,
        max_cost_usd=request.max_cost_usd,
        max_latency_ms=request.max_latency_ms,
        min_quality=request.min_quality,
    )

    return {
        "selected_model": selected_model,
        "strategy": strategy.value,
        "scores": [s.to_dict() for s in scores[:5]],  # Top 5
    }


@enterprise_router.get("/route/stats")
async def get_routing_stats(
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Get model routing statistics."""
    router = get_smart_router()
    stats = await router.get_routing_stats()
    return {"models": stats}


# ============================================================================
# Guardrails Endpoints
# ============================================================================

@enterprise_router.post("/guardrails/check")
async def check_guardrails(
    request: GuardrailCheckRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Check content against guardrails."""
    pipeline = get_guardrail_pipeline()
    tenant_id = api_key.key_id if api_key else "global"

    if request.check_type == "input":
        passed, modified_content, results = await pipeline.check_input(
            request.content,
            context={"tenant_id": tenant_id},
        )
    else:
        passed, modified_content, results = await pipeline.check_output(
            request.content,
            context={"tenant_id": tenant_id},
        )

    # Emit event if blocked
    if not passed:
        emitter = get_event_emitter()
        await emitter.emit(
            EventType.GUARDRAIL_TRIGGERED,
            data={
                "check_type": request.check_type,
                "results": [r.to_dict() for r in results],
            },
            tenant_id=tenant_id,
        )

    return {
        "passed": passed,
        "modified_content": modified_content if modified_content != request.content else None,
        "results": [r.to_dict() for r in results],
    }


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@enterprise_router.post("/evaluate")
async def evaluate_response(
    request: EvaluateRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Evaluate a model response for quality."""
    pipeline = get_evaluation_pipeline()
    tenant_id = api_key.key_id if api_key else "global"

    evaluation = await pipeline.evaluate(
        prompt=request.prompt,
        response=request.response,
        context={"tenant_id": tenant_id},
    )

    return {
        "overall_score": evaluation.overall_score,
        "passed": evaluation.passed,
        "results": [r.to_dict() for r in evaluation.results],
    }


# ============================================================================
# Audit Log Endpoints
# ============================================================================

@enterprise_router.post("/audit/query")
async def query_audit_logs(
    request: AuditQueryRequest,
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Query audit logs (admin only)."""
    audit = get_audit_logger()
    tenant_id = api_key.key_id if api_key else "global"

    action = None
    if request.action:
        try:
            action = AuditAction(request.action)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Valid values: {[a.value for a in AuditAction]}"
            ) from None

    entries = await audit.query(
        tenant_id=tenant_id,
        action=action,
        start_time=request.start_time,
        end_time=request.end_time,
        actor_id=request.actor_id,
        limit=request.limit,
    )

    return {
        "entries": [e.to_dict() for e in entries],
        "count": len(entries),
    }


@enterprise_router.get("/audit/export")
async def export_audit_logs(
    start_time: datetime = Query(..., description="Export start time"),
    end_time: datetime = Query(..., description="Export end time"),
    format: str = Query("json", description="Export format (json, csv)"),
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """Export audit logs for compliance."""
    audit = get_audit_logger()
    tenant_id = api_key.key_id if api_key else "global"

    export_data = await audit.export(
        tenant_id=tenant_id,
        start_time=start_time,
        end_time=end_time,
        format=format,
    )

    return {
        "data": export_data,
        "format": format,
        "tenant_id": tenant_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }


# ============================================================================
# Observability Endpoints
# ============================================================================

@enterprise_router.get("/traces")
async def list_traces(
    limit: int = Query(50, ge=1, le=500),
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """List recent traces (requires Redis)."""
    # This would query Redis for stored traces
    # For now, return placeholder
    return {
        "traces": [],
        "message": "Trace querying requires Redis storage",
    }


@enterprise_router.get("/features")
async def list_features(
    api_key: APIKey | None = Depends(get_current_api_key),
) -> dict[str, Any]:
    """List all enterprise features and their status."""
    return {
        "features": {
            "semantic_caching": {
                "enabled": True,
                "description": "Embeddings-based response caching for similar prompts",
            },
            "prompt_management": {
                "enabled": True,
                "description": "Versioned prompt templates with A/B testing",
            },
            "ab_testing": {
                "enabled": True,
                "description": "Statistical A/B testing for prompts and models",
            },
            "smart_routing": {
                "enabled": True,
                "description": "ML-based model selection optimizing cost/quality/latency",
            },
            "guardrails": {
                "enabled": True,
                "description": "Input/output filtering, PII detection, prompt injection prevention",
            },
            "evaluation": {
                "enabled": True,
                "description": "Automated quality scoring for model outputs",
            },
            "audit_logging": {
                "enabled": True,
                "description": "Compliance-grade audit trails with integrity verification",
            },
            "distributed_tracing": {
                "enabled": True,
                "description": "OpenTelemetry-compatible request tracing",
            },
            "webhooks": {
                "enabled": True,
                "description": "Real-time event notifications with HMAC signing",
            },
        },
    }
