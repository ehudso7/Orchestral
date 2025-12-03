"""Integration tests for the FastAPI server."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from orchestral.api.server import app, get_orchestrator
from orchestral.core.orchestrator import Orchestrator
from orchestral.core.models import (
    CompletionResponse,
    ModelProvider,
    UsageStats,
)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orch = Orchestrator(
        openai_api_key="test-key",
        auto_configure=False,
    )
    return orch


@pytest.fixture
def mock_response():
    """Create a mock completion response."""
    return CompletionResponse(
        id="test-123",
        model="gpt-4o",
        provider=ModelProvider.OPENAI,
        content="Test response",
        finish_reason="stop",
        usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
        latency_ms=100.0,
    )


@pytest.fixture
def client(mock_orchestrator, mock_response):
    """Create test client with mocked orchestrator."""
    def override_get_orchestrator():
        return mock_orchestrator

    app.dependency_overrides[get_orchestrator] = override_get_orchestrator

    with patch.object(
        mock_orchestrator,
        "complete",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with TestClient(app) as client:
            yield client

    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Orchestral API"


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestCompletionEndpoint:
    """Tests for completion endpoint."""

    def test_simple_completion(self, client, mock_orchestrator, mock_response):
        with patch.object(
            mock_orchestrator,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/v1/simple",
                json={
                    "prompt": "Hello!",
                    "model": "gpt-4o",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Test response"
            assert data["model"] == "gpt-4o"


class TestCompareEndpoint:
    """Tests for comparison endpoint."""

    def test_compare_models(self, client, mock_orchestrator):
        from orchestral.core.models import ComparisonResult, ModelResult

        mock_comparison = ComparisonResult(
            id="cmp-123",
            prompt="Test",
            results=[
                ModelResult(
                    model="gpt-4o",
                    provider=ModelProvider.OPENAI,
                    response=CompletionResponse(
                        id="1",
                        model="gpt-4o",
                        provider=ModelProvider.OPENAI,
                        content="Response",
                    ),
                    success=True,
                )
            ],
        )

        with patch.object(
            mock_orchestrator,
            "compare",
            new_callable=AsyncMock,
            return_value=mock_comparison,
        ):
            response = client.post(
                "/v1/compare",
                json={
                    "prompt": "Test prompt",
                    "models": ["gpt-4o"],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "comparison_id" in data
            assert len(data["results"]) == 1


class TestRouteEndpoint:
    """Tests for routing endpoint."""

    def test_route_request(self, client, mock_orchestrator, mock_response):
        with patch.object(
            mock_orchestrator,
            "route",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/v1/route",
                json={
                    "prompt": "Test prompt",
                    "strategy": "best",
                    "task_category": "coding",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "completion"


class TestRecommendEndpoint:
    """Tests for model recommendation endpoint."""

    def test_recommend_model(self, client, mock_orchestrator):
        with patch.object(
            mock_orchestrator,
            "get_best_model_for_task",
            return_value="claude-opus-4-5-20251101",
        ):
            response = client.get("/v1/recommend?task=coding")

            assert response.status_code == 200
            data = response.json()
            assert data["task"] == "coding"
            assert "recommended_model" in data

    def test_recommend_invalid_task(self, client):
        response = client.get("/v1/recommend?task=invalid")
        assert response.status_code == 400
