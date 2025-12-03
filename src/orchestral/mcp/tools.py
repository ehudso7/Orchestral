"""
MCP Tools for Orchestral.

Provides tools that can be used by Claude and other MCP-compatible clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""

    name: str
    description: str
    parameters: list[ToolParameter]

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


# Define Orchestral MCP Tools

OrchestraTool = ToolDefinition(
    name="orchestral_complete",
    description="Generate a completion from a specified AI model (ChatGPT, Claude, or Gemini)",
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt or question to send to the model",
        ),
        ToolParameter(
            name="model",
            type="string",
            description="The model to use for completion",
            required=False,
            enum=[
                "gpt-5.1", "gpt-4o", "gpt-4o-mini",
                "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
                "gemini-3-ultra", "gemini-3-pro-preview", "gemini-3-flash",
            ],
            default="gpt-4o",
        ),
        ToolParameter(
            name="temperature",
            type="number",
            description="Sampling temperature (0.0 to 2.0)",
            required=False,
            default=0.7,
        ),
        ToolParameter(
            name="max_tokens",
            type="integer",
            description="Maximum tokens in response",
            required=False,
            default=4096,
        ),
    ],
)


ComparisonTool = ToolDefinition(
    name="orchestral_compare",
    description="Compare responses from multiple AI models on the same prompt",
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt to send to all models",
        ),
        ToolParameter(
            name="models",
            type="array",
            description="List of models to compare (defaults to one from each provider)",
            required=False,
        ),
        ToolParameter(
            name="include_metrics",
            type="boolean",
            description="Include performance metrics (latency, tokens, cost)",
            required=False,
            default=True,
        ),
    ],
)


RoutingTool = ToolDefinition(
    name="orchestral_route",
    description="Intelligently route a request to the best model for the task",
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt or task to execute",
        ),
        ToolParameter(
            name="strategy",
            type="string",
            description="Routing strategy to use",
            required=False,
            enum=["single", "fastest", "cheapest", "best", "compare", "fallback", "consensus"],
            default="best",
        ),
        ToolParameter(
            name="task_category",
            type="string",
            description="Category of the task for optimal routing",
            required=False,
            enum=["coding", "reasoning", "creative", "analysis", "multimodal", "conversation", "summarization", "translation"],
        ),
    ],
)


AnalysisTool = ToolDefinition(
    name="orchestral_analyze",
    description="Analyze content using multiple AI models and synthesize insights",
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="The content to analyze",
        ),
        ToolParameter(
            name="analysis_type",
            type="string",
            description="Type of analysis to perform",
            required=False,
            enum=["sentiment", "summary", "entities", "code_review", "fact_check", "comparison"],
            default="summary",
        ),
        ToolParameter(
            name="depth",
            type="string",
            description="Depth of analysis",
            required=False,
            enum=["quick", "standard", "deep"],
            default="standard",
        ),
    ],
)


ALL_TOOLS = [OrchestraTool, ComparisonTool, RoutingTool, AnalysisTool]


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get JSON schemas for all tools."""
    return [tool.to_schema() for tool in ALL_TOOLS]


def get_tool_by_name(name: str) -> ToolDefinition | None:
    """Get a tool definition by name."""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None
