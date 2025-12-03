"""
MCP Server implementation for Orchestral.

Provides a Model Context Protocol server that can be used by Claude Code
and other MCP-compatible clients.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import structlog

from orchestral.core.orchestrator import Orchestrator
from orchestral.core.models import (
    Message,
    RoutingConfig,
    RoutingStrategy,
    TaskCategory,
)
from orchestral.mcp.tools import get_tool_schemas, get_tool_by_name

logger = structlog.get_logger()


class MCPServer:
    """
    Model Context Protocol server for Orchestral.

    Exposes Orchestral capabilities as MCP tools that can be used
    by Claude Code and other MCP clients.
    """

    def __init__(self, orchestrator: Orchestrator | None = None):
        """Initialize the MCP server."""
        self.orchestrator = orchestrator or Orchestrator()
        self._running = False

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                return self._handle_initialize(request_id)
            elif method == "tools/list":
                return self._handle_list_tools(request_id)
            elif method == "tools/call":
                return await self._handle_call_tool(request_id, params)
            elif method == "resources/list":
                return self._handle_list_resources(request_id)
            elif method == "prompts/list":
                return self._handle_list_prompts(request_id)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
        except Exception as e:
            logger.exception("Error handling MCP request", method=method)
            return self._error_response(request_id, -32603, str(e))

    def _handle_initialize(self, request_id: Any) -> dict[str, Any]:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "orchestral",
                    "version": "1.0.0",
                },
            },
        }

    def _handle_list_tools(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": get_tool_schemas(),
            },
        }

    async def _handle_call_tool(
        self,
        request_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool_def = get_tool_by_name(tool_name)
        if not tool_def:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")

        try:
            result = await self._execute_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, default=str),
                        }
                    ],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error executing tool: {str(e)}",
                        }
                    ],
                    "isError": True,
                },
            }

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool and return results."""
        if tool_name == "orchestral_complete":
            response = await self.orchestrator.complete(
                messages=arguments["prompt"],
                model=arguments.get("model", "gpt-4o"),
                temperature=arguments.get("temperature", 0.7),
                max_tokens=arguments.get("max_tokens", 4096),
            )
            return {
                "model": response.model,
                "content": response.content,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "latency_ms": response.latency_ms,
            }

        elif tool_name == "orchestral_compare":
            comparison = await self.orchestrator.compare(
                messages=arguments["prompt"],
                models=arguments.get("models"),
            )
            results = []
            for r in comparison.results:
                result_data = {
                    "model": r.model,
                    "success": r.success,
                }
                if r.success and r.response:
                    result_data["content"] = r.response.content
                    if arguments.get("include_metrics", True):
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
            }

        elif tool_name == "orchestral_route":
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

            routing = RoutingConfig(
                strategy=strategy_map.get(
                    arguments.get("strategy", "best"),
                    RoutingStrategy.BEST_FOR_TASK,
                ),
                task_category=category_map.get(arguments.get("task_category")),
            )

            result = await self.orchestrator.route(
                messages=arguments["prompt"],
                routing=routing,
            )

            # Handle both single response and comparison results
            if hasattr(result, "content"):
                return {
                    "type": "completion",
                    "model": result.model,
                    "content": result.content,
                    "latency_ms": result.latency_ms,
                }
            else:
                return {
                    "type": "comparison",
                    "comparison_id": result.id,
                    "results_count": len(result.results),
                }

        elif tool_name == "orchestral_analyze":
            # Multi-model analysis with synthesis
            content = arguments["content"]
            analysis_type = arguments.get("analysis_type", "summary")
            depth = arguments.get("depth", "standard")

            # Build analysis prompt
            prompts = {
                "summary": f"Provide a {depth} summary of the following content:\n\n{content}",
                "sentiment": f"Analyze the sentiment of the following content ({depth} analysis):\n\n{content}",
                "entities": f"Extract key entities from the following content ({depth} analysis):\n\n{content}",
                "code_review": f"Review the following code ({depth} review):\n\n{content}",
                "fact_check": f"Fact-check the following content ({depth} verification):\n\n{content}",
                "comparison": f"Compare and contrast elements in the following content:\n\n{content}",
            }

            prompt = prompts.get(analysis_type, prompts["summary"])

            # Use best model for analysis
            response = await self.orchestrator.complete(
                messages=prompt,
                model="claude-opus-4-5-20251101",  # Best for analysis
            )

            return {
                "analysis_type": analysis_type,
                "depth": depth,
                "model": response.model,
                "analysis": response.content,
            }

        raise ValueError(f"Unknown tool: {tool_name}")

    def _handle_list_resources(self, request_id: Any) -> dict[str, Any]:
        """Handle resources/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": [
                    {
                        "uri": "orchestral://models",
                        "name": "Available Models",
                        "description": "List of available AI models",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "orchestral://health",
                        "name": "Health Status",
                        "description": "Health status of all providers",
                        "mimeType": "application/json",
                    },
                ],
            },
        }

    def _handle_list_prompts(self, request_id: Any) -> dict[str, Any]:
        """Handle prompts/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "prompts": [
                    {
                        "name": "compare_models",
                        "description": "Compare how different AI models respond to a prompt",
                        "arguments": [
                            {
                                "name": "prompt",
                                "description": "The prompt to compare across models",
                                "required": True,
                            },
                        ],
                    },
                    {
                        "name": "best_for_task",
                        "description": "Get recommendation for the best model for a specific task",
                        "arguments": [
                            {
                                "name": "task",
                                "description": "The task category (coding, reasoning, creative, etc.)",
                                "required": True,
                            },
                        ],
                    },
                ],
            },
        }

    def _error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    async def run_stdio(self) -> None:
        """Run the MCP server over stdio."""
        self._running = True
        logger.info("Starting MCP server over stdio")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        while self._running:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode())
                response = await self.handle_request(request)

                response_json = json.dumps(response) + "\n"
                writer.write(response_json.encode())
                await writer.drain()

            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
            except Exception as e:
                logger.exception("Error in MCP server loop")

        logger.info("MCP server stopped")

    def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False


async def run_mcp_server() -> None:
    """Entry point for running the MCP server."""
    server = MCPServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
