"""Vetch MCP server — stdio transport.

Registers all Vetch tools and resources for consumption by MCP clients.

Sync tool handlers are dispatched via ``asyncio.to_thread`` to avoid
blocking the event loop (important for ``vetch_compare`` which iterates
over multiple models and for any tool that hits the grid API).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, Resource, TextContent, Tool
except ImportError as exc:
    raise ImportError(
        "The 'mcp' package is required for the Vetch MCP server. "
        "Install it with: pip install vetch[mcp]"
    ) from exc

from vetch.mcp.resources import (
    get_config,
    get_energy_data,
    get_pricing_data,
    get_version,
    list_models,
)
from vetch.mcp.tools import (
    vetch_check_budget,
    vetch_cleanest_region,
    vetch_compare,
    vetch_estimate,
    vetch_grid_intensity,
    vetch_registry_lookup,
    vetch_session_stats,
    vetch_status,
)

app = Server("vetch")


# ── Tools ──────────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="vetch_estimate",
        description=(
            "Estimate energy (Wh), carbon (gCO2e), water (mL), and cost (USD) "
            "for a single LLM inference call. Includes confidence level and "
            "training emissions context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": (
                        "Model name (e.g. 'claude-sonnet-4-20250514')"
                    ),
                },
                "input_tokens": {
                    "type": "integer",
                    "description": "Number of input tokens",
                },
                "output_tokens": {
                    "type": "integer",
                    "description": "Number of output tokens",
                },
                "region": {
                    "type": "string",
                    "description": (
                        "Grid region code (e.g. 'US-CAL-CISO'). "
                        "Uses VETCH_REGION env if omitted."
                    ),
                },
            },
            "required": ["model", "input_tokens", "output_tokens"],
        },
    ),
    Tool(
        name="vetch_compare",
        description=(
            "Compare energy, carbon, water, and cost across multiple models "
            "for the same workload. Marks cheapest and greenest options."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to compare",
                },
                "input_tokens": {
                    "type": "integer",
                    "description": "Number of input tokens",
                },
                "output_tokens": {
                    "type": "integer",
                    "description": "Number of output tokens",
                },
                "region": {
                    "type": "string",
                    "description": "Grid region code",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["cost_usd", "carbon_g", "energy_wh", "water_ml"],
                    "description": "Sort by this field (default: cost_usd)",
                },
            },
            "required": ["models", "input_tokens", "output_tokens"],
        },
    ),
    Tool(
        name="vetch_session_stats",
        description=(
            "Return real-time session statistics: total tokens, energy, "
            "carbon, cost, advisories (caching, RAG, stall detection), "
            "and training context."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="vetch_status",
        description=(
            "Return Vetch health status, version, and budget information."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="vetch_check_budget",
        description=(
            "Check remaining budget for cost, energy, and carbon. "
            "Returns threshold, accumulated, remaining, and percentage "
            "used for each configured budget."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="vetch_grid_intensity",
        description="Return the current grid carbon intensity for a region.",
        inputSchema={
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": (
                        "Grid region code (e.g. 'US-CAL-CISO')"
                    ),
                },
            },
            "required": ["region"],
        },
    ),
    Tool(
        name="vetch_cleanest_region",
        description=(
            "Find the lowest-carbon region from a list of candidates."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of region codes to compare",
                },
            },
            "required": ["regions"],
        },
    ),
    Tool(
        name="vetch_registry_lookup",
        description=(
            "Look up energy coefficients and pricing data for a model "
            "from the Vetch registry."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model name to look up",
                },
            },
            "required": ["model"],
        },
    ),
]


RESOURCES = [
    Resource(
        uri="vetch://registry/models",
        name="Model list",
        description="All model names in the energy registry",
    ),
    Resource(
        uri="vetch://config",
        name="Configuration",
        description="Current Vetch configuration",
    ),
    Resource(
        uri="vetch://version",
        name="Version",
        description="Vetch version string",
    ),
]


# ── Handlers ───────────────────────────────────────────────────────────────

_TOOL_HANDLERS: dict[str, Any] = {
    "vetch_estimate": vetch_estimate,
    "vetch_compare": vetch_compare,
    "vetch_session_stats": vetch_session_stats,
    "vetch_status": vetch_status,
    "vetch_check_budget": vetch_check_budget,
    "vetch_grid_intensity": vetch_grid_intensity,
    "vetch_cleanest_region": vetch_cleanest_region,
    "vetch_registry_lookup": vetch_registry_lookup,
}


@app.list_tools()  # type: ignore[untyped-decorator]
async def handle_list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()  # type: ignore[untyped-decorator]
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> CallToolResult:
    arguments = arguments or {}

    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return _error_result(f"Unknown tool: {name}")

    # Dispatch synchronous Vetch calls off the event loop to avoid
    # blocking the MCP server's heartbeat / message processing.
    result = await asyncio.to_thread(handler, **arguments)

    # If the tool handler itself returned an error dict (via @_safe),
    # signal it as a tool error so the agent knows it failed.
    if isinstance(result, dict) and "error" in result:
        return _error_result(result["error"])

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2))],
    )


def _error_result(message: str) -> CallToolResult:
    """Build an MCP tool-error response with ``isError=True``.

    This tells the calling agent that the tool invocation failed,
    preventing it from interpreting the response as valid measurement
    data.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": message}))],
        isError=True,
    )


@app.list_resources()  # type: ignore[untyped-decorator]
async def handle_list_resources() -> list[Resource]:
    return RESOURCES


@app.read_resource()  # type: ignore[untyped-decorator]
async def handle_read_resource(uri: str) -> str:
    if uri == "vetch://registry/models":
        return json.dumps(list_models(), indent=2)
    elif uri.startswith("vetch://registry/energy/"):
        model = uri.removeprefix("vetch://registry/energy/")
        return json.dumps(get_energy_data(model), indent=2)
    elif uri.startswith("vetch://registry/pricing/"):
        model = uri.removeprefix("vetch://registry/pricing/")
        return json.dumps(get_pricing_data(model), indent=2)
    elif uri == "vetch://config":
        return json.dumps(get_config(), indent=2)
    elif uri == "vetch://version":
        return get_version()
    else:
        return json.dumps({"error": f"Unknown resource: {uri}"})


async def main() -> None:
    """Run the Vetch MCP server on stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream, app.create_initialization_options()
        )
