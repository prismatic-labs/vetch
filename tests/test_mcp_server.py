"""Tests for MCP server registration and request handling.

These tests verify:
- Tool and resource listing
- Tool dispatch returns valid JSON via asyncio.to_thread
- Error responses set isError=True (proper MCP error signaling)
- Resource reading returns valid data
- Unknown tool/resource handling
- Server entrypoint is importable
"""

from __future__ import annotations

import json

import pytest

mcp_mod = pytest.importorskip("mcp", reason="mcp package required for server tests")

from vetch.mcp.server import (  # noqa: E402
    RESOURCES,
    TOOLS,
    _error_result,
    handle_call_tool,
    handle_list_resources,
    handle_list_tools,
    handle_read_resource,
)


class TestToolListing:
    """Tests for tool registration."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all(self) -> None:
        """All 8 tools are registered."""
        tools = await handle_list_tools()
        assert len(tools) == 8

    @pytest.mark.asyncio
    async def test_tool_names(self) -> None:
        """All expected tool names are present."""
        tools = await handle_list_tools()
        names = {t.name for t in tools}
        expected = {
            "vetch_estimate",
            "vetch_compare",
            "vetch_session_stats",
            "vetch_status",
            "vetch_check_budget",
            "vetch_grid_intensity",
            "vetch_cleanest_region",
            "vetch_registry_lookup",
        }
        assert names == expected

    def test_tools_have_schemas(self) -> None:
        """Every tool has an inputSchema."""
        for tool in TOOLS:
            assert tool.inputSchema is not None
            assert tool.inputSchema["type"] == "object"


class TestToolDispatch:
    """Tests for handle_call_tool dispatch."""

    @pytest.mark.asyncio
    async def test_estimate_returns_json(self) -> None:
        """vetch_estimate returns parseable JSON via asyncio.to_thread."""
        result = await handle_call_tool(
            "vetch_estimate",
            {"model": "gpt-4o", "input_tokens": 1000, "output_tokens": 500},
        )
        assert not result.isError
        data = json.loads(result.content[0].text)
        assert data["model"] == "gpt-4o"
        assert "energy_wh" in data

    @pytest.mark.asyncio
    async def test_session_stats_no_args(self) -> None:
        """vetch_session_stats works with no arguments."""
        result = await handle_call_tool("vetch_session_stats", None)
        assert not result.isError
        data = json.loads(result.content[0].text)
        assert "total_requests" in data

    @pytest.mark.asyncio
    async def test_status_no_args(self) -> None:
        """vetch_status works with no arguments."""
        result = await handle_call_tool("vetch_status", None)
        assert not result.isError
        data = json.loads(result.content[0].text)
        assert "version" in data

    @pytest.mark.asyncio
    async def test_check_budget_no_args(self) -> None:
        """vetch_check_budget works with no arguments."""
        result = await handle_call_tool("vetch_check_budget", None)
        assert not result.isError
        data = json.loads(result.content[0].text)
        assert "budgets" in data

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_with_flag(self) -> None:
        """Unknown tool name returns isError=True."""
        result = await handle_call_tool("nonexistent_tool", {})
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_tool_handler_error_returns_is_error(self) -> None:
        """When a tool handler returns an error dict, isError=True."""
        result = await handle_call_tool(
            "vetch_registry_lookup", {"model": "nonexistent-xyz"},
        )
        assert result.isError is True


class TestResourceListing:
    """Tests for resource registration."""

    @pytest.mark.asyncio
    async def test_list_resources(self) -> None:
        """Resources are listed."""
        resources = await handle_list_resources()
        assert len(resources) == 3

    def test_resource_uris(self) -> None:
        """All expected URIs are present."""
        uris = {str(r.uri) for r in RESOURCES}
        assert "vetch://registry/models" in uris
        assert "vetch://config" in uris
        assert "vetch://version" in uris


class TestResourceReading:
    """Tests for handle_read_resource."""

    @pytest.mark.asyncio
    async def test_read_models(self) -> None:
        """Reading models resource returns a JSON list."""
        result = await handle_read_resource("vetch://registry/models")
        models = json.loads(result)
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_read_config(self) -> None:
        """Reading config resource returns expected keys."""
        result = await handle_read_resource("vetch://config")
        config = json.loads(result)
        assert "region" in config
        assert "default_pue" in config

    @pytest.mark.asyncio
    async def test_read_version(self) -> None:
        """Reading version resource returns a string."""
        result = await handle_read_resource("vetch://version")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_read_energy_for_model(self) -> None:
        """Reading energy data via URI template."""
        result = await handle_read_resource("vetch://registry/energy/gpt-4o")
        data = json.loads(result)
        assert data["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_read_pricing_for_model(self) -> None:
        """Reading pricing data via URI template."""
        result = await handle_read_resource("vetch://registry/pricing/gpt-4o")
        data = json.loads(result)
        assert data["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_read_unknown_resource(self) -> None:
        """Unknown URI returns error JSON."""
        result = await handle_read_resource("vetch://nonexistent")
        data = json.loads(result)
        assert "error" in data


class TestErrorResult:
    """Tests for _error_result helper."""

    def test_error_result_sets_is_error(self) -> None:
        """_error_result returns CallToolResult with isError=True."""
        result = _error_result("something broke")
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert data["error"] == "something broke"


class TestEntrypoint:
    """Tests for __main__ module."""

    def test_main_importable(self) -> None:
        """vetch.mcp.__main__ is importable."""
        from vetch.mcp.__main__ import _run
        assert callable(_run)

    def test_server_main_importable(self) -> None:
        """vetch.mcp.server.main is an async function."""
        import asyncio

        from vetch.mcp.server import main
        assert asyncio.iscoroutinefunction(main)
