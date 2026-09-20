"""Tests for research MCP server structure and initialization."""

import pytest
from mcp_servers.research_server import create_research_server


@pytest.mark.anyio
async def test_research_server_initialization():
    server = create_research_server()
    tools = await server.list_tools()
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "ping_research" in names

    res = await server.call_tool("ping_research", {})
    assert res.is_error is False
    import json
    data = json.loads(res.content[0].text) if res.content else {}
    assert data["status"] == "ok"
    assert "Phase 4" in data["phase"]
