"""Unit tests for MCPToolClient interface, permission enforcement, and state logging."""

import pytest
from mcp_servers.common.permissions import PermissionTier, permission_registry
from mcp_servers.ml_server import create_ml_server
from orchestrator.client import MCPToolClient
from orchestrator.state import ResearchState


@pytest.fixture
def mcp_server():
    return create_ml_server()


@pytest.mark.anyio
async def test_client_lists_available_tools(mcp_server):
    client = MCPToolClient(server=mcp_server)
    tools = await client.list_available_tools()
    assert "inspect_dataset" in tools
    assert "create_split" in tools
    assert "train_model" in tools
    assert len(tools) == 8


@pytest.mark.anyio
async def test_client_calls_mcp_tool_and_logs_to_state(mcp_server):
    client = MCPToolClient(server=mcp_server)
    state = ResearchState(original_question="Test inspect")

    success, data, err = await client.call_tool(
        tool_name="inspect_dataset",
        arguments={"dataset_name": "breast_cancer"},
        state=state,
    )

    assert success is True
    assert err is None
    assert data["row_count"] == 569
    assert len(state.tool_calls) == 1
    record = state.tool_calls[0]
    assert record.tool_name == "inspect_dataset"
    assert record.success is True
    assert record.duration_ms > 0.0


@pytest.mark.anyio
async def test_client_rejects_unregistered_tool(mcp_server):
    client = MCPToolClient(server=mcp_server)
    state = ResearchState(original_question="Test invalid tool")

    success, data, err = await client.call_tool(
        tool_name="nonexistent_mcp_tool",
        arguments={},
        state=state,
    )

    assert success is False
    assert "not registered" in err
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].success is False


@pytest.mark.anyio
async def test_client_enforces_permission_tiers(mcp_server):
    # READ_ONLY caller attempting SAFE_WRITE tool
    ro_client = MCPToolClient(server=mcp_server, caller_tier=PermissionTier.READ_ONLY)
    state = ResearchState(original_question="Permission test")

    success, data, err = await ro_client.call_tool(
        tool_name="create_split",
        arguments={"dataset_name": "breast_cancer"},
        state=state,
    )

    assert success is False
    assert "Permission Denied" in err
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].success is False


@pytest.mark.anyio
async def test_client_blocks_high_risk_tools(mcp_server):
    # Register temporary mock HIGH_RISK tool to verify blockage
    permission_registry.register("mock_high_risk_tool", PermissionTier.HIGH_RISK)
    try:
        # Even with SAFE_WRITE caller tier, HIGH_RISK must be blocked
        client = MCPToolClient(server=mcp_server, caller_tier=PermissionTier.SAFE_WRITE)
        state = ResearchState(original_question="High risk test")

        # Mock server list to include it
        client._cached_tool_names = ["mock_high_risk_tool"]

        success, data, err = await client.call_tool(
            tool_name="mock_high_risk_tool",
            arguments={},
            state=state,
        )

        assert success is False
        assert "HIGH_RISK" in err
        assert state.tool_calls[0].success is False
    finally:
        # Clean up mock tool
        permission_registry._registry.pop("mock_high_risk_tool", None)
