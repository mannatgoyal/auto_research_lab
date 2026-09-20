"""Unit tests for the MCP three-tier permission model."""

import pytest
from mcp_servers.common.permissions import PermissionRegistry, PermissionTier, permission_registry
from mcp_servers.ml_server import create_ml_server
from mcp_servers.ml_server.tools import tool_train_model, tool_inspect_dataset


@pytest.fixture(scope="module", autouse=True)
def init_server():
    """Ensure all tools are registered in permission_registry."""
    create_ml_server()


def test_permission_hierarchy():
    read_only = PermissionTier.READ_ONLY
    safe_write = PermissionTier.SAFE_WRITE
    high_risk = PermissionTier.HIGH_RISK

    assert read_only.level == 1
    assert safe_write.level == 2
    assert high_risk.level == 3

    assert safe_write.can_access(read_only) is True
    assert safe_write.can_access(safe_write) is True
    assert safe_write.can_access(high_risk) is False

    assert read_only.can_access(read_only) is True
    assert read_only.can_access(safe_write) is False
    assert read_only.can_access(high_risk) is False


def test_all_expected_tools_registered_with_correct_tiers():
    expected_read_only = {
        "inspect_dataset",
        "evaluate_model",
        "measure_inference_latency",
        "get_experiment_result",
        "get_model_lineage",
        "predict",
    }
    expected_safe_write = {
        "create_split",
        "train_model",
    }

    for tool in expected_read_only:
        assert permission_registry.is_registered(tool), f"Missing tool: {tool}"
        assert permission_registry.get_tier(tool) == PermissionTier.READ_ONLY

    for tool in expected_safe_write:
        assert permission_registry.is_registered(tool), f"Missing tool: {tool}"
        assert permission_registry.get_tier(tool) == PermissionTier.SAFE_WRITE


def test_no_high_risk_tools_exposed_in_phase_4():
    """Critical safety invariant: Phase 4 must not expose any HIGH_RISK tools."""
    assert permission_registry.has_high_risk_tools() is False
    high_risk_tools = permission_registry.list_tools_by_tier(PermissionTier.HIGH_RISK)
    assert len(high_risk_tools) == 0


def test_permission_enforcement_blocks_unauthorized_call():
    # Attempting to call SAFE_WRITE tool with READ_ONLY credentials
    res = tool_train_model(
        model_name="logistic_regression",
        _caller_tier=PermissionTier.READ_ONLY,
    )
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"
    assert "cannot execute tool 'train_model'" in res["error"]["message"]


def test_permission_enforcement_allows_authorized_call():
    # Calling READ_ONLY tool with READ_ONLY credentials succeeds
    res = tool_inspect_dataset(
        dataset_name="breast_cancer",
        _caller_tier=PermissionTier.READ_ONLY,
    )
    assert res["success"] is True
    assert res["dataset_name"] == "UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic)"
