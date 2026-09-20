"""Unit tests for the MCP audit logging framework."""

import os
import tempfile
import pytest
from mcp_servers.common.audit import AuditLogger, sanitize_arguments
from mcp_servers.common.permissions import PermissionTier
from mcp_servers.ml_server.tools import tool_inspect_dataset


def test_sanitize_arguments():
    raw_args = {
        "dataset_name": "breast_cancer",
        "api_key": "secret_12345",
        "auth_token": "bearer_abcde",
        "password_hash": "hash_xyz",
        "nested": {
            "user_token": "token_val",
            "safe_val": 42,
        },
        "safe_param": "lbfgs",
    }

    clean = sanitize_arguments(raw_args)
    assert clean["dataset_name"] == "breast_cancer"
    assert clean["api_key"] == "[REDACTED]"
    assert clean["auth_token"] == "[REDACTED]"
    assert clean["password_hash"] == "[REDACTED]"
    assert clean["nested"]["user_token"] == "[REDACTED]"
    assert clean["nested"]["safe_val"] == 42
    assert clean["safe_param"] == "lbfgs"


def test_audit_logger_records_success_and_duration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test_audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        record = logger.log(
            tool_name="inspect_dataset",
            tier=PermissionTier.READ_ONLY,
            input_args={"dataset_name": "breast_cancer", "token": "secret"},
            success=True,
            duration_ms=12.5,
            entity_id="test_entity_123",
        )

        assert record.tool_name == "inspect_dataset"
        assert record.permission_tier == "READ_ONLY"
        assert record.sanitized_input["token"] == "[REDACTED]"
        assert record.success is True
        assert record.duration_ms == 12.5
        assert record.entity_id == "test_entity_123"

        # Check file persistence
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert "inspect_dataset" in lines[0]


def test_audit_logger_records_failure():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test_audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        record = logger.log(
            tool_name="train_model",
            tier=PermissionTier.SAFE_WRITE,
            input_args={"model_name": "unknown_model"},
            success=False,
            duration_ms=5.0,
            error={"code": "VALIDATION_ERROR", "message": "Unknown architecture"},
        )

        assert record.success is False
        assert record.error is not None
        assert record.error["code"] == "VALIDATION_ERROR"


def test_tool_execution_triggers_audit_log():
    from mcp_servers.common.audit import audit_logger

    audit_logger.clear()
    res = tool_inspect_dataset(dataset_name="breast_cancer")
    assert res["success"] is True

    recent = audit_logger.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].tool_name == "inspect_dataset"
    assert recent[0].permission_tier == "READ_ONLY"
    assert recent[0].success is True
    assert recent[0].duration_ms > 0.0
