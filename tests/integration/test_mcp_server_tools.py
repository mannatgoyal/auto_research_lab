import json
import os
from pathlib import Path
import tempfile
import pytest
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

from mcp_servers.ml_server import create_ml_server
from mcp_servers.common.permissions import PermissionTier, permission_registry


@pytest.fixture(scope="module")
def mcp_server():
    """Initializes the ML MCP Server."""
    return create_ml_server()


@pytest.fixture(scope="module")
def sample_raw_feature_dict():
    raw = load_breast_cancer(as_frame=True)
    row = raw.frame.iloc[0].drop("target").to_dict()
    return {col: float(val) for col, val in row.items()}


@pytest.fixture(scope="module")
def shared_mcp_setup(tmp_path_factory):
    """Dynamically trains and tracks a model for MCP integration tests."""
    from ml.config import ExperimentConfig
    from ml.runner import execute_experiment
    from ml.tracking.mlflow_tracker import MLflowTracker
    from mlflow.tracking import MlflowClient

    tmp_dir = tmp_path_factory.mktemp("mcp_test_shared")
    tracking_uri = f"file:///{tmp_dir.as_posix()}/mlruns"
    results_dir = tmp_dir / "results"
    results_dir.mkdir()

    yaml_config = """
    name: "mcp_integration_experiment"
    seed: 42
    dataset:
      name: "breast_cancer"
    split:
      strategy: "stratified"
      test_size: 0.2
      val_size: 0.1
      seed: 42
    preprocessing:
      scaler: "standard"
    models:
      - name: "logistic_regression"
        hyperparameters:
          max_iter: 200
          random_state: 42
    """
    config = ExperimentConfig.from_yaml(yaml_config)
    execution = execute_experiment(config, output_dir=results_dir.as_posix())

    tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="mcp_test")
    parent_run_id = tracker.log_execution(execution, log_models=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    child_runs = client.search_runs(
        experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.parent_run_id = '{parent_run_id}'",
    )
    child_run_id = child_runs[0].info.run_id

    return {
        "tracking_uri": tracking_uri,
        "results_dir": results_dir.as_posix(),
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
        "experiment_id": execution.result.experiment_id,
    }


def _extract_result(res) -> dict:
    if res.structured_content and "result" in res.structured_content:
        return res.structured_content["result"]
    if res.structured_content:
        return res.structured_content
    if res.content and len(res.content) > 0 and hasattr(res.content[0], "text"):
        return json.loads(res.content[0].text)
    return {}


@pytest.mark.anyio
async def test_mcp_server_lists_all_8_tools(mcp_server):
    tools = await mcp_server.list_tools()
    tool_names = {t.name for t in tools}
    expected = {
        "inspect_dataset",
        "create_split",
        "train_model",
        "evaluate_model",
        "measure_inference_latency",
        "get_experiment_result",
        "get_model_lineage",
        "predict",
    }
    assert tool_names == expected
    assert len(tools) == 8


@pytest.mark.anyio
async def test_tool_inspect_dataset(mcp_server):
    res = await mcp_server.call_tool("inspect_dataset", {"dataset_name": "breast_cancer"})
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["row_count"] == 569
    assert data["column_count"] == 31
    assert data["target_column"] == "target"
    assert len(data["features"]) == 30
    assert len(data["fingerprint_sha256"]) == 64


@pytest.mark.anyio
async def test_tool_create_split(mcp_server):
    res = await mcp_server.call_tool(
        "create_split",
        {
            "dataset_name": "breast_cancer",
            "split_strategy": "stratified",
            "test_size": 0.2,
            "val_size": 0.1,
            "seed": 42,
        },
    )
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["train_rows"] == 398
    assert data["val_rows"] == 57
    assert data["test_rows"] == 114
    assert len(data["train_sha256"]) == 64
    assert len(data["test_sha256"]) == 64


@pytest.mark.anyio
async def test_tool_train_model_end_to_end(mcp_server):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).as_posix()
        tracking_uri = f"file:///{tmp_path}/mlruns"
        res = await mcp_server.call_tool(
            "train_model",
            {
                "model_name": "logistic_regression",
                "hyperparameters": {"C": 0.5},
                "dataset_name": "breast_cancer",
                "split_strategy": "stratified",
                "test_size": 0.2,
                "val_size": 0.1,
                "seed": 42,
                "tracking_uri": tracking_uri,
                "experiment_name": "mcp_train_exp",
                "warmup_iterations": 5,
                "benchmark_iterations": 20,
            },
        )
        assert res.is_error is False
        data = _extract_result(res)
        assert data["success"] is True
        assert data["model_name"] == "logistic_regression"
        assert data["test_metrics"]["accuracy"] > 0.8
        assert data["test_metrics"]["f1_binary"] > 0.8
        assert len(data["latency_benchmarks"]) >= 1

        # Check lineage
        assert len(data["lineage"]["config_hash_sha256"]) == 64
        assert len(data["lineage"]["dataset_fingerprint"]) == 64

        # Check MLflow run tracking
        assert data["tracking"]["parent_run_id"] is not None
        assert data["tracking"]["child_run_id"] is not None


@pytest.mark.anyio
async def test_tool_evaluate_model_direct(mcp_server):
    res = await mcp_server.call_tool(
        "evaluate_model",
        {
            "y_true": [0, 1, 1, 0, 1],
            "y_pred": [0, 1, 1, 0, 1],
            "y_prob": [0.1, 0.9, 0.85, 0.05, 0.95],
        },
    )
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["accuracy"] == 1.0
    assert data["precision_binary"] == 1.0
    assert data["recall_binary"] == 1.0
    assert data["f1_binary"] == 1.0
    assert data["source"] == "direct_evaluation"


@pytest.mark.anyio
async def test_tool_measure_inference_latency(mcp_server, shared_mcp_setup):
    res = await mcp_server.call_tool(
        "measure_inference_latency",
        {
            "run_id": shared_mcp_setup["child_run_id"],
            "batch_sizes": [1],
            "warmup_iterations": 5,
            "benchmark_iterations": 20,
            "tracking_uri": shared_mcp_setup["tracking_uri"],
        },
    )
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["model_name"] == "logistic_regression"
    assert len(data["benchmarks"]) == 1
    assert data["benchmarks"][0]["p95_latency_ms"] > 0.0


@pytest.mark.anyio
async def test_tool_get_model_lineage(mcp_server, shared_mcp_setup):
    res = await mcp_server.call_tool(
        "get_model_lineage",
        {
            "run_id": shared_mcp_setup["child_run_id"],
            "tracking_uri": shared_mcp_setup["tracking_uri"],
        },
    )
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["run_id"] == shared_mcp_setup["child_run_id"]
    assert data["parent_run_id"] == shared_mcp_setup["parent_run_id"]
    assert data["model_name"] == "logistic_regression"
    assert len(data["config_hash_sha256"]) == 64
    assert len(data["dataset_fingerprint"]) == 64


@pytest.mark.anyio
async def test_tool_get_experiment_result(mcp_server, shared_mcp_setup):
    res = await mcp_server.call_tool(
        "get_experiment_result",
        {
            "experiment_id": shared_mcp_setup["experiment_id"],
            "results_dir": shared_mcp_setup["results_dir"],
        },
    )
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["experiment_id"] == shared_mcp_setup["experiment_id"]
    assert len(data["model_results"]) >= 1


@pytest.mark.anyio
async def test_tool_predict_matches_direct_inference(mcp_server, sample_raw_feature_dict, shared_mcp_setup):
    res = await mcp_server.call_tool(
        "predict",
        {
            "features": sample_raw_feature_dict,
            "run_id": shared_mcp_setup["child_run_id"],
            "tracking_uri": shared_mcp_setup["tracking_uri"],
        },
    )
    assert res.is_error is False
    data = _extract_result(res)
    assert data["success"] is True
    assert data["predicted_class"] in (0, 1)
    assert data["predicted_label"] in ("malignant", "benign")
    assert "malignant" in data["probabilities"]
    assert data["inference_latency_ms"] > 0.0


@pytest.mark.anyio
async def test_mcp_safety_rejects_directory_traversal(mcp_server):
    res = await mcp_server.call_tool(
        "inspect_dataset",
        {"dataset_name": "../../sensitive_system_file"},
    )
    data = _extract_result(res)
    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "Directory traversal" in data["error"]["message"]


@pytest.mark.anyio
async def test_mcp_safety_rejects_code_injection_in_model_name(mcp_server):
    res = await mcp_server.call_tool(
        "train_model",
        {
            "model_name": "__import__('os').system('echo pwned')",
            "dataset_name": "breast_cancer",
        },
    )
    data = _extract_result(res)
    assert data["success"] is False
    assert data["error"]["code"] in ("VALIDATION_ERROR", "ENGINE_ERROR")
