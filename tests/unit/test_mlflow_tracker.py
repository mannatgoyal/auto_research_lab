"""Unit tests for MLflowTracker adapter."""

import os
from pathlib import Path
import tempfile
import mlflow
from mlflow.tracking import MlflowClient
import pytest

from ml.config import ExperimentConfig
from ml.runner import execute_experiment
from ml.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def test_execution():
    yaml_config = """
    name: "test_mlflow_exp"
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
          C: 0.5
      - name: "xgboost"
        hyperparameters:
          n_estimators: 20
          max_depth: 3
    latency:
      batch_sizes: [1, 16]
      warmup_iterations: 5
      benchmark_iterations: 20
    """
    config = ExperimentConfig.from_yaml(yaml_config)
    return execute_experiment(config)


def test_mlflow_parent_and_child_runs(test_execution):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).as_posix()
        tracking_uri = f"file:///{tmp_path}/mlruns"
        tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="unit_test_exp")

        parent_run_id = tracker.log_execution(test_execution, log_models=True)
        assert parent_run_id is not None
        assert len(parent_run_id) > 0

        client = MlflowClient(tracking_uri=tracking_uri)

        # 1. Verify Parent Run
        parent_run = client.get_run(parent_run_id)
        assert parent_run.data.tags["run_type"] == "experiment_parent"
        assert parent_run.data.tags["experiment_id"] == test_execution.result.experiment_id
        assert len(parent_run.data.tags["config_hash_sha256"]) == 64
        assert len(parent_run.data.tags["dataset_fingerprint"]) == 64

        # Check Parent Parameters
        assert parent_run.data.params["split_strategy"] == "stratified"
        assert parent_run.data.params["preprocessing_scaler"] == "standard"
        assert int(parent_run.data.params["train_rows"]) == test_execution.result.split_info["train_rows"]

        # Check Parent Artifacts
        artifacts = [a.path for a in client.list_artifacts(parent_run_id)]
        assert "results.json" in artifacts
        assert "dataset_metadata.json" in artifacts
        assert "split_metadata.json" in artifacts
        assert "environment.json" in artifacts

        # 2. Verify Child Runs
        # Search for runs with parent_run_id tag
        child_runs = client.search_runs(
            experiment_ids=[parent_run.info.experiment_id],
            filter_string=f"tags.parent_run_id = '{parent_run_id}'",
        )
        assert len(child_runs) == 2

        for child in child_runs:
            model_name = child.data.tags["model_name"]
            assert model_name in ("logistic_regression", "xgboost")
            assert child.data.tags["run_type"] == "model_child"

            # Check Child Parameters
            assert child.data.params["model_type"] == model_name
            assert child.data.params["seed"] == "42"

            # Check Child Metrics
            assert "test_accuracy" in child.data.metrics
            assert "test_balanced_accuracy" in child.data.metrics
            assert "test_f1_binary" in child.data.metrics
            assert "latency_b1_p95_ms" in child.data.metrics
            assert "latency_b1_throughput_samples_per_sec" in child.data.metrics
            assert child.data.metrics["test_accuracy"] > 0.8

            # Check Child Artifacts
            child_artifacts = [a.path for a in client.list_artifacts(child.info.run_id)]
            assert "model_metadata.json" in child_artifacts
            assert "model" in child_artifacts


def test_mlflow_model_logging_disabled(test_execution):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).as_posix()
        tracking_uri = f"file:///{tmp_path}/mlruns"
        tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="no_model_log_exp")

        parent_run_id = tracker.log_execution(test_execution, log_models=False)
        client = MlflowClient(tracking_uri=tracking_uri)

        child_runs = client.search_runs(
            experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
            filter_string=f"tags.parent_run_id = '{parent_run_id}'",
        )

        for child in child_runs:
            child_artifacts = [a.path for a in client.list_artifacts(child.info.run_id)]
            assert "model_metadata.json" in child_artifacts
            # model folder should NOT be present when log_models=False
            assert "model" not in child_artifacts
