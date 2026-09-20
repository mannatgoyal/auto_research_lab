"""Integration test establishing the Lineage Invariant for Autonomous ML Lab.

Lineage Invariant:
Every tracked model candidate logged in MLflow must be cryptographically and structurally
traceable to its exact computational inputs:
  1. dataset_fingerprint (SHA-256 of data, schema, and index)
  2. split_hashes (train_sha256 and test_sha256)
  3. config_hash_sha256 (SHA-256 of canonical configuration)
  4. random seed
  5. Git commit SHA
  6. parent_run_id (grouping the candidate within the empirical experiment)
"""

from pathlib import Path
import tempfile
from mlflow.tracking import MlflowClient
import pytest

from ml.config import ExperimentConfig
from ml.runner import execute_experiment
from ml.tracking.mlflow_tracker import MLflowTracker


def test_lineage_invariant_end_to_end():
    yaml_config = """
    name: "lineage_verification_experiment"
    seed: 777
    dataset:
      name: "breast_cancer"
    split:
      strategy: "stratified"
      test_size: 0.2
      val_size: 0.1
      seed: 777
    preprocessing:
      scaler: "standard"
      numeric_imputer: "median"
    models:
      - name: "logistic_regression"
        hyperparameters:
          C: 1.0
      - name: "torch_mlp"
        hyperparameters:
          hidden_dims: [32]
          batch_size: 32
          max_epochs: 10
    latency:
      batch_sizes: [1]
      warmup_iterations: 5
      benchmark_iterations: 20
    """
    config = ExperimentConfig.from_yaml(yaml_config)

    # 1. Deterministic Execution
    execution = execute_experiment(config)
    res = execution.result

    expected_config_hash = res.config_hash_sha256
    expected_dataset_fingerprint = res.split_info["dataset_fingerprint"]
    expected_train_sha256 = res.split_info["train_sha256"]
    expected_test_sha256 = res.split_info["test_sha256"]
    expected_seed = str(res.seed)
    expected_git_commit = res.environment["git_commit"]

    # 2. Track with MLflow
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).as_posix()
        tracking_uri = f"file:///{tmp_path}/mlruns"
        tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="lineage_test_exp")
        parent_run_id = tracker.log_execution(execution, log_models=True)

        client = MlflowClient(tracking_uri=tracking_uri)

        # 3. Verify Parent Run Lineage Links
        parent_run = client.get_run(parent_run_id)
        assert parent_run.data.tags["config_hash_sha256"] == expected_config_hash
        assert parent_run.data.tags["dataset_fingerprint"] == expected_dataset_fingerprint
        assert parent_run.data.tags["git_commit"] == expected_git_commit
        assert parent_run.data.params["train_sha256"] == expected_train_sha256
        assert parent_run.data.params["test_sha256"] == expected_test_sha256

        # 4. Verify Every Child Run Lineage Invariant
        child_runs = client.search_runs(
            experiment_ids=[parent_run.info.experiment_id],
            filter_string=f"tags.parent_run_id = '{parent_run_id}'",
        )
        assert len(child_runs) == 2

        for child in child_runs:
            # Check cryptographic lineage tags
            tags = child.data.tags
            params = child.data.params

            assert tags["parent_run_id"] == parent_run_id, "Missing parent run association"
            assert tags["config_hash_sha256"] == expected_config_hash, "Config hash mismatch"
            assert tags["dataset_fingerprint"] == expected_dataset_fingerprint, "Dataset fingerprint mismatch"
            assert tags["train_sha256"] == expected_train_sha256, "Train split hash mismatch"
            assert tags["test_sha256"] == expected_test_sha256, "Test split hash mismatch"
            assert tags["git_commit"] == expected_git_commit, "Git commit mismatch"
            assert params["seed"] == expected_seed, "Seed mismatch"

            # Check that model artifact is present
            artifacts = [a.path for a in client.list_artifacts(child.info.run_id)]
            assert "model" in artifacts, f"Missing model binary for {tags['model_name']}"
            assert "model_metadata.json" in artifacts
