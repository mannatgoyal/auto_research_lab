"""Unit tests for ModelLoader and InferencePipeline abstraction."""

import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
from mlflow.tracking import MlflowClient

from api.config import ServingConfig
from api.loader import ModelLoader, InferencePipeline
from ml.config import ExperimentConfig
from ml.runner import execute_experiment
from ml.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture(scope="module")
def tracked_experiment_artifact():
    """Runs a quick deterministic experiment and logs to a temporary MLflow store."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).as_posix()
        tracking_uri = f"file:///{tmp_path}/mlruns"
        yaml_config = """
        name: "test_loader_exp"
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
          numeric_imputer: "median"
        models:
          - name: "logistic_regression"
            hyperparameters:
              C: 1.0
        latency:
          batch_sizes: [1]
          warmup_iterations: 5
          benchmark_iterations: 20
        """
        config = ExperimentConfig.from_yaml(yaml_config)
        execution = execute_experiment(config)

        tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="test_loader_experiment")
        parent_run_id = tracker.log_execution(execution, log_models=True)

        client = MlflowClient(tracking_uri=tracking_uri)
        children = client.search_runs(
            experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
            filter_string=f"tags.parent_run_id = '{parent_run_id}'",
        )
        child_run_id = children[0].info.run_id

        yield {
            "tracking_uri": tracking_uri,
            "parent_run_id": parent_run_id,
            "child_run_id": child_run_id,
            "execution": execution,
        }


def test_model_loader_success(tracked_experiment_artifact):
    artifact = tracked_experiment_artifact
    config = ServingConfig(
        model_source="mlflow",
        tracking_uri=artifact["tracking_uri"],
        run_id=artifact["child_run_id"],
    )

    loader = ModelLoader(tracking_uri=artifact["tracking_uri"])
    pipeline = loader.load_pipeline(config)

    assert isinstance(pipeline, InferencePipeline)
    assert pipeline.model_name == "logistic_regression"
    assert pipeline.run_id == artifact["child_run_id"]
    assert pipeline.parent_run_id == artifact["parent_run_id"]

    # Lineage metadata verification
    meta = pipeline.metadata
    assert meta["dataset_fingerprint"] == artifact["execution"].result.split_info["dataset_fingerprint"]
    assert meta["config_hash_sha256"] == artifact["execution"].result.config_hash_sha256
    assert len(meta["expected_features"]) == 30

    # Test that preprocessor and model can predict on raw sample DataFrame
    sample_df = artifact["execution"].result.dataset_profile
    # Take a raw row from test set
    raw_bundle = artifact["execution"]
    # Create test input DataFrame
    test_row = pd.DataFrame([{feat: 1.0 for feat in meta["expected_features"]}])
    pred_class, probs = pipeline.predict(test_row)
    assert pred_class in (0, 1)
    assert probs is not None
    assert "malignant" in probs and "benign" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_model_loader_nonexistent_run(tracked_experiment_artifact):
    config = ServingConfig(
        model_source="mlflow",
        tracking_uri=tracked_experiment_artifact["tracking_uri"],
        run_id="nonexistent_run_id_12345",
    )
    loader = ModelLoader(tracking_uri=tracked_experiment_artifact["tracking_uri"])
    with pytest.raises(ValueError, match="Could not retrieve MLflow run"):
        loader.load_pipeline(config)


def test_model_loader_missing_parent_tag(tracked_experiment_artifact):
    # Create an independent run with no parent_run_id tag
    tracking_uri = tracked_experiment_artifact["tracking_uri"]
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.create_experiment("orphan_exp")
    with mlflow.start_run(experiment_id=exp) as orphan_run:
        orphan_id = orphan_run.info.run_id

    config = ServingConfig(
        model_source="mlflow",
        tracking_uri=tracking_uri,
        run_id=orphan_id,
    )
    loader = ModelLoader(tracking_uri=tracking_uri)
    with pytest.raises(ValueError, match="missing required 'parent_run_id' tag"):
        loader.load_pipeline(config)
