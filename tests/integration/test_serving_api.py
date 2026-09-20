"""Integration tests for FastAPI model serving endpoints and mathematical equivalence."""

import os
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import joblib
import mlflow

from pathlib import Path
from mlflow.tracking import MlflowClient

from api.config import ServingConfig
from api.loader import ModelLoader
from api.server import app, inference_service
from api.schemas import FEATURE_NAMES_30
from ml.config import ExperimentConfig
from ml.runner import execute_experiment
from ml.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture(scope="module")
def serving_setup(tmp_path_factory):
    """Dynamically trains and tracks a model in MLflow for self-contained serving tests."""
    tmp_dir = tmp_path_factory.mktemp("serving_api_test")
    tracking_uri = f"file:///{tmp_dir.as_posix()}/mlruns"

    yaml_config = """
    name: "serving_integration_experiment"
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
    execution = execute_experiment(config)

    tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="serving_test")
    parent_run_id = tracker.log_execution(execution, log_models=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    child_runs = client.search_runs(
        experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.parent_run_id = '{parent_run_id}'",
    )
    child_run_id = child_runs[0].info.run_id

    return {
        "tracking_uri": tracking_uri,
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
    }


@pytest.fixture(scope="module")
def api_client(serving_setup):
    """Initializes the TestClient using the dynamically-tracked model."""
    old_cfg_path = os.environ.get("SERVING_CONFIG_PATH")
    os.environ["SERVING_CONFIG_PATH"] = "__non_existent_test_config__.yaml"
    try:
        with TestClient(app) as client:
            srv_config = ServingConfig(
                model_source="mlflow",
                tracking_uri=serving_setup["tracking_uri"],
                run_id=serving_setup["child_run_id"],
            )
            inference_service.initialize(srv_config)
            yield client
    finally:
        if old_cfg_path is not None:
            os.environ["SERVING_CONFIG_PATH"] = old_cfg_path
        else:
            os.environ.pop("SERVING_CONFIG_PATH", None)


@pytest.fixture
def sample_raw_feature_dict():
    """Extracts a real raw sample row from the UCI Breast Cancer dataset."""
    raw = load_breast_cancer(as_frame=True)
    row = raw.frame.iloc[0].drop("target").to_dict()
    return {col: float(val) for col, val in row.items()}


def test_health_endpoint_when_loaded(api_client, serving_setup):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "logistic_regression"
    assert data["run_id"] == serving_setup["child_run_id"]


def test_health_endpoint_when_unloaded(api_client):
    # Temporarily unset the pipeline to test degraded state
    saved_pipeline = inference_service.pipeline
    try:
        inference_service.pipeline = None
        response = api_client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
    finally:
        inference_service.pipeline = saved_pipeline


def test_model_metadata_endpoint(api_client, serving_setup):
    response = api_client.get("/v1/models/current")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "logistic_regression"
    assert data["run_id"] == serving_setup["child_run_id"]
    assert data["parent_run_id"] == serving_setup["parent_run_id"]
    assert "dataset_fingerprint" in data and len(data["dataset_fingerprint"]) == 64
    assert "config_hash_sha256" in data and len(data["config_hash_sha256"]) == 64
    assert len(data["expected_features"]) == 30
    assert data["transformed_features_count"] == 30


def test_predict_valid_request(api_client, sample_raw_feature_dict, serving_setup):
    response = api_client.post("/predict", json={"features": sample_raw_feature_dict})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in (0, 1)
    assert data["predicted_label"] in ("malignant", "benign")
    assert data["probabilities"] is not None
    assert "malignant" in data["probabilities"]
    assert "benign" in data["probabilities"]
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-4

    # Check model identity and server-side latency
    assert data["model_identity"]["run_id"] == serving_setup["child_run_id"]
    assert data["model_identity"]["parent_run_id"] == serving_setup["parent_run_id"]
    assert data["inference_latency_ms"] > 0.0


def test_predict_flat_dictionary_support(api_client, sample_raw_feature_dict):
    # Tests that request body can also be a flat feature dictionary
    response = api_client.post("/predict", json=sample_raw_feature_dict)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in (0, 1)


def test_predict_missing_feature_rejected(api_client, sample_raw_feature_dict):
    del sample_raw_feature_dict["worst fractal dimension"]
    response = api_client.post("/predict", json={"features": sample_raw_feature_dict})
    assert response.status_code == 422
    data = response.json()
    assert "Input schema validation error" in data["detail"]
    assert any("worst fractal dimension" in err for err in data["errors"])


def test_predict_unexpected_field_rejected(api_client, sample_raw_feature_dict):
    sample_raw_feature_dict["spurious_extra_column"] = 12.34
    response = api_client.post("/predict", json={"features": sample_raw_feature_dict})
    assert response.status_code == 422
    data = response.json()
    assert "Input schema validation error" in data["detail"]
    assert any("spurious_extra_column" in err for err in data["errors"])


def test_predict_nan_rejected(api_client, sample_raw_feature_dict):
    sample_raw_feature_dict["mean radius"] = None
    response = api_client.post("/predict", json={"features": sample_raw_feature_dict})
    assert response.status_code == 422


def test_predict_when_model_unloaded_returns_503(api_client, sample_raw_feature_dict):
    saved_pipeline = inference_service.pipeline
    try:
        inference_service.pipeline = None
        response = api_client.post("/predict", json={"features": sample_raw_feature_dict})
        assert response.status_code == 503
        assert "loaded" in response.json()["detail"].lower()
    finally:
        inference_service.pipeline = saved_pipeline


# ==============================================================================
# CRITICAL MATHEMATICAL EQUIVALENCE TEST
# ==============================================================================

def test_mathematical_equivalence_direct_vs_api(api_client, sample_raw_feature_dict, serving_setup):
    """Proves exact mathematical equivalence between direct Phase 1 inference and the serving API.

    1. Executes direct inference by loading the exact preprocessor and model estimator artifacts.
    2. Sends the exact same raw feature input through the FastAPI HTTP service.
    3. Asserts predictions and probabilities match within numerical precision.
    """
    # 1. Direct Inference: Download exact artifacts directly from MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    prep_path = mlflow.artifacts.download_artifacts(
        run_id=serving_setup["parent_run_id"],
        artifact_path="preprocessing/pipeline.joblib",
        tracking_uri=serving_setup["tracking_uri"],
    )
    direct_preprocessor = joblib.load(prep_path)

    model_path = mlflow.artifacts.download_artifacts(
        run_id=serving_setup["child_run_id"],
        artifact_path="model/model.joblib",
        tracking_uri=serving_setup["tracking_uri"],
    )
    direct_model = joblib.load(model_path)

    # 2. Compute Direct Phase 1 predictions
    row_df = pd.DataFrame([sample_raw_feature_dict], columns=FEATURE_NAMES_30)
    X_preprocessed = direct_preprocessor.transform(row_df)
    X_array = np.asarray(X_preprocessed, dtype=np.float32)

    direct_pred = int(direct_model.predict(X_array)[0])
    direct_probs = direct_model.predict_proba(X_array)[0]
    direct_prob_neg = float(direct_probs[0])
    direct_prob_pos = float(direct_probs[1])

    # 3. Compute API predictions via HTTP POST /predict
    response = api_client.post("/predict", json={"features": sample_raw_feature_dict})
    assert response.status_code == 200
    api_data = response.json()

    # 4. Strict Mathematical Assertions
    assert api_data["predicted_class"] == direct_pred, (
        f"Class mismatch: API predicted {api_data['predicted_class']} vs direct {direct_pred}"
    )

    api_prob_neg = api_data["probabilities"]["malignant"]
    api_prob_pos = api_data["probabilities"]["benign"]

    np.testing.assert_allclose(
        api_prob_neg,
        direct_prob_neg,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Malignant probability mismatch between API and direct Phase 1 inference",
    )
    np.testing.assert_allclose(
        api_prob_pos,
        direct_prob_pos,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Benign probability mismatch between API and direct Phase 1 inference",
    )
