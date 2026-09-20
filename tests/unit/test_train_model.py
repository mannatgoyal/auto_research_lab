"""Unit tests for the high-level train_model standalone entrypoint."""

import json
import os
from pathlib import Path
import tempfile
import pytest

from ml import StructuredTrainingResult, train_model


def test_train_model_with_mlflow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).as_posix()
        tracking_uri = f"file:///{tmp_path}/mlruns"

        res = train_model(
            model_name="logistic_regression",
            hyperparameters={"C": 0.5, "max_iter": 500},
            dataset_name="breast_cancer",
            split_strategy="stratified",
            test_size=0.2,
            val_size=0.1,
            scaler="standard",
            seed=42,
            warmup_iterations=5,
            benchmark_iterations=20,
            tracking_uri=tracking_uri,
            experiment_name="unit_train_model_exp",
            log_models=True,
        )

        assert isinstance(res, StructuredTrainingResult)
        assert res.model_name == "logistic_regression"
        assert res.hyperparameters["C"] == 0.5
        assert res.training_duration_seconds > 0.0

        # Metrics verification
        assert res.test_metrics["accuracy"] > 0.8
        assert res.test_metrics["f1_binary"] > 0.8
        assert "roc_auc" in res.test_metrics
        assert res.train_metrics["accuracy"] > 0.8

        # Latency verification
        assert len(res.latency_benchmarks) >= 1
        assert res.latency_benchmarks[0]["p95_latency_ms"] > 0.0

        # Lineage verification
        assert len(res.lineage["config_hash_sha256"]) == 64
        assert len(res.lineage["dataset_fingerprint"]) == 64
        assert len(res.lineage["train_sha256"]) == 64
        assert len(res.lineage["test_sha256"]) == 64
        assert res.lineage["seed"] == 42

        # MLflow Tracking verification
        assert res.tracking["tracking_uri"] == tracking_uri
        assert res.tracking["parent_run_id"] is not None
        assert res.tracking["child_run_id"] is not None

        # Machine-readable JSON safety
        json_str = res.to_json()
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "logistic_regression"
        assert parsed["tracking"]["parent_run_id"] == res.tracking["parent_run_id"]


def test_train_model_without_tracking():
    # Verify pure deterministic engine execution when tracking_uri=None
    res = train_model(
        model_name="logistic_regression",
        hyperparameters={"C": 1.0},
        dataset_name="breast_cancer",
        tracking_uri=None,
        warmup_iterations=5,
        benchmark_iterations=20,
    )

    assert isinstance(res, StructuredTrainingResult)
    assert res.model_name == "logistic_regression"
    assert res.test_metrics["accuracy"] > 0.8
    assert res.tracking["tracking_uri"] is None
    assert res.tracking["parent_run_id"] is None
    assert res.tracking["child_run_id"] is None
    assert len(res.lineage["config_hash_sha256"]) == 64


def test_train_model_hyperparameters_forwarded():
    res = train_model(
        model_name="random_forest",
        hyperparameters={"n_estimators": 25, "max_depth": 3},
        dataset_name="breast_cancer",
        tracking_uri=None,
        warmup_iterations=5,
        benchmark_iterations=20,
    )
    assert res.model_name == "random_forest"
    assert res.hyperparameters["n_estimators"] == 25
    assert res.hyperparameters["max_depth"] == 3
