"""Unit tests verifying modular Python functions for future MCP compatibility."""

import numpy as np
from ml.config import (
    DatasetConfig,
    ExperimentConfig,
    LatencyConfig,
    ModelConfig,
    PreprocessingConfig,
    SplitConfig,
)
from ml.datasets.loader import load_dataset
from ml.preprocessing.pipeline import build_and_fit_pipeline
from ml.preprocessing.splitters import create_splits
from ml.runner import compute_config_hash, inspect_dataset_profile, train_and_evaluate_model


def test_inspect_dataset_profile_callable():
    profile = inspect_dataset_profile("breast_cancer")
    assert profile.row_count == 569
    assert profile.column_count == 31
    assert "0" in profile.target_counts


def test_train_and_evaluate_model_callable():
    bundle = load_dataset(DatasetConfig(name="breast_cancer"))
    split_data = create_splits(bundle, SplitConfig(seed=42))
    _, preprocessed = build_and_fit_pipeline(split_data, PreprocessingConfig())

    model_cfg = ModelConfig(name="logistic_regression", hyperparameters={"C": 0.5})
    latency_cfg = LatencyConfig(batch_sizes=[1], warmup_iterations=5, benchmark_iterations=20)

    result = train_and_evaluate_model(
        model_cfg=model_cfg,
        preprocessed=preprocessed,
        latency_cfg=latency_cfg,
        seed=42,
    )

    assert result.model_name == "logistic_regression"
    assert result.test_metrics.accuracy > 0.85
    assert result.test_metrics.balanced_accuracy > 0.85
    assert len(result.latency_benchmarks) == 1
    assert result.latency_benchmarks[0].throughput_samples_per_sec > 0.0


def test_config_hash_determinism():
    yaml_str = """
    name: "hash_check"
    seed: 42
    dataset:
      name: "breast_cancer"
    models:
      - name: "logistic_regression"
    """
    cfg1 = ExperimentConfig.from_yaml(yaml_str)
    cfg2 = ExperimentConfig.from_yaml(yaml_str)

    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)

    assert h1 == h2
    assert len(h1) == 64
