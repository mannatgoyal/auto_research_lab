"""Integration test running an end-to-end deterministic ML experiment.

Guarantees:
- Runs without any LLM, MCP server, database, or external service.
- Tests complete flow: Dataset -> Split -> Preprocessing -> Train -> Evaluate -> Latency.
- Confirms reproducibility across identical runs.
"""

import tempfile
import pytest
from ml.config import ExperimentConfig
from ml.runner import run_experiment


def test_full_pipeline_execution():
    yaml_config = """
    name: "integration_test_experiment"
    seed: 42
    dataset:
      name: "breast_cancer"
    split:
      strategy: "stratified"
      test_size: 0.2
      val_size: 0.1
      seed: 42
    preprocessing:
      numeric_imputer: "median"
      scaler: "standard"
    models:
      - name: "logistic_regression"
        hyperparameters:
          C: 1.0
          max_iter: 500
      - name: "xgboost"
        hyperparameters:
          n_estimators: 25
          max_depth: 3
    latency:
      batch_sizes: [1, 16]
      warmup_iterations: 10
      benchmark_iterations: 30
      device: "cpu"
    """
    config = ExperimentConfig.from_yaml(yaml_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_experiment(config, output_dir=tmp_dir)

        # Verify metadata
        assert result.experiment_name == "integration_test_experiment"
        assert result.seed == 42
        assert len(result.model_results) == 2

        # Verify dataset profiling was generated
        assert result.dataset_profile["row_count"] == 569
        assert result.dataset_profile["column_count"] == 31

        # Verify split hashes exist and are valid SHA-256
        assert len(result.split_info["train_sha256"]) == 64
        assert len(result.split_info["test_sha256"]) == 64

        # Verify model results
        for m in result.model_results:
            assert m.test_metrics.accuracy > 0.85
            assert m.test_metrics.f1_binary > 0.85
            assert m.test_metrics.confusion_matrix.true_positive > 0
            assert len(m.latency_benchmarks) == 2
            for lb in m.latency_benchmarks:
                assert lb.p95_latency_ms > 0.0
                assert lb.throughput_qps > 0.0

        # Verify summary table formatting
        summary = result.summary_table()
        assert "logistic_regression" in summary
        assert "xgboost" in summary


def test_pipeline_reproducibility():
    """Confirms identical random seeds produce identical numerical test metrics."""
    yaml_config = """
    name: "reproducibility_check"
    seed: 999
    dataset:
      name: "breast_cancer"
    split:
      strategy: "stratified"
      test_size: 0.2
      val_size: 0.1
      seed: 999
    preprocessing:
      scaler: "standard"
    models:
      - name: "logistic_regression"
        hyperparameters:
          C: 1.0
    latency:
      batch_sizes: [1]
      warmup_iterations: 5
      benchmark_iterations: 20
    """
    config1 = ExperimentConfig.from_yaml(yaml_config)
    config2 = ExperimentConfig.from_yaml(yaml_config)

    res1 = run_experiment(config1)
    res2 = run_experiment(config2)

    # Identical dataset and split hashes
    assert res1.split_info["train_sha256"] == res2.split_info["train_sha256"]
    assert res1.split_info["test_sha256"] == res2.split_info["test_sha256"]

    # Identical metrics
    m1 = res1.model_results[0].test_metrics
    m2 = res2.model_results[0].test_metrics
    assert m1.accuracy == m2.accuracy
    assert m1.f1_binary == m2.f1_binary
    assert m1.roc_auc == m2.roc_auc
