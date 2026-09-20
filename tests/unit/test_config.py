"""Unit tests for experiment configuration parsing and validation."""

import pytest
from ml.config import ExperimentConfig, SplitConfig


def test_valid_yaml_parsing():
    yaml_content = """
    name: "test_exp"
    seed: 123
    dataset:
      name: "breast_cancer"
    split:
      strategy: "stratified"
      test_size: 0.2
      val_size: 0.1
      seed: 123
    preprocessing:
      numeric_imputer: "mean"
      scaler: "robust"
    models:
      - name: "logistic_regression"
        hyperparameters:
          C: 0.5
    latency:
      batch_sizes: [1, 8]
      warmup_iterations: 10
      benchmark_iterations: 50
    """
    config = ExperimentConfig.from_yaml(yaml_content)
    assert config.name == "test_exp"
    assert config.seed == 123
    assert config.dataset.name == "breast_cancer"
    assert config.split.val_size == 0.1
    assert config.preprocessing.scaler == "robust"
    assert len(config.models) == 1
    assert config.models[0].name == "logistic_regression"
    assert config.latency.batch_sizes == [1, 8]


def test_wrapped_experiment_key():
    yaml_content = """
    experiment:
      name: "wrapped_exp"
      seed: 42
      dataset:
        name: "breast_cancer"
      models:
        - name: "random_forest"
    """
    config = ExperimentConfig.from_yaml(yaml_content)
    assert config.name == "wrapped_exp"
    assert config.models[0].name == "random_forest"


def test_invalid_split_sizes_raise_error():
    with pytest.raises(ValueError, match="must be < 1.0"):
        SplitConfig(test_size=0.5, val_size=0.5)


def test_empty_models_raise_error():
    with pytest.raises(Exception):
        ExperimentConfig(
            name="bad_exp",
            dataset={"name": "breast_cancer"},
            models=[],
        )
