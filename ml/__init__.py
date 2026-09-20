"""Autonomous ML Research Lab - Deterministic ML Core Engine."""

from ml.config import (
    DatasetConfig,
    ExperimentConfig,
    LatencyConfig,
    ModelConfig,
    PreprocessingConfig,
    SplitConfig,
)
from ml.datasets.loader import DatasetBundle, load_dataset
from ml.datasets.profiler import DatasetProfile, profile_dataset
from ml.evaluation.latency import LatencyBenchmark, benchmark_inference_latency
from ml.evaluation.metrics import EvaluationMetrics, evaluate_predictions
from ml.models.base import BaseModel, create_model
from ml.preprocessing.pipeline import PreprocessedData, PreprocessorPipeline, build_and_fit_pipeline
from ml.preprocessing.splitters import SplitData, create_splits
from ml.runner import (
    ExperimentExecution,
    ExperimentResult,
    ModelResult,
    StructuredTrainingResult,
    compute_config_hash,
    execute_experiment,
    inspect_dataset_profile,
    run_experiment,
    train_and_evaluate_model,
    train_model,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetConfig",
    "ExperimentConfig",
    "LatencyConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "SplitConfig",
    "DatasetBundle",
    "load_dataset",
    "DatasetProfile",
    "profile_dataset",
    "LatencyBenchmark",
    "benchmark_inference_latency",
    "EvaluationMetrics",
    "evaluate_predictions",
    "BaseModel",
    "create_model",
    "PreprocessedData",
    "PreprocessorPipeline",
    "build_and_fit_pipeline",
    "SplitData",
    "create_splits",
    "ExperimentExecution",
    "ExperimentResult",
    "ModelResult",
    "compute_config_hash",
    "execute_experiment",
    "inspect_dataset_profile",
    "run_experiment",
    "train_and_evaluate_model",
    "StructuredTrainingResult",
    "train_model",
]
