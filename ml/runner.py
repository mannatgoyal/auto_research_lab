"""Experiment execution runner orchestrating deterministic data prep, training, and evaluation."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import sklearn
import torch
import xgboost

from ml.config import DatasetConfig, ExperimentConfig, LatencyConfig, ModelConfig
from ml.datasets.loader import load_dataset
from ml.datasets.profiler import DatasetProfile, profile_dataset
from ml.evaluation.latency import LatencyBenchmark, benchmark_inference_latency
from ml.evaluation.metrics import EvaluationMetrics, evaluate_predictions
from ml.models.base import BaseModel, create_model
from ml.preprocessing.pipeline import PreprocessedData, PreprocessorPipeline, build_and_fit_pipeline
from ml.preprocessing.splitters import create_splits


@dataclass
class ModelResult:
    model_name: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float
    train_metrics: EvaluationMetrics
    val_metrics: Optional[EvaluationMetrics]
    test_metrics: EvaluationMetrics
    latency_benchmarks: List[LatencyBenchmark]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "training_duration_seconds": round(self.training_duration_seconds, 4),
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict() if self.val_metrics else None,
            "test_metrics": self.test_metrics.to_dict(),
            "latency_benchmarks": [b.to_dict() for b in self.latency_benchmarks],
        }


@dataclass
class ExperimentResult:
    experiment_name: str
    experiment_id: str
    config_hash_sha256: str
    timestamp: str
    seed: int
    environment: Dict[str, str]
    dataset_profile: Dict[str, Any]
    split_info: Dict[str, Any]
    preprocessing_info: Dict[str, Any]
    model_results: List[ModelResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "config_hash_sha256": self.config_hash_sha256,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "environment": self.environment,
            "dataset_profile": self.dataset_profile,
            "split_info": self.split_info,
            "preprocessing_info": self.preprocessing_info,
            "model_results": [m.to_dict() for m in self.model_results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> str:
        """Returns a formatted comparison table of models."""
        lines = [
            f"=== Experiment: {self.experiment_name} (ID: {self.experiment_id}) ===",
            f"Config Hash: {self.config_hash_sha256[:16]}... | Seed: {self.seed}",
            f"Dataset Fingerprint: {self.split_info.get('dataset_fingerprint', 'N/A')[:16]}...",
            "-" * 88,
            f"{'Model':<22} | {'Test Acc':<9} | {'Test F1':<8} | {'ROC-AUC':<8} | {'p95 Lat (b=1)':<14} | {'Train (s)':<9}",
            "-" * 88,
        ]
        for m in self.model_results:
            roc_str = f"{m.test_metrics.roc_auc:.4f}" if m.test_metrics.roc_auc is not None else "N/A"
            p95_lat = "N/A"
            for lb in m.latency_benchmarks:
                if lb.batch_size == 1:
                    p95_lat = f"{lb.p95_latency_ms:.3f} ms"
                    break
            lines.append(
                f"{m.model_name:<22} | {m.test_metrics.accuracy:<9.4f} | {m.test_metrics.f1_binary:<8.4f} | "
                f"{roc_str:<8} | {p95_lat:<14} | {m.training_duration_seconds:<9.3f}"
            )
        lines.append("-" * 88)
        return "\n".join(lines)


def get_git_commit_sha() -> str:
    """Retrieves current Git commit hash or returns 'untracked'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "untracked"


def compute_config_hash(config: ExperimentConfig) -> str:
    """Computes a deterministic SHA-256 fingerprint of the experiment configuration."""
    hasher = hashlib.sha256()
    hasher.update(config.to_yaml().encode("utf-8"))
    return hasher.hexdigest()


def inspect_dataset_profile(
    dataset_name: str, target_column: Optional[str] = None
) -> DatasetProfile:
    """Inspects and profiles a dataset by name or path. Callable directly by MCP tools."""
    bundle = load_dataset(DatasetConfig(name=dataset_name, target_column=target_column))
    return profile_dataset(bundle)


@dataclass
class ExperimentExecution:
    """Runtime execution container separating JSON-safe evidence from live fitted model objects and preprocessor."""
    result: ExperimentResult
    fitted_models: Dict[str, BaseModel]
    pipeline: Optional[PreprocessorPipeline] = None


def train_and_evaluate_model(
    model_cfg: ModelConfig,
    preprocessed: PreprocessedData,
    latency_cfg: LatencyConfig,
    seed: int = 42,
    return_model: bool = False,
) -> ModelResult | Tuple[ModelResult, BaseModel]:
    """Trains, evaluates, and latency-profiles a single model. Callable directly by MCP tools."""
    model = create_model(model_cfg, seed=seed)

    # 1. Train with wall-clock timing
    t0 = time.perf_counter()
    has_val = len(preprocessed.y_val) > 0
    model.fit(
        X=preprocessed.X_train,
        y=preprocessed.y_train,
        X_val=preprocessed.X_val if has_val else None,
        y_val=preprocessed.y_val if has_val else None,
    )
    training_duration = time.perf_counter() - t0

    # 2. Evaluation
    train_pred = model.predict(preprocessed.X_train)
    train_proba = model.predict_proba(preprocessed.X_train)
    train_metrics = evaluate_predictions(preprocessed.y_train, train_pred, train_proba)

    val_metrics = None
    if has_val:
        val_pred = model.predict(preprocessed.X_val)
        val_proba = model.predict_proba(preprocessed.X_val)
        val_metrics = evaluate_predictions(preprocessed.y_val, val_pred, val_proba)

    test_pred = model.predict(preprocessed.X_test)
    test_proba = model.predict_proba(preprocessed.X_test)
    test_metrics = evaluate_predictions(preprocessed.y_test, test_pred, test_proba)

    # 3. Latency Microbenchmark
    latency_benchmarks = benchmark_inference_latency(
        model=model,
        sample_features=preprocessed.X_test,
        config=latency_cfg,
    )

    result = ModelResult(
        model_name=model_cfg.name,
        hyperparameters=model.get_params()["hyperparameters"],
        training_duration_seconds=training_duration,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        latency_benchmarks=latency_benchmarks,
    )

    if return_model:
        return result, model
    return result


def execute_experiment(
    config: ExperimentConfig, output_dir: Optional[str] = None
) -> ExperimentExecution:
    """Executes an experiment, returning both the JSON-safe ExperimentResult and fitted model objects."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = start_time.isoformat()
    experiment_id = f"{config.name}_{int(start_time.timestamp())}"
    config_hash = compute_config_hash(config)

    # 1. Environment metadata
    env_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "torch_version": torch.__version__,
        "git_commit": get_git_commit_sha(),
    }

    # 2. Dataset loading & profiling
    bundle = load_dataset(config.dataset)
    profile = profile_dataset(bundle)

    # 3. Deterministic Splitting
    split_data = create_splits(bundle, config.split)
    split_info = {
        "strategy": config.split.strategy,
        "seed": config.split.seed,
        "train_rows": split_data.train_rows,
        "val_rows": split_data.val_rows,
        "test_rows": split_data.test_rows,
        "dataset_fingerprint": bundle.fingerprint_sha256,
        "train_sha256": split_data.train_sha256,
        "val_sha256": split_data.val_sha256,
        "test_sha256": split_data.test_sha256,
    }

    # 4. Leak-free Preprocessing
    pipeline, preprocessed = build_and_fit_pipeline(split_data, config.preprocessing)
    prep_info = {
        "numeric_features": pipeline.numeric_features,
        "categorical_features": pipeline.categorical_features,
        "transformed_features_count": len(pipeline.transformed_feature_names),
        "scaler": config.preprocessing.scaler,
        "numeric_imputer": config.preprocessing.numeric_imputer,
    }

    # 5. Model Training & Evaluation
    model_results: List[ModelResult] = []
    fitted_models: Dict[str, BaseModel] = {}

    for model_cfg in config.models:
        res, model = train_and_evaluate_model(
            model_cfg=model_cfg,
            preprocessed=preprocessed,
            latency_cfg=config.latency,
            seed=config.seed,
            return_model=True,
        )
        model_results.append(res)
        fitted_models[model_cfg.name] = model

    result = ExperimentResult(
        experiment_name=config.name,
        experiment_id=experiment_id,
        config_hash_sha256=config_hash,
        timestamp=timestamp_str,
        seed=config.seed,
        environment=env_info,
        dataset_profile=profile.to_dict(),
        split_info=split_info,
        preprocessing_info=prep_info,
        model_results=model_results,
    )

    # Save artifact if directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{experiment_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())

    return ExperimentExecution(
        result=result,
        fitted_models=fitted_models,
        pipeline=pipeline,
    )


def run_experiment(config: ExperimentConfig, output_dir: Optional[str] = None) -> ExperimentResult:
    """Executes an end-to-end deterministic machine learning experiment returning the JSON-safe result."""
    execution = execute_experiment(config, output_dir=output_dir)
    return execution.result


@dataclass
class StructuredTrainingResult:
    """Clean, JSON-safe result returned by train_model tool."""
    model_name: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float
    train_metrics: Dict[str, Any]
    val_metrics: Optional[Dict[str, Any]]
    test_metrics: Dict[str, Any]
    latency_benchmarks: List[Dict[str, Any]]
    lineage: Dict[str, Any]
    tracking: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def train_model(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    dataset_name: str = "breast_cancer",
    target_column: Optional[str] = None,
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaler: str = "standard",
    numeric_imputer: str = "median",
    encode_categoricals: bool = True,
    seed: int = 42,
    latency_batch_sizes: Optional[List[int]] = None,
    warmup_iterations: int = 25,
    benchmark_iterations: int = 100,
    tracking_uri: Optional[str] = "sqlite:///mlflow.db",
    experiment_name: Optional[str] = None,
    log_models: bool = True,
) -> StructuredTrainingResult:
    """High-level standalone training entrypoint.

    Architectural Pattern:
        train_model(...)
               ↓
        deterministic engine
               ↓
        MLflow (evidence observation)
               ↓
        structured result (JSON-safe data contract)
    """
    from ml.config import (
        DatasetConfig,
        ExperimentConfig,
        LatencyConfig,
        ModelConfig,
        PreprocessingConfig,
        SplitConfig,
    )

    exp_name = experiment_name or f"{dataset_name}_{model_name}"
    config = ExperimentConfig(
        name=exp_name,
        seed=seed,
        dataset=DatasetConfig(name=dataset_name, target_column=target_column),
        split=SplitConfig(
            strategy=split_strategy,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        ),
        preprocessing=PreprocessingConfig(
            scaler=scaler,
            numeric_imputer=numeric_imputer,
            encode_categoricals=encode_categoricals,
        ),
        models=[
            ModelConfig(name=model_name, hyperparameters=hyperparameters or {}),
        ],
        latency=LatencyConfig(
            batch_sizes=latency_batch_sizes or [1, 16],
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        ),
    )

    # 1. Deterministic Engine execution
    execution = execute_experiment(config)
    res = execution.result
    model_res = res.model_results[0]

    # 2. Optional MLflow tracking
    parent_run_id = None
    child_run_id = None
    if tracking_uri is not None:
        from ml.tracking import MLflowTracker
        tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name=config.name)
        parent_run_id = tracker.log_execution(execution, log_models=log_models)

        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        children = client.search_runs(
            experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
            filter_string=f"tags.parent_run_id = '{parent_run_id}'",
        )
        if children:
            child_run_id = children[0].info.run_id

    # 3. Clean, machine-readable structured result
    lineage = {
        "config_hash_sha256": res.config_hash_sha256,
        "dataset_fingerprint": res.split_info.get("dataset_fingerprint"),
        "train_sha256": res.split_info.get("train_sha256"),
        "val_sha256": res.split_info.get("val_sha256"),
        "test_sha256": res.split_info.get("test_sha256"),
        "git_commit": res.environment.get("git_commit", "untracked"),
        "seed": res.seed,
    }
    tracking_info = {
        "tracking_uri": tracking_uri,
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
    }

    return StructuredTrainingResult(
        model_name=model_res.model_name,
        hyperparameters=model_res.hyperparameters,
        training_duration_seconds=model_res.training_duration_seconds,
        train_metrics=model_res.train_metrics.to_dict(),
        val_metrics=model_res.val_metrics.to_dict() if model_res.val_metrics else None,
        test_metrics=model_res.test_metrics.to_dict(),
        latency_benchmarks=[b.to_dict() for b in model_res.latency_benchmarks],
        lineage=lineage,
        tracking=tracking_info,
    )
