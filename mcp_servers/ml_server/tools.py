"""Deterministic ML tool implementations for Model Context Protocol."""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from api.config import ServingConfig
from api.loader import ModelLoader
from api.schemas import BreastCancerFeatures, CLASS_LABELS
from mcp_servers.common.audit import audit_logger
from mcp_servers.common.errors import (
    MCPEngineError,
    MCPNotFoundError,
    MCPPermissionError,
    MCPToolError,
    MCPValidationError,
    format_error_dict,
)
from mcp_servers.common.permissions import PermissionTier, permission_registry
from mcp_servers.common.schemas import (
    CreateSplitInput,
    CreateSplitOutput,
    EvaluateModelInput,
    EvaluateModelOutput,
    GetExperimentResultInput,
    GetExperimentResultOutput,
    GetModelLineageInput,
    GetModelLineageOutput,
    InspectDatasetInput,
    InspectDatasetOutput,
    MeasureLatencyInput,
    MeasureLatencyOutput,
    PredictInput,
    PredictOutput,
    TrainModelInput,
    TrainModelOutput,
)
import ml
from ml.config import DatasetConfig, LatencyConfig, SplitConfig
from ml.datasets.loader import load_dataset
from ml.evaluation.latency import benchmark_inference_latency
from ml.evaluation.metrics import evaluate_predictions
from ml.preprocessing.splitters import create_splits
from ml.runner import inspect_dataset_profile

logger = logging.getLogger(__name__)


def mcp_tool_wrapper(tool_name: str, permission: PermissionTier):
    """Decorator registering permissions, recording audit logs, and returning structured errors."""
    permission_registry.register(tool_name, permission)

    def decorator(fn: Callable[..., Any]):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # 1. Enforce caller permissions if explicitly passed in kwargs
            caller_tier = kwargs.pop("_caller_tier", PermissionTier.SAFE_WRITE)
            if not permission_registry.check_permission(tool_name, caller_tier):
                err = MCPPermissionError(
                    f"Permission denied: Caller tier '{caller_tier.value}' cannot execute "
                    f"tool '{tool_name}' which requires tier '{permission.value}'."
                )
                audit_logger.log(
                    tool_name=tool_name,
                    tier=permission,
                    input_args=kwargs,
                    success=False,
                    duration_ms=0.0,
                    error=err.to_dict(),
                )
                return format_error_dict(err)

            t0 = time.perf_counter()
            entity_id = None
            try:
                # 2. Execute deterministic function
                result = fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000.0

                # Extract entity ID if present
                if isinstance(result, dict):
                    entity_id = (
                        result.get("run_id")
                        or result.get("experiment_id")
                        or (result.get("tracking", {}).get("child_run_id") if isinstance(result.get("tracking"), dict) else None)
                    )

                # 3. Audit log success
                audit_logger.log(
                    tool_name=tool_name,
                    tier=permission,
                    input_args=kwargs,
                    success=True,
                    duration_ms=duration_ms,
                    entity_id=entity_id,
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                error_dict = format_error_dict(e)
                logger.error("Error executing MCP tool '%s': %s", tool_name, e, exc_info=True)
                audit_logger.log(
                    tool_name=tool_name,
                    tier=permission,
                    input_args=kwargs,
                    success=False,
                    duration_ms=duration_ms,
                    entity_id=entity_id,
                    error=error_dict["error"],
                )
                return error_dict

        return wrapper

    return decorator


# ==============================================================================
# Tool 1: inspect_dataset (READ_ONLY)
# ==============================================================================

@mcp_tool_wrapper("inspect_dataset", PermissionTier.READ_ONLY)
def tool_inspect_dataset(
    dataset_name: str = "breast_cancer",
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspects a dataset, computing descriptive statistics, distributions, and SHA-256 fingerprint."""
    # Sanitize dataset path/name
    if ".." in dataset_name:
        raise MCPValidationError("Directory traversal '..' is strictly forbidden.")

    profile = inspect_dataset_profile(dataset_name=dataset_name, target_column=target_column)
    p_dict = profile.to_dict()

    features_list = [f["name"] for f in p_dict["features"]]
    missing_dict = {f["name"]: f["missing_count"] for f in p_dict["features"]}
    output = InspectDatasetOutput(
        dataset_name=p_dict["dataset_name"],
        row_count=p_dict["row_count"],
        column_count=p_dict["column_count"],
        target_column=p_dict["target_column"],
        imbalance_ratio=p_dict["imbalance_ratio"],
        features=features_list,
        missing_values=missing_dict,
        class_distribution={str(k): int(v) for k, v in p_dict["target_counts"].items()},
        high_correlations=p_dict["high_correlation_pairs"],
        fingerprint_sha256=p_dict["fingerprint_sha256"],
    )
    return output.model_dump()


# ==============================================================================
# Tool 2: create_split (SAFE_WRITE)
# ==============================================================================

@mcp_tool_wrapper("create_split", PermissionTier.SAFE_WRITE)
def tool_create_split(
    dataset_name: str = "breast_cancer",
    target_column: Optional[str] = None,
    split_strategy: str = "stratified",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Creates deterministic, leak-free train/val/test splits with cryptographic partition hashes."""
    if test_size + val_size >= 1.0:
        raise MCPValidationError(
            f"test_size ({test_size}) + val_size ({val_size}) must be strictly < 1.0"
        )

    bundle = load_dataset(DatasetConfig(name=dataset_name, target_column=target_column))
    split_cfg = SplitConfig(
        strategy=split_strategy,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    split_data = create_splits(bundle, split_cfg)

    output = CreateSplitOutput(
        dataset_name=dataset_name,
        strategy=split_strategy,
        seed=seed,
        train_rows=split_data.train_rows,
        val_rows=split_data.val_rows,
        test_rows=split_data.test_rows,
        train_sha256=split_data.train_sha256,
        val_sha256=split_data.val_sha256,
        test_sha256=split_data.test_sha256,
        dataset_fingerprint=bundle.fingerprint_sha256,
        feature_names=split_data.feature_names,
        target_column=split_data.target_column,
    )
    return output.model_dump()


# ==============================================================================
# Tool 3: train_model (SAFE_WRITE)
# ==============================================================================

@mcp_tool_wrapper("train_model", PermissionTier.SAFE_WRITE)
def tool_train_model(
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
) -> Dict[str, Any]:
    """Trains a model candidate via the deterministic engine, logs to MLflow, and returns structured result."""
    # Directly delegates to ml.train_model() without duplicating ML logic
    res = ml.train_model(
        model_name=model_name,
        hyperparameters=hyperparameters,
        dataset_name=dataset_name,
        target_column=target_column,
        split_strategy=split_strategy,
        test_size=test_size,
        val_size=val_size,
        scaler=scaler,
        numeric_imputer=numeric_imputer,
        encode_categoricals=encode_categoricals,
        seed=seed,
        latency_batch_sizes=latency_batch_sizes,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        log_models=log_models,
    )

    output = TrainModelOutput(
        model_name=res.model_name,
        hyperparameters=res.hyperparameters,
        training_duration_seconds=res.training_duration_seconds,
        train_metrics=res.train_metrics,
        val_metrics=res.val_metrics,
        test_metrics=res.test_metrics,
        latency_benchmarks=res.latency_benchmarks,
        lineage=res.lineage,
        tracking=res.tracking,
    )
    return output.model_dump()


# ==============================================================================
# Tool 4: evaluate_model (READ_ONLY)
# ==============================================================================

@mcp_tool_wrapper("evaluate_model", PermissionTier.READ_ONLY)
def tool_evaluate_model(
    run_id: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    y_true: Optional[List[int]] = None,
    y_pred: Optional[List[int]] = None,
    y_prob: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Evaluates model performance: retrieves stored MLflow metrics or computes metrics on raw labels."""
    if run_id is not None:
        import mlflow
        from mlflow.tracking import MlflowClient
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        client = MlflowClient(tracking_uri)
        try:
            run = client.get_run(run_id)
        except Exception as e:
            raise MCPNotFoundError(f"Could not find MLflow run '{run_id}': {e}") from e

        metrics = run.data.metrics
        # Retrieve confusion matrix artifact if present
        cm_dict = None
        try:
            meta_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model_metadata.json", tracking_uri=tracking_uri
            )
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                cm_dict = {"matrix": meta.get("confusion_matrix_test")}
        except Exception:
            pass

        output = EvaluateModelOutput(
            accuracy=metrics.get("test_accuracy", 0.0),
            balanced_accuracy=metrics.get("test_balanced_accuracy", 0.0),
            precision_binary=metrics.get("test_precision_binary", 0.0),
            recall_binary=metrics.get("test_recall_binary", 0.0),
            f1_binary=metrics.get("test_f1_binary", 0.0),
            f1_macro=metrics.get("test_f1_macro", 0.0),
            f1_weighted=metrics.get("test_f1_weighted", 0.0),
            roc_auc=metrics.get("test_roc_auc"),
            pr_auc=metrics.get("test_pr_auc"),
            log_loss_val=metrics.get("test_log_loss"),
            confusion_matrix=cm_dict,
            source=f"mlflow_run:{run_id}",
        )
        return output.model_dump()

    elif y_true is not None and y_pred is not None:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        y_prob_arr = None
        if y_prob is not None:
            # Construct (N, 2) array for binary classification
            p1 = np.asarray(y_prob, dtype=float)
            p0 = 1.0 - p1
            y_prob_arr = np.column_stack([p0, p1])

        eval_res = evaluate_predictions(y_true_arr, y_pred_arr, y_prob_arr)
        cm_dict = {
            "true_negative": eval_res.confusion_matrix.true_negative,
            "false_positive": eval_res.confusion_matrix.false_positive,
            "false_negative": eval_res.confusion_matrix.false_negative,
            "true_positive": eval_res.confusion_matrix.true_positive,
            "matrix": eval_res.confusion_matrix.matrix,
        }
        output = EvaluateModelOutput(
            accuracy=eval_res.accuracy,
            balanced_accuracy=eval_res.balanced_accuracy,
            precision_binary=eval_res.precision_binary,
            recall_binary=eval_res.recall_binary,
            f1_binary=eval_res.f1_binary,
            f1_macro=eval_res.f1_macro,
            f1_weighted=eval_res.f1_weighted,
            roc_auc=eval_res.roc_auc,
            pr_auc=eval_res.pr_auc,
            log_loss_val=eval_res.log_loss_val,
            confusion_matrix=cm_dict,
            source="direct_evaluation",
        )
        return output.model_dump()

    raise MCPValidationError("Either 'run_id' or both ('y_true', 'y_pred') must be provided.")


# ==============================================================================
# Tool 5: measure_inference_latency (READ_ONLY)
# ==============================================================================

@mcp_tool_wrapper("measure_inference_latency", PermissionTier.READ_ONLY)
def tool_measure_inference_latency(
    run_id: str,
    batch_sizes: Optional[List[int]] = None,
    warmup_iterations: int = 25,
    benchmark_iterations: int = 100,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Dict[str, Any]:
    """Profiles microsecond inference latency across specified batch sizes on a loaded model artifact."""
    loader = ModelLoader(tracking_uri=tracking_uri)
    config = ServingConfig(model_source="mlflow", tracking_uri=tracking_uri, run_id=run_id)
    try:
        pipeline = loader.load_pipeline(config)
    except Exception as e:
        raise MCPEngineError(f"Failed to load model pipeline for latency benchmarking: {e}") from e

    # Generate test feature matrix matching input dimensionality
    input_dim = pipeline.metadata.get("transformed_features_count", 30)
    sample_features = np.random.randn(200, input_dim).astype(np.float32)

    latency_cfg = LatencyConfig(
        batch_sizes=batch_sizes or [1, 16],
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )

    benchmarks = benchmark_inference_latency(
        model=pipeline.model,
        sample_features=sample_features,
        config=latency_cfg,
    )

    output = MeasureLatencyOutput(
        run_id=run_id,
        model_name=pipeline.model_name,
        benchmarks=[b.to_dict() for b in benchmarks],
    )
    return output.model_dump()


# ==============================================================================
# Tool 6: get_experiment_result (READ_ONLY)
# ==============================================================================

@mcp_tool_wrapper("get_experiment_result", PermissionTier.READ_ONLY)
def tool_get_experiment_result(
    experiment_id: str,
    results_dir: str = "experiments/results",
    tracking_uri: Optional[str] = "sqlite:///mlflow.db",
) -> Dict[str, Any]:
    """Retrieves full machine-readable experiment evidence JSON from disk or MLflow."""
    if ".." in experiment_id or ".." in results_dir:
        raise MCPValidationError("Directory traversal '..' is strictly forbidden.")

    # 1. Try local file in results_dir
    file_path = os.path.join(results_dir, f"{experiment_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return GetExperimentResultOutput.model_validate(data).model_dump()

    # 2. Try querying MLflow parent run artifact
    if tracking_uri is not None:
        import mlflow
        from mlflow.tracking import MlflowClient
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        client = MlflowClient(tracking_uri)
        try:
            # Check if experiment_id is a run_id or experiment_id tag
            parent_run = None
            try:
                parent_run = client.get_run(experiment_id)
            except Exception:
                # Search by tag
                runs = client.search_runs(
                    experiment_ids=["0", "1", "2"],
                    filter_string=f"tags.experiment_id = '{experiment_id}'",
                )
                if runs:
                    parent_run = runs[0]

            if parent_run is not None:
                art_path = mlflow.artifacts.download_artifacts(
                    run_id=parent_run.info.run_id,
                    artifact_path="results.json",
                    tracking_uri=tracking_uri,
                )
                with open(art_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return GetExperimentResultOutput.model_validate(data).model_dump()
        except Exception:
            pass

    raise MCPNotFoundError(f"Experiment result '{experiment_id}' could not be located.")


# ==============================================================================
# Tool 7: get_model_lineage (READ_ONLY)
# ==============================================================================

@mcp_tool_wrapper("get_model_lineage", PermissionTier.READ_ONLY)
def tool_get_model_lineage(
    run_id: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Dict[str, Any]:
    """Traces a candidate child run to its parent experiment run and cryptographic lineage hashes."""
    import mlflow
    from mlflow.tracking import MlflowClient
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    client = MlflowClient(tracking_uri)

    try:
        child_run = client.get_run(run_id)
    except Exception as e:
        raise MCPNotFoundError(f"Could not find MLflow run with ID '{run_id}': {e}") from e

    parent_run_id = child_run.data.tags.get("parent_run_id")
    parent_tags = {}
    if parent_run_id:
        try:
            parent_run = client.get_run(parent_run_id)
            parent_tags = parent_run.data.tags
        except Exception:
            pass

    # Extract lineage
    config_hash = child_run.data.tags.get("config_hash_sha256", parent_tags.get("config_hash_sha256", "unknown"))
    dataset_fingerprint = child_run.data.tags.get("dataset_fingerprint", parent_tags.get("dataset_fingerprint", "unknown"))
    train_sha256 = child_run.data.tags.get("train_sha256", child_run.data.params.get("train_sha256", "unknown"))
    test_sha256 = child_run.data.tags.get("test_sha256", child_run.data.params.get("test_sha256", "unknown"))
    git_commit = child_run.data.tags.get("git_commit", parent_tags.get("git_commit", "untracked"))
    seed = int(child_run.data.params.get("seed", 42))

    # Extract hyperparameters
    hparams = {k[3:]: v for k, v in child_run.data.params.items() if k.startswith("hp_")}

    output = GetModelLineageOutput(
        run_id=run_id,
        parent_run_id=parent_run_id,
        model_name=child_run.data.tags.get("model_name", "unknown"),
        model_type=child_run.data.params.get("model_type", "unknown"),
        config_hash_sha256=config_hash,
        dataset_fingerprint=dataset_fingerprint,
        train_sha256=train_sha256,
        val_sha256=child_run.data.tags.get("val_sha256"),
        test_sha256=test_sha256,
        git_commit=git_commit,
        seed=seed,
        hyperparameters=hparams,
    )
    return output.model_dump()


# ==============================================================================
# Tool 8: predict (READ_ONLY)
# ==============================================================================

@mcp_tool_wrapper("predict", PermissionTier.READ_ONLY)
def tool_predict(
    features: Dict[str, float],
    run_id: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Dict[str, Any]:
    """Transforms raw input features via training-time preprocessor and returns model predictions."""
    # 1. Strict Pydantic input validation
    try:
        validated_features = BreastCancerFeatures(**features)
    except Exception as e:
        raise MCPValidationError(f"Invalid feature inputs: {e}") from e

    # 2. Resolve run ID (fallback to serving config if None)
    target_run_id = run_id
    if not target_run_id:
        cfg_path = "configs/serving_config.yaml"
        if os.path.exists(cfg_path):
            cfg = ServingConfig.from_yaml(cfg_path)
            target_run_id = cfg.run_id
            tracking_uri = cfg.tracking_uri
        else:
            raise MCPValidationError("No 'run_id' provided and 'configs/serving_config.yaml' not found.")

    # 3. Load model and preprocessor
    loader = ModelLoader(tracking_uri=tracking_uri)
    config = ServingConfig(model_source="mlflow", tracking_uri=tracking_uri, run_id=target_run_id)
    try:
        pipeline = loader.load_pipeline(config)
    except Exception as e:
        raise MCPEngineError(f"Failed to load inference pipeline: {e}") from e

    # 4. Predict
    t0 = time.perf_counter()
    df = validated_features.to_dataframe()
    pred_class, probs = pipeline.predict(df)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    output = PredictOutput(
        predicted_class=pred_class,
        predicted_label=CLASS_LABELS.get(pred_class, f"class_{pred_class}"),
        probabilities=probs,
        model_identity={
            "model_name": pipeline.model_name,
            "run_id": pipeline.run_id,
            "parent_run_id": pipeline.parent_run_id,
            "dataset_fingerprint": pipeline.metadata.get("dataset_fingerprint"),
            "config_hash_sha256": pipeline.metadata.get("config_hash_sha256"),
        },
        inference_latency_ms=round(latency_ms, 4),
    )
    return output.model_dump()
