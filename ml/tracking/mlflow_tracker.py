"""MLflow tracking adapter for the Autonomous ML Research Lab.

Strict Architectural Invariant:
MLflow is an observation and evidence layer. It consumes pre-computed ExperimentExecution
bundles produced by the deterministic engine. It NEVER calculates metrics, NEVER calls
predict(), and NEVER alters preprocessing, training, or model evaluation.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional
import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost

# Allow filesystem tracking stores in MLflow 3.x for local test directories
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from ml.runner import ExperimentExecution, ModelResult

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Records deterministic experiment execution evidence to a local MLflow tracking store."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        self.experiment_name = experiment_name or "autonomous_ml_lab"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_execution(
        self,
        execution: ExperimentExecution,
        log_models: bool = True,
    ) -> str:
        """Logs an ExperimentExecution bundle as a Parent Run with nested Child Runs for each model.

        Returns:
            The MLflow Parent Run ID.
        """
        res = execution.result

        # ==========================================
        # 1. PARENT RUN (Experiment Level Evidence)
        # ==========================================
        parent_run_name = res.experiment_name
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id

            # Parent Tags (Lineage Invariant Evidence)
            parent_tags = {
                "experiment_id": res.experiment_id,
                "config_hash_sha256": res.config_hash_sha256,
                "dataset_fingerprint": str(res.split_info.get("dataset_fingerprint", "")),
                "git_commit": str(res.environment.get("git_commit", "untracked")),
                "run_type": "experiment_parent",
                "python_version": str(res.environment.get("python_version", "")),
                "platform": str(res.environment.get("platform", "")),
            }
            mlflow.set_tags(parent_tags)

            # Parent Parameters
            parent_params = {
                "dataset_name": str(res.dataset_profile.get("dataset_name", "")),
                "dataset_rows": str(res.dataset_profile.get("row_count", "")),
                "dataset_columns": str(res.dataset_profile.get("column_count", "")),
                "target_column": str(res.dataset_profile.get("target_column", "")),
                "imbalance_ratio": str(res.dataset_profile.get("imbalance_ratio", "")),
                "split_strategy": str(res.split_info.get("strategy", "")),
                "split_seed": str(res.split_info.get("seed", "")),
                "train_rows": str(res.split_info.get("train_rows", "")),
                "val_rows": str(res.split_info.get("val_rows", "")),
                "test_rows": str(res.split_info.get("test_rows", "")),
                "train_sha256": str(res.split_info.get("train_sha256", "")),
                "val_sha256": str(res.split_info.get("val_sha256", "")),
                "test_sha256": str(res.split_info.get("test_sha256", "")),
                "preprocessing_scaler": str(res.preprocessing_info.get("scaler", "")),
                "preprocessing_imputer": str(res.preprocessing_info.get("numeric_imputer", "")),
                "transformed_features_count": str(res.preprocessing_info.get("transformed_features_count", "")),
            }
            mlflow.log_params(parent_params)

            # Parent Structured JSON Artifacts
            mlflow.log_dict(res.to_dict(), "results.json")
            mlflow.log_dict(res.dataset_profile, "dataset_metadata.json")
            mlflow.log_dict(res.split_info, "split_metadata.json")
            mlflow.log_dict(res.preprocessing_info, "preprocessing_metadata.json")
            mlflow.log_dict(res.environment, "environment.json")

            # Parent Preprocessor Artifact (Experiment-level fitted transformer)
            if execution.pipeline is not None:
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        prep_dir = os.path.join(tmp_dir, "preprocessing")
                        os.makedirs(prep_dir, exist_ok=True)
                        joblib.dump(execution.pipeline, os.path.join(prep_dir, "pipeline.joblib"))
                        mlflow.log_artifacts(prep_dir, artifact_path="preprocessing")
                except Exception as e:
                    logger.warning("Failed to log fitted preprocessing pipeline artifact: %s", e)

            # ==========================================
            # 2. CHILD RUNS (Per-Model Candidates)
            # ==========================================
            for model_res in res.model_results:
                self._log_child_model_run(
                    model_res=model_res,
                    parent_run_id=parent_run_id,
                    execution=execution,
                    log_models=log_models,
                )

        return parent_run_id

    def _log_child_model_run(
        self,
        model_res: ModelResult,
        parent_run_id: str,
        execution: ExperimentExecution,
        log_models: bool,
    ) -> str:
        """Logs an individual model candidate as a nested child run in MLflow."""
        res = execution.result

        with mlflow.start_run(run_name=model_res.model_name, nested=True) as child_run:
            child_run_id = child_run.info.run_id

            # Child Tags (Lineage Connection)
            child_tags = {
                "parent_run_id": parent_run_id,
                "model_name": model_res.model_name,
                "config_hash_sha256": res.config_hash_sha256,
                "dataset_fingerprint": str(res.split_info.get("dataset_fingerprint", "")),
                "train_sha256": str(res.split_info.get("train_sha256", "")),
                "test_sha256": str(res.split_info.get("test_sha256", "")),
                "git_commit": str(res.environment.get("git_commit", "untracked")),
                "run_type": "model_child",
            }
            mlflow.set_tags(child_tags)

            # Child Parameters: model type, seed, and flattened hyperparameters
            child_params: Dict[str, str] = {
                "model_type": model_res.model_name,
                "seed": str(res.seed),
            }
            for k, v in model_res.hyperparameters.items():
                child_params[f"hp_{k}"] = str(v)
            mlflow.log_params(child_params)

            # Child Metrics: Test metrics
            tm = model_res.test_metrics
            child_metrics: Dict[str, float] = {
                "test_accuracy": tm.accuracy,
                "test_balanced_accuracy": tm.balanced_accuracy,
                "test_precision_binary": tm.precision_binary,
                "test_recall_binary": tm.recall_binary,
                "test_f1_binary": tm.f1_binary,
                "test_f1_macro": tm.f1_macro,
                "test_f1_weighted": tm.f1_weighted,
                "training_duration_seconds": model_res.training_duration_seconds,
            }
            if tm.roc_auc is not None:
                child_metrics["test_roc_auc"] = tm.roc_auc
            if tm.pr_auc is not None:
                child_metrics["test_pr_auc"] = tm.pr_auc
            if tm.log_loss_val is not None:
                child_metrics["test_log_loss"] = tm.log_loss_val

            # Train metrics
            tr_m = model_res.train_metrics
            child_metrics["train_accuracy"] = tr_m.accuracy
            child_metrics["train_balanced_accuracy"] = tr_m.balanced_accuracy
            child_metrics["train_f1_binary"] = tr_m.f1_binary

            # Validation metrics (if present)
            if model_res.val_metrics is not None:
                vm = model_res.val_metrics
                child_metrics["val_accuracy"] = vm.accuracy
                child_metrics["val_balanced_accuracy"] = vm.balanced_accuracy
                child_metrics["val_f1_binary"] = vm.f1_binary
                if vm.roc_auc is not None:
                    child_metrics["val_roc_auc"] = vm.roc_auc

            # Latency metrics per batch size
            for lb in model_res.latency_benchmarks:
                bs = lb.batch_size
                child_metrics[f"latency_b{bs}_mean_ms"] = lb.mean_latency_ms
                child_metrics[f"latency_b{bs}_p50_ms"] = lb.median_latency_ms
                child_metrics[f"latency_b{bs}_p90_ms"] = lb.p90_latency_ms
                child_metrics[f"latency_b{bs}_p95_ms"] = lb.p95_latency_ms
                child_metrics[f"latency_b{bs}_p99_ms"] = lb.p99_latency_ms
                child_metrics[f"latency_b{bs}_min_ms"] = lb.min_latency_ms
                child_metrics[f"latency_b{bs}_max_ms"] = lb.max_latency_ms
                child_metrics[f"latency_b{bs}_throughput_samples_per_sec"] = lb.throughput_samples_per_sec

            mlflow.log_metrics(child_metrics)

            # Child Structured Artifacts
            model_metadata = {
                "model_name": model_res.model_name,
                "hyperparameters": model_res.hyperparameters,
                "training_duration_seconds": model_res.training_duration_seconds,
                "confusion_matrix_test": tm.confusion_matrix.matrix,
                "confusion_matrix_train": tr_m.confusion_matrix.matrix,
            }
            mlflow.log_dict(model_metadata, "model_metadata.json")

            # Model Artifact Logging
            if log_models and model_res.model_name in execution.fitted_models:
                fitted_wrapper = execution.fitted_models[model_res.model_name]
                try:
                    self._log_model_artifact(fitted_wrapper)
                except Exception as e:
                    logger.warning(
                        "Failed to log model artifact for %s: %s",
                        model_res.model_name,
                        e,
                    )

            return child_run_id

    def _log_model_artifact(self, model_wrapper: Any) -> None:
        """Serializes and logs model artifacts using MLflow flavor integrations and file artifacts."""
        name = model_wrapper.model_name

        # 1. MLflow Flavor Logging (Native Model Output)
        try:
            if name in ("logistic_regression", "random_forest"):
                mlflow.sklearn.log_model(model_wrapper._model, name="model")
            elif name == "xgboost":
                mlflow.xgboost.log_model(model_wrapper._model, name="model")
            elif name == "torch_mlp":
                import torch
                # In MLflow 3.x, pytorch flavor uses pt2 export format by default and requires input_example
                dummy_input = torch.randn(1, model_wrapper._input_dim, dtype=torch.float32).numpy()
                mlflow.pytorch.log_model(model_wrapper._net, name="model", input_example=dummy_input)
        except Exception as e:
            logger.warning("MLflow flavor log_model failed for %s: %s", name, e)

        # 2. Portable Binary Artifact in run's artifact directory (under 'model/')
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_dir = os.path.join(tmp_dir, "model")
                os.makedirs(model_dir, exist_ok=True)
                if name in ("logistic_regression", "random_forest"):
                    joblib.dump(model_wrapper._model, os.path.join(model_dir, "model.joblib"))
                elif name == "xgboost":
                    model_wrapper._model.save_model(os.path.join(model_dir, "model.json"))
                elif name == "torch_mlp":
                    import torch
                    torch.save(model_wrapper._net.state_dict(), os.path.join(model_dir, "model_weights.pt"))
                mlflow.log_artifacts(model_dir, artifact_path="model")
        except Exception as e:
            logger.warning("Saving model file artifact failed for %s: %s", name, e)
