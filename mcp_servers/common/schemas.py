"""Typed Pydantic request and response schemas for all MCP tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# 1. inspect_dataset
# ==============================================================================

class InspectDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(
        default="breast_cancer",
        description="Dataset identifier ('breast_cancer' or path to local CSV/Parquet).",
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name if loading a custom dataset file.",
    )


class InspectDatasetOutput(BaseModel):
    success: bool = True
    dataset_name: str
    row_count: int
    column_count: int
    target_column: str
    imbalance_ratio: float
    features: List[str]
    missing_values: Dict[str, int]
    class_distribution: Dict[str, int]
    high_correlations: List[Dict[str, Any]]
    fingerprint_sha256: str


# ==============================================================================
# 2. create_split
# ==============================================================================

class CreateSplitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(
        default="breast_cancer",
        description="Name of standard dataset or filepath.",
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target label column name.",
    )
    split_strategy: str = Field(
        default="stratified",
        description="Partition strategy ('stratified' or 'random').",
    )
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Proportion allocated to test partition.",
    )
    val_size: float = Field(
        default=0.1,
        ge=0.0,
        le=0.3,
        description="Proportion allocated to validation partition.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible partitioning.",
    )


class CreateSplitOutput(BaseModel):
    success: bool = True
    dataset_name: str
    strategy: str
    seed: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_sha256: str
    val_sha256: str
    test_sha256: str
    dataset_fingerprint: str
    feature_names: List[str]
    target_column: str


# ==============================================================================
# 3. train_model
# ==============================================================================

class TrainModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(
        ...,
        description="Candidate model architecture ('logistic_regression', 'random_forest', 'xgboost', 'torch_mlp').",
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional architecture hyperparameters.",
    )
    dataset_name: str = Field(
        default="breast_cancer",
        description="Target dataset identifier.",
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name if loading a custom dataset file.",
    )
    split_strategy: str = Field(
        default="stratified",
        description="Partitioning strategy ('stratified' or 'random').",
    )
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    val_size: float = Field(default=0.1, ge=0.0, le=0.3)
    scaler: str = Field(
        default="standard",
        description="Scaling transformer ('standard', 'robust', 'minmax', 'none').",
    )
    numeric_imputer: str = Field(
        default="median",
        description="Imputer strategy ('median', 'mean', 'zero').",
    )
    encode_categoricals: bool = Field(default=True)
    seed: int = Field(default=42)
    latency_batch_sizes: Optional[List[int]] = Field(default=None)
    warmup_iterations: int = Field(default=25, ge=5)
    benchmark_iterations: int = Field(default=100, ge=20)
    tracking_uri: Optional[str] = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking store URI or None to disable tracking.",
    )
    experiment_name: Optional[str] = Field(default=None)
    log_models: bool = Field(default=True)


class TrainModelOutput(BaseModel):
    success: bool = True
    model_name: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float
    train_metrics: Dict[str, Any]
    val_metrics: Optional[Dict[str, Any]]
    test_metrics: Dict[str, Any]
    latency_benchmarks: List[Dict[str, Any]]
    lineage: Dict[str, Any]
    tracking: Dict[str, Any]


# ==============================================================================
# 4. evaluate_model
# ==============================================================================

class EvaluateModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: Optional[str] = Field(
        default=None,
        description="MLflow child run ID to retrieve evaluated metrics from.",
    )
    tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking store URI.",
    )
    y_true: Optional[List[int]] = Field(
        default=None,
        description="Ground truth binary labels (0 or 1) for direct metric calculation.",
    )
    y_pred: Optional[List[int]] = Field(
        default=None,
        description="Predicted binary labels (0 or 1) for direct metric calculation.",
    )
    y_prob: Optional[List[float]] = Field(
        default=None,
        description="Predicted class probabilities for positive class (1).",
    )


class EvaluateModelOutput(BaseModel):
    success: bool = True
    accuracy: float
    balanced_accuracy: float
    precision_binary: float
    recall_binary: float
    f1_binary: float
    f1_macro: float
    f1_weighted: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    log_loss_val: Optional[float] = None
    confusion_matrix: Optional[Dict[str, Any]] = None
    source: str = "direct_computation"


# ==============================================================================
# 5. measure_inference_latency
# ==============================================================================

class MeasureLatencyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(
        ...,
        description="MLflow child model run ID to load and benchmark.",
    )
    batch_sizes: Optional[List[int]] = Field(
        default=None,
        description="Batch sizes to benchmark (e.g. [1, 16]). Defaults to [1, 16].",
    )
    warmup_iterations: int = Field(default=25, ge=5)
    benchmark_iterations: int = Field(default=100, ge=20)
    tracking_uri: str = Field(default="sqlite:///mlflow.db")


class MeasureLatencyOutput(BaseModel):
    success: bool = True
    run_id: str
    model_name: str
    benchmarks: List[Dict[str, Any]]


# ==============================================================================
# 6. get_experiment_result
# ==============================================================================

class GetExperimentResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str = Field(
        ...,
        description="Experiment ID or run ID to retrieve results for.",
    )
    results_dir: str = Field(
        default="experiments/results",
        description="Local directory containing saved experiment JSON files.",
    )
    tracking_uri: Optional[str] = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking URI fallback.",
    )


class GetExperimentResultOutput(BaseModel):
    success: bool = True
    experiment_id: str
    experiment_name: str
    config_hash_sha256: Optional[str] = None
    timestamp: str
    seed: int
    environment: Dict[str, Any]
    dataset_profile: Dict[str, Any]
    split_info: Dict[str, Any]
    preprocessing_info: Dict[str, Any]
    model_results: List[Dict[str, Any]]


# ==============================================================================
# 7. get_model_lineage
# ==============================================================================

class GetModelLineageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(
        ...,
        description="MLflow child model run ID.",
    )
    tracking_uri: str = Field(default="sqlite:///mlflow.db")


class GetModelLineageOutput(BaseModel):
    success: bool = True
    run_id: str
    parent_run_id: Optional[str]
    model_name: str
    model_type: str
    config_hash_sha256: str
    dataset_fingerprint: str
    train_sha256: str
    val_sha256: Optional[str]
    test_sha256: str
    git_commit: str
    seed: int
    hyperparameters: Dict[str, Any]


# ==============================================================================
# 8. predict
# ==============================================================================

class PredictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping 30 feature names to numeric values.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="MLflow child model run ID. If None, uses default serving config.",
    )
    tracking_uri: str = Field(default="sqlite:///mlflow.db")


class PredictOutput(BaseModel):
    success: bool = True
    predicted_class: int
    predicted_label: str
    probabilities: Optional[Dict[str, float]]
    model_identity: Dict[str, Any]
    inference_latency_ms: float
