"""Structured configuration definitions for deterministic ML experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
import yaml


class DatasetConfig(BaseModel):
    name: str = Field(..., description="Dataset name or path to CSV/Parquet file.")
    target_column: Optional[str] = Field(
        default=None, description="Target column name. If None, uses default for dataset."
    )
    source: Optional[str] = Field(
        default=None, description="Source description or URL of the dataset."
    )


class SplitConfig(BaseModel):
    strategy: Literal["stratified", "random"] = Field(
        default="stratified", description="Splitting strategy to partition data."
    )
    test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Proportion for test set.")
    val_size: float = Field(default=0.1, ge=0.0, le=0.5, description="Proportion for validation set.")
    seed: int = Field(default=42, description="Random seed for deterministic splitting.")

    @model_validator(mode="after")
    def validate_split_sizes(self) -> SplitConfig:
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"Sum of test_size ({self.test_size}) and val_size ({self.val_size}) must be < 1.0"
            )
        return self


class PreprocessingConfig(BaseModel):
    numeric_imputer: Literal["median", "mean", "zero"] = Field(
        default="median", description="Strategy to impute missing numeric values."
    )
    categorical_imputer: Literal["most_frequent", "constant"] = Field(
        default="most_frequent", description="Strategy to impute missing categorical values."
    )
    scaler: Literal["standard", "robust", "minmax", "none"] = Field(
        default="standard", description="Scaling method for continuous numeric features."
    )
    encode_categoricals: bool = Field(
        default=True, description="Whether to one-hot encode categorical features."
    )


class ModelConfig(BaseModel):
    name: Literal["logistic_regression", "random_forest", "xgboost", "torch_mlp"] = Field(
        ..., description="Supported model architecture type."
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Architecture & training hyperparameters."
    )


class LatencyConfig(BaseModel):
    batch_sizes: List[int] = Field(
        default=[1, 16], description="List of batch sizes to benchmark."
    )
    warmup_iterations: int = Field(
        default=50, ge=5, description="Warm-up inferences before taking measurements."
    )
    benchmark_iterations: int = Field(
        default=500, ge=20, description="Number of timed benchmark loops."
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Compute device for latency benchmarking."
    )


class ExperimentConfig(BaseModel):
    name: str = Field(..., description="Unique human-readable experiment identifier.")
    description: Optional[str] = Field(default=None, description="Goal or hypothesis.")
    seed: int = Field(default=42, description="Global random seed.")
    dataset: DatasetConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: List[ModelConfig] = Field(..., min_length=1)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)

    @classmethod
    def from_yaml(cls, path_or_content: str) -> ExperimentConfig:
        """Loads configuration from YAML file or raw YAML string."""
        try:
            with open(path_or_content, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, FileNotFoundError):
            data = yaml.safe_load(path_or_content)
        if not isinstance(data, dict):
            raise ValueError(f"YAML content must deserialize to a dict, got {type(data)}")
        if "experiment" in data and isinstance(data["experiment"], dict):
            data = data["experiment"]
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serializes configuration to YAML string."""
        return yaml.dump(self.model_dump(), sort_keys=False)
