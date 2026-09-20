"""Pydantic schemas for request validation, responses, and lineage metadata."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FEATURE_NAMES_30: List[str] = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

# Mapping between binary targets and semantic labels for breast cancer
CLASS_LABELS: Dict[int, str] = {
    0: "malignant",
    1: "benign",
}


class BreastCancerFeatures(BaseModel):
    """Strict schema for the 30 continuous features of UCI Breast Cancer Wisconsin.

    Enforces:
    - All 30 features must be present.
    - Features must be finite float numbers (rejecting NaN, Inf, -Inf).
    - Rejects any extra or unexpected fields (extra='forbid').
    - Supports both original space-separated names ("mean radius") and snake_case ("mean_radius").
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    mean_radius: float = Field(..., alias="mean radius")
    mean_texture: float = Field(..., alias="mean texture")
    mean_perimeter: float = Field(..., alias="mean perimeter")
    mean_area: float = Field(..., alias="mean area")
    mean_smoothness: float = Field(..., alias="mean smoothness")
    mean_compactness: float = Field(..., alias="mean compactness")
    mean_concavity: float = Field(..., alias="mean concavity")
    mean_concave_points: float = Field(..., alias="mean concave points")
    mean_symmetry: float = Field(..., alias="mean symmetry")
    mean_fractal_dimension: float = Field(..., alias="mean fractal dimension")

    radius_error: float = Field(..., alias="radius error")
    texture_error: float = Field(..., alias="texture error")
    perimeter_error: float = Field(..., alias="perimeter error")
    area_error: float = Field(..., alias="area error")
    smoothness_error: float = Field(..., alias="smoothness error")
    compactness_error: float = Field(..., alias="compactness error")
    concavity_error: float = Field(..., alias="concavity error")
    concave_points_error: float = Field(..., alias="concave points error")
    symmetry_error: float = Field(..., alias="symmetry error")
    fractal_dimension_error: float = Field(..., alias="fractal dimension error")

    worst_radius: float = Field(..., alias="worst radius")
    worst_texture: float = Field(..., alias="worst texture")
    worst_perimeter: float = Field(..., alias="worst perimeter")
    worst_area: float = Field(..., alias="worst area")
    worst_smoothness: float = Field(..., alias="worst smoothness")
    worst_compactness: float = Field(..., alias="worst compactness")
    worst_concavity: float = Field(..., alias="worst concavity")
    worst_concave_points: float = Field(..., alias="worst concave points")
    worst_symmetry: float = Field(..., alias="worst symmetry")
    worst_fractal_dimension: float = Field(..., alias="worst fractal dimension")

    @field_validator("*", mode="after")
    @classmethod
    def validate_finite_number(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError("Feature value must be a finite float (NaN and Inf are forbidden).")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Converts validated features to a single-row DataFrame with exact training column order."""
        row = {
            "mean radius": self.mean_radius,
            "mean texture": self.mean_texture,
            "mean perimeter": self.mean_perimeter,
            "mean area": self.mean_area,
            "mean smoothness": self.mean_smoothness,
            "mean compactness": self.mean_compactness,
            "mean concavity": self.mean_concavity,
            "mean concave points": self.mean_concave_points,
            "mean symmetry": self.mean_symmetry,
            "mean fractal dimension": self.mean_fractal_dimension,
            "radius error": self.radius_error,
            "texture error": self.texture_error,
            "perimeter error": self.perimeter_error,
            "area error": self.area_error,
            "smoothness error": self.smoothness_error,
            "compactness error": self.compactness_error,
            "concavity error": self.concavity_error,
            "concave points error": self.concave_points_error,
            "symmetry error": self.symmetry_error,
            "fractal dimension error": self.fractal_dimension_error,
            "worst radius": self.worst_radius,
            "worst texture": self.worst_texture,
            "worst perimeter": self.worst_perimeter,
            "worst area": self.worst_area,
            "worst smoothness": self.worst_smoothness,
            "worst compactness": self.worst_compactness,
            "worst concavity": self.worst_concavity,
            "worst concave points": self.worst_concave_points,
            "worst symmetry": self.worst_symmetry,
            "worst fractal dimension": self.worst_fractal_dimension,
        }
        return pd.DataFrame([row], columns=FEATURE_NAMES_30)


class PredictionRequest(BaseModel):
    """Container request for inference prediction.

    Supports either:
    1. {"features": { ... 30 features ... }}
    2. Direct flat feature dictionary: { ... 30 features ... }
    """

    model_config = ConfigDict(extra="forbid")
    features: BreastCancerFeatures

    @model_validator(mode="before")
    @classmethod
    def allow_flat_or_nested(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "features" in data and isinstance(data["features"], dict):
                return data
            # Allow direct flat dictionary
            return {"features": data}
        return data


class PredictionResponse(BaseModel):
    """Structured response model for model inference."""

    predicted_class: int = Field(..., description="Predicted binary class integer (0 or 1).")
    predicted_label: str = Field(..., description="Human-readable label ('malignant' or 'benign').")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probability distribution across classes.",
    )
    model_identity: Dict[str, Any] = Field(
        ...,
        description="Cryptographic and MLflow metadata identifying the loaded model.",
    )
    inference_latency_ms: float = Field(
        ...,
        description="Server-side inference execution latency in milliseconds.",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="'ok' if model is loaded and ready, 'degraded' otherwise.")
    model_loaded: bool = Field(..., description="True if inference pipeline is active.")
    model_name: Optional[str] = Field(default=None, description="Name of the active candidate model.")
    run_id: Optional[str] = Field(default=None, description="MLflow child run ID.")


class ModelMetadataResponse(BaseModel):
    """Audit metadata for the currently active model."""

    model_name: str = Field(..., description="Model candidate name.")
    run_id: str = Field(..., description="MLflow child run ID.")
    parent_run_id: Optional[str] = Field(default=None, description="MLflow parent experiment run ID.")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name.")
    model_type: str = Field(..., description="Model architecture type.")
    dataset_fingerprint: str = Field(..., description="SHA-256 fingerprint of training dataset.")
    config_hash_sha256: str = Field(..., description="SHA-256 fingerprint of experiment configuration.")
    git_commit: str = Field(..., description="Git commit hash at experiment execution time.")
    expected_features: List[str] = Field(..., description="Ordered list of required feature names.")
    transformed_features_count: int = Field(..., description="Number of features after preprocessing.")
