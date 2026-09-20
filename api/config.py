"""Configuration models for the model serving and inference API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class ServingConfig(BaseModel):
    """Configuration for loading and serving a tracked model artifact."""

    model_source: str = Field(
        default="mlflow",
        description="Source of the model artifact (e.g. 'mlflow').",
    )
    tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking URI (e.g. 'sqlite:///mlflow.db').",
    )
    run_id: str = Field(
        ...,
        description="MLflow child model run ID to load for serving.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host interface to bind the FastAPI server to.",
    )
    port: int = Field(
        default=8000,
        description="Port to bind the FastAPI server to.",
    )
    service_name: str = Field(
        default="autonomous-ml-inference-service",
        description="Logical name of the inference service.",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_nested_serving(cls, data: Any) -> Any:
        """Allows config with a top-level 'serving' key or direct attributes."""
        if isinstance(data, dict):
            if "serving" in data and isinstance(data["serving"], dict):
                return data["serving"]
        return data

    @classmethod
    def from_yaml(cls, path_or_content: str | Path) -> ServingConfig:
        """Loads ServingConfig from a YAML file or string."""
        if isinstance(path_or_content, Path) or (
            isinstance(path_or_content, str) and os.path.exists(path_or_content)
        ):
            with open(path_or_content, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        else:
            raw = yaml.safe_load(path_or_content)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML configuration: expected dict, got {type(raw)}")
        return cls.model_validate(raw)
