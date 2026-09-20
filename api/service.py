"""Inference service managing model lifecycle, server-side timing, and prediction execution."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from api.config import ServingConfig
from api.loader import InferencePipeline, ModelLoader
from api.schemas import (
    CLASS_LABELS,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when an inference request is made but no model is loaded."""
    pass


class InferenceExecutionError(Exception):
    """Raised when an internal error occurs during model prediction."""
    pass


class InferenceService:
    """Singleton-ready service maintaining the active InferencePipeline."""

    def __init__(self, loader: Optional[ModelLoader] = None) -> None:
        self.loader = loader or ModelLoader()
        self.pipeline: Optional[InferencePipeline] = None
        self.config: Optional[ServingConfig] = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def initialize(self, config: ServingConfig) -> None:
        """Loads and initializes the inference pipeline using the provided configuration."""
        self.config = config
        self.loader = ModelLoader(tracking_uri=config.tracking_uri)
        logger.info("Initializing InferenceService with run_id: %s", config.run_id)
        self.pipeline = self.loader.load_pipeline(config)
        logger.info("InferenceService initialized successfully. Active model: %s", self.pipeline.model_name)

    def health(self) -> HealthResponse:
        """Returns current service health and model status."""
        if not self.is_loaded:
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                model_name=None,
                run_id=None,
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=self.pipeline.model_name,
            run_id=self.pipeline.run_id,
        )

    def get_metadata(self) -> ModelMetadataResponse:
        """Returns lineage and configuration metadata for the loaded model."""
        if not self.is_loaded:
            raise ModelNotLoadedError("No model is currently loaded in the inference service.")

        meta = self.pipeline.metadata
        return ModelMetadataResponse(
            model_name=str(meta.get("model_name", "unknown")),
            run_id=str(meta.get("run_id", "unknown")),
            parent_run_id=meta.get("parent_run_id"),
            experiment_name=meta.get("experiment_name"),
            model_type=str(meta.get("model_type", "unknown")),
            dataset_fingerprint=str(meta.get("dataset_fingerprint", "unknown")),
            config_hash_sha256=str(meta.get("config_hash_sha256", "unknown")),
            git_commit=str(meta.get("git_commit", "untracked")),
            expected_features=meta.get("expected_features", []),
            transformed_features_count=int(meta.get("transformed_features_count", 0)),
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Executes end-to-end inference on the validated request and measures server-side latency."""
        if not self.is_loaded:
            raise ModelNotLoadedError("Inference service is running, but no model artifact is loaded.")

        # 1. High-resolution server-side latency measurement
        t0 = time.perf_counter()

        try:
            # 2. Convert features to DataFrame
            df = request.features.to_dataframe()

            # 3. Transform via training-time preprocessor and predict
            pred_class, probs = self.pipeline.predict(df)
        except Exception as e:
            logger.error("Inference execution error: %s", e, exc_info=True)
            raise InferenceExecutionError(f"Model inference failed during computation: {e}") from e

        # Server-side inference latency in milliseconds
        latency_ms = (time.perf_counter() - t0) * 1000.0

        label = CLASS_LABELS.get(pred_class, f"class_{pred_class}")

        return PredictionResponse(
            predicted_class=pred_class,
            predicted_label=label,
            probabilities=probs,
            model_identity={
                "model_name": self.pipeline.model_name,
                "run_id": self.pipeline.run_id,
                "parent_run_id": self.pipeline.parent_run_id,
                "dataset_fingerprint": self.pipeline.metadata.get("dataset_fingerprint"),
                "config_hash_sha256": self.pipeline.metadata.get("config_hash_sha256"),
            },
            inference_latency_ms=round(latency_ms, 4),
        )
