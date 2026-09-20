"""FastAPI production model serving server."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.config import ServingConfig
from api.schemas import (
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.service import (
    InferenceExecutionError,
    InferenceService,
    ModelNotLoadedError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global inference service instance
inference_service = InferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model on startup if config is available."""
    config_path = os.getenv("SERVING_CONFIG_PATH", "configs/serving_config.yaml")
    if os.path.exists(config_path):
        try:
            cfg = ServingConfig.from_yaml(config_path)
            inference_service.initialize(cfg)
            logger.info("Model loaded successfully on startup from %s", config_path)
        except Exception as e:
            logger.error("Failed to load model on startup: %s", e, exc_info=True)
    else:
        logger.warning(
            "Serving config file '%s' not found. Server starting in degraded (unloaded) state.",
            config_path,
        )
    yield


app = FastAPI(
    title="Autonomous ML Lab - Production Model Serving API",
    version="1.0.0",
    description="Production-oriented model serving API hosting tracked MLflow model artifacts with deterministic training-time preprocessing.",
    lifespan=lifespan,
)


@app.exception_handler(ModelNotLoadedError)
async def handle_model_not_loaded(_: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP(503),
        content={"detail": str(exc)},
    )


@app.exception_handler(InferenceExecutionError)
async def handle_inference_error(_: Request, exc: InferenceExecutionError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Model inference failed during computation."},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    """Formats validation errors cleanly without exposing Python internals."""
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        errors.append(f"{field_loc}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input schema validation error.",
            "errors": errors,
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Returns service health and indicates whether a model is loaded and ready for predictions.",
    tags=["Health"],
)
def get_health():
    resp = inference_service.health()
    if not resp.model_loaded:
        # Return 503 if service is running but model is not loaded
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=resp.model_dump(),
        )
    return resp


@app.get(
    "/v1/models/current",
    response_model=ModelMetadataResponse,
    summary="Current model lineage metadata",
    description="Returns cryptographic and MLflow tracking metadata for the active model artifact.",
    tags=["Model Metadata"],
)
def get_current_model():
    if not inference_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model artifact is currently loaded.",
        )
    return inference_service.get_metadata()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Model inference endpoint",
    description="Transforms raw input features via the exact training-time preprocessor and returns prediction and class probabilities.",
    tags=["Inference"],
)
def predict(request: PredictionRequest):
    if not inference_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is running, but no model artifact is loaded.",
        )
    return inference_service.predict(request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous ML Lab - Model Serving Server")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/serving_config.yaml",
        help="Path to serving YAML configuration file.",
    )
    parser.add_argument("--host", type=str, default=None, help="Host to bind server to.")
    parser.add_argument("--port", type=int, default=None, help="Port to bind server to.")

    args = parser.parse_args()

    os.environ["SERVING_CONFIG_PATH"] = args.config
    host = args.host or "127.0.0.1"
    port = args.port or 8000

    if os.path.exists(args.config):
        cfg = ServingConfig.from_yaml(args.config)
        host = args.host or cfg.host
        port = args.port or cfg.port

    logger.info("Starting server on http://%s:%d (config: %s)", host, port, args.config)
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
