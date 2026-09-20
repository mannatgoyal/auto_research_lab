"""Model Context Protocol (MCP) server for the Autonomous ML Research Lab."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from mcp_servers.ml_server.tools import (
    tool_create_split,
    tool_evaluate_model,
    tool_get_experiment_result,
    tool_get_model_lineage,
    tool_inspect_dataset,
    tool_measure_inference_latency,
    tool_predict,
    tool_train_model,
)

logger = logging.getLogger(__name__)


def create_ml_server() -> MCPServer:
    """Creates and configures the ML MCP server with all 8 deterministic tools registered."""
    server = MCPServer("autonomous-ml-lab")

    # 1. inspect_dataset (READ_ONLY)
    @server.tool(
        name="inspect_dataset",
        description="Inspects dataset schema, row/column counts, missing values, class balance, and SHA-256 fingerprint.",
    )
    def inspect_dataset(
        dataset_name: str = "breast_cancer",
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        return tool_inspect_dataset(dataset_name=dataset_name, target_column=target_column)

    # 2. create_split (SAFE_WRITE)
    @server.tool(
        name="create_split",
        description="Creates deterministic, leak-free train/validation/test partitions with cryptographic SHA-256 hashes.",
    )
    def create_split(
        dataset_name: str = "breast_cancer",
        target_column: Optional[str] = None,
        split_strategy: str = "stratified",
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return tool_create_split(
            dataset_name=dataset_name,
            target_column=target_column,
            split_strategy=split_strategy,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

    # 3. train_model (SAFE_WRITE)
    @server.tool(
        name="train_model",
        description="Trains a candidate model architecture using the deterministic engine, logs to MLflow, and returns structured result.",
    )
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
    ) -> Dict[str, Any]:
        return tool_train_model(
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

    # 4. evaluate_model (READ_ONLY)
    @server.tool(
        name="evaluate_model",
        description="Retrieves evaluated metrics & confusion matrices from a tracked MLflow run, or computes metrics on raw labels.",
    )
    def evaluate_model(
        run_id: Optional[str] = None,
        tracking_uri: str = "sqlite:///mlflow.db",
        y_true: Optional[List[int]] = None,
        y_pred: Optional[List[int]] = None,
        y_prob: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        return tool_evaluate_model(
            run_id=run_id,
            tracking_uri=tracking_uri,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

    # 5. measure_inference_latency (READ_ONLY)
    @server.tool(
        name="measure_inference_latency",
        description="Profiles microsecond-level inference latency across batch sizes on a loaded MLflow model artifact.",
    )
    def measure_inference_latency(
        run_id: str,
        batch_sizes: Optional[List[int]] = None,
        warmup_iterations: int = 25,
        benchmark_iterations: int = 100,
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> Dict[str, Any]:
        return tool_measure_inference_latency(
            run_id=run_id,
            batch_sizes=batch_sizes,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            tracking_uri=tracking_uri,
        )

    # 6. get_experiment_result (READ_ONLY)
    @server.tool(
        name="get_experiment_result",
        description="Retrieves full machine-readable experiment evidence JSON from local storage or MLflow.",
    )
    def get_experiment_result(
        experiment_id: str,
        results_dir: str = "experiments/results",
        tracking_uri: Optional[str] = "sqlite:///mlflow.db",
    ) -> Dict[str, Any]:
        return tool_get_experiment_result(
            experiment_id=experiment_id,
            results_dir=results_dir,
            tracking_uri=tracking_uri,
        )

    # 7. get_model_lineage (READ_ONLY)
    @server.tool(
        name="get_model_lineage",
        description="Traces a candidate model child run to parent run ID, dataset fingerprint, split hashes, config hash, and git commit.",
    )
    def get_model_lineage(
        run_id: str,
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> Dict[str, Any]:
        return tool_get_model_lineage(
            run_id=run_id,
            tracking_uri=tracking_uri,
        )

    # 8. predict (READ_ONLY)
    @server.tool(
        name="predict",
        description="Performs model inference on 30 continuous features, transforming via training-time preprocessor.",
    )
    def predict(
        features: Dict[str, float],
        run_id: Optional[str] = None,
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> Dict[str, Any]:
        return tool_predict(
            features=features,
            run_id=run_id,
            tracking_uri=tracking_uri,
        )

    return server


def main() -> None:
    """CLI entrypoint running the ML MCP Server over stdio."""
    server = create_ml_server()
    logger.info("Starting Autonomous ML Lab MCP Server over stdio...")
    try:
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        logger.info("MCP server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
