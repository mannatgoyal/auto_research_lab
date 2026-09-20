"""CLI entrypoint for running deterministic experiments with optional MLflow tracking."""

from __future__ import annotations

import argparse
import sys
from ml.config import ExperimentConfig
from ml.runner import execute_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous ML Research Lab - Deterministic ML Runner & Lineage Tracker"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="experiments/results",
        help="Directory to save structured JSON experiment results.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        default=False,
        help="Disable MLflow tracking layer completely.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking store URI (default: sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (defaults to experiment name from config).",
    )
    parser.add_argument(
        "--no-model-artifacts",
        action="store_true",
        default=False,
        help="Skip logging binary model weights to MLflow.",
    )

    args = parser.parse_args()

    try:
        config = ExperimentConfig.from_yaml(args.config)
        print(f"\n[Autonomous ML Lab] Loading experiment config: '{config.name}' from {args.config}...")

        # 1. Execute deterministic experiment (Pure computation)
        execution = execute_experiment(config, output_dir=args.output_dir)
        print("\n" + execution.result.summary_table() + "\n")
        print(f"[Autonomous ML Lab] Experiment completed! Results saved to '{args.output_dir}/{execution.result.experiment_id}.json'.")

        # 2. Optional MLflow Tracking (Evidence & Lineage Observer)
        if not args.no_tracking:
            from ml.tracking import MLflowTracker
            tracker = MLflowTracker(
                tracking_uri=args.tracking_uri,
                experiment_name=args.experiment_name or config.name,
            )
            run_id = tracker.log_execution(
                execution,
                log_models=not args.no_model_artifacts,
            )
            print(f"[MLflow Tracking] Logged to MLflow (Parent Run ID: {run_id}) at '{tracker.tracking_uri}'.\n")
        else:
            print("[MLflow Tracking] Tracking disabled (--no-tracking). Skipping MLflow logging.\n")

        sys.exit(0)
    except Exception as e:
        print(f"\n[Autonomous ML Lab Error] Execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
