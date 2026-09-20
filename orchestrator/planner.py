"""Research question parser and structured experiment planner."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from orchestrator.schemas import ExperimentStep, Hypothesis, ResearchPlan


class ResearchPlanner:
    """Parses natural-language research questions into structured experiment plans."""

    SUPPORTED_MODELS: Dict[str, List[str]] = {
        "logistic_regression": ["logistic regression", "logistic_regression", "logreg", "lr"],
        "random_forest": ["random forest", "random_forest", "rf"],
        "xgboost": ["xgboost", "xgb", "gradient boosting"],
        "torch_mlp": ["neural network", "mlp", "torch_mlp", "multilayer perceptron", "pytorch"],
    }

    SUPPORTED_DATASETS: Dict[str, List[str]] = {
        "breast_cancer": ["breast cancer", "breast_cancer", "cancer", "wisconsin"],
    }

    def plan(self, question: str) -> ResearchPlan:
        """Translates a research question into a structured ResearchPlan."""
        q_lower = question.lower()

        # 1. Detect candidate models
        detected_models: List[str] = []
        for canonical_name, aliases in self.SUPPORTED_MODELS.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                    if canonical_name not in detected_models:
                        detected_models.append(canonical_name)
                    break

        if len(detected_models) < 2:
            # Fallback to standard baseline pair if underspecified
            if len(detected_models) == 1:
                alt = "random_forest" if detected_models[0] != "random_forest" else "logistic_regression"
                detected_models.append(alt)
            else:
                detected_models = ["logistic_regression", "random_forest"]

        # 2. Detect dataset
        dataset_name = "breast_cancer"
        for canonical_ds, aliases in self.SUPPORTED_DATASETS.items():
            for alias in aliases:
                if alias in q_lower:
                    dataset_name = canonical_ds
                    break

        # 3. Detect primary and secondary metrics
        primary_metric = "balanced_accuracy"
        if "f1" in q_lower:
            primary_metric = "f1_binary"
        elif "accuracy" in q_lower and "balanced" not in q_lower:
            primary_metric = "accuracy"

        secondary_metric = "p95_latency_ms" if "latency" in q_lower or "speed" in q_lower or "inference" in q_lower else None

        # 4. Formulate testable hypotheses
        hypotheses: List[Hypothesis] = []
        model_a = detected_models[0]
        model_b = detected_models[1]

        hypotheses.append(
            Hypothesis(
                id="H1",
                statement=f"{model_b} achieves higher {primary_metric} than {model_a}.",
                target_metric=primary_metric,
                comparator=">",
                model_a=model_a,
                model_b=model_b,
                predicted_winner=model_b,
            )
        )

        if secondary_metric:
            hypotheses.append(
                Hypothesis(
                    id="H2",
                    statement=f"{model_a} exhibits lower {secondary_metric} (faster inference) than {model_b}.",
                    target_metric=secondary_metric,
                    comparator="<",
                    model_a=model_a,
                    model_b=model_b,
                    predicted_winner=model_a,
                )
            )

        # 5. Build experiment steps
        steps: List[ExperimentStep] = []
        step_idx = 1

        # Step 1: inspect_dataset
        steps.append(
            ExperimentStep(
                step_id=step_idx,
                tool_name="inspect_dataset",
                arguments={"dataset_name": dataset_name},
                purpose=f"Inspect shape, feature count, distributions, and fingerprint for '{dataset_name}'.",
            )
        )
        step_idx += 1

        # Step 2: create_split
        steps.append(
            ExperimentStep(
                step_id=step_idx,
                tool_name="create_split",
                arguments={
                    "dataset_name": dataset_name,
                    "split_strategy": "stratified",
                    "test_size": 0.2,
                    "val_size": 0.1,
                    "seed": 42,
                },
                purpose=f"Generate deterministic stratified train/val/test splits for '{dataset_name}'.",
            )
        )
        step_idx += 1

        # Step 3..N: Train candidate models
        for m in detected_models:
            steps.append(
                ExperimentStep(
                    step_id=step_idx,
                    tool_name="train_model",
                    arguments={
                        "model_name": m,
                        "dataset_name": dataset_name,
                        "split_strategy": "stratified",
                        "test_size": 0.2,
                        "val_size": 0.1,
                        "seed": 42,
                    },
                    purpose=f"Train model '{m}' with leak-free preprocessing and evaluate test metrics.",
                    model_target=m,
                )
            )
            step_idx += 1

        # Step N+1..: Benchmark latency if requested
        if secondary_metric:
            for m in detected_models:
                steps.append(
                    ExperimentStep(
                        step_id=step_idx,
                        tool_name="measure_inference_latency",
                        arguments={
                            "batch_sizes": [1],
                            "warmup_iterations": 10,
                            "benchmark_iterations": 50,
                        },
                        purpose=f"Benchmark inference latency percentiles for '{m}'.",
                        model_target=m,
                    )
                )
                step_idx += 1

        required_tools = sorted(list({s.tool_name for s in steps}))

        return ResearchPlan(
            research_question=question,
            dataset_name=dataset_name,
            candidate_models=detected_models,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            hypotheses=hypotheses,
            steps=steps,
            required_tools=required_tools,
            constraints={"no_data_leakage": True, "seed": 42},
        )
