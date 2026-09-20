"""Unit tests for ResearchPlanner."""

import pytest
from orchestrator.planner import ResearchPlanner


@pytest.fixture
def planner():
    return ResearchPlanner()


def test_planner_parses_comparison_question(planner):
    q = (
        "Compare logistic regression and random forest on the breast cancer dataset "
        "and determine which model has better balanced accuracy while also considering inference latency."
    )
    plan = planner.plan(q)

    assert plan.research_question == q
    assert plan.dataset_name == "breast_cancer"
    assert "logistic_regression" in plan.candidate_models
    assert "random_forest" in plan.candidate_models
    assert plan.primary_metric == "balanced_accuracy"
    assert plan.secondary_metric == "p95_latency_ms"

    # Hypotheses
    assert len(plan.hypotheses) == 2
    h1 = plan.hypotheses[0]
    assert h1.target_metric == "balanced_accuracy"
    assert h1.model_a == "logistic_regression"
    assert h1.model_b == "random_forest"

    h2 = plan.hypotheses[1]
    assert h2.target_metric == "p95_latency_ms"
    assert h2.comparator == "<"

    # Steps
    tool_names = [s.tool_name for s in plan.steps]
    assert tool_names[0] == "inspect_dataset"
    assert tool_names[1] == "create_split"
    assert tool_names.count("train_model") == 2
    assert tool_names.count("measure_inference_latency") == 2

    # Required tools
    assert set(plan.required_tools) == {
        "inspect_dataset",
        "create_split",
        "train_model",
        "measure_inference_latency",
    }


def test_planner_handles_neural_network_and_xgboost(planner):
    q = "Compare xgboost and neural network on accuracy."
    plan = planner.plan(q)

    assert "xgboost" in plan.candidate_models
    assert "torch_mlp" in plan.candidate_models
    assert plan.primary_metric == "accuracy"
    assert plan.secondary_metric is None
    assert len(plan.hypotheses) == 1
    assert set(plan.required_tools) == {"inspect_dataset", "create_split", "train_model"}


def test_planner_fallback_for_single_model(planner):
    q = "Evaluate random forest on balanced accuracy."
    plan = planner.plan(q)

    # Automatically pairs with baseline logistic_regression
    assert len(plan.candidate_models) == 2
    assert "random_forest" in plan.candidate_models
    assert "logistic_regression" in plan.candidate_models
