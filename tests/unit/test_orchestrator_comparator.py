"""Unit tests for EvidenceComparator and Zero-Hallucination verification."""

import pytest
from orchestrator.comparator import EvidenceComparator
from orchestrator.schemas import ExperimentStep, Hypothesis, ResearchPlan
from orchestrator.state import ResearchState, ToolCallRecord


@pytest.fixture
def sample_plan():
    return ResearchPlan(
        research_question="Compare LR and RF",
        dataset_name="breast_cancer",
        candidate_models=["logistic_regression", "random_forest"],
        primary_metric="balanced_accuracy",
        secondary_metric="p95_latency_ms",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="random_forest achieves higher balanced_accuracy than logistic_regression.",
                target_metric="balanced_accuracy",
                comparator=">",
                model_a="logistic_regression",
                model_b="random_forest",
            ),
            Hypothesis(
                id="H2",
                statement="logistic_regression exhibits lower p95_latency_ms than random_forest.",
                target_metric="p95_latency_ms",
                comparator="<",
                model_a="logistic_regression",
                model_b="random_forest",
            ),
        ],
    )


def test_zero_hallucination_extracts_verified_metrics(sample_plan):
    state = ResearchState(original_question=sample_plan.research_question)
    comparator = EvidenceComparator()

    # Add tool call records for LR and RF
    lr_call = ToolCallRecord(
        tool_name="train_model",
        arguments={"model_name": "logistic_regression"},
        result={
            "model_name": "logistic_regression",
            "test_metrics": {"balanced_accuracy": 0.965},
            "latency_benchmarks": [{"p95_latency_ms": 0.35}],
            "tracking": {"child_run_id": "run_lr_123"},
        },
        success=True,
    )
    rf_call = ToolCallRecord(
        tool_name="train_model",
        arguments={"model_name": "random_forest"},
        result={
            "model_name": "random_forest",
            "test_metrics": {"balanced_accuracy": 0.945},
            "latency_benchmarks": [{"p95_latency_ms": 12.5}],
            "tracking": {"child_run_id": "run_rf_456"},
        },
        success=True,
    )
    state.add_tool_call(lr_call)
    state.add_tool_call(rf_call)

    claims = comparator.extract_evidence_claims(sample_plan, state)
    assert len(claims) == 4  # 2 models * 2 metrics

    claims_map = {(c.model_name, c.metric_name): c for c in claims}
    lr_acc = claims_map[("logistic_regression", "balanced_accuracy")]
    assert lr_acc.status == "VERIFIED"
    assert lr_acc.value == 0.965
    assert lr_acc.provenance is not None
    assert lr_acc.provenance.call_id == lr_call.call_id
    assert lr_acc.provenance.run_id == "run_lr_123"

    rf_acc = claims_map[("random_forest", "balanced_accuracy")]
    assert rf_acc.status == "VERIFIED"
    assert rf_acc.value == 0.945

    # Comparisons
    comparisons = comparator.compare_models(sample_plan, claims)
    comp_map = {c.metric_name: c for c in comparisons}

    acc_comp = comp_map["balanced_accuracy"]
    assert acc_comp.status == "VERIFIED"
    assert acc_comp.value_a == 0.965
    assert acc_comp.value_b == 0.945
    assert acc_comp.delta == -0.02
    assert acc_comp.superior_model == "logistic_regression"

    lat_comp = comp_map["p95_latency_ms"]
    assert lat_comp.status == "VERIFIED"
    assert lat_comp.value_a == 0.35
    assert lat_comp.value_b == 12.5
    assert lat_comp.delta == 12.15
    assert lat_comp.superior_model == "logistic_regression"  # lower is better for latency!

    # Hypotheses evaluation
    hyp_evals = comparator.evaluate_hypotheses(sample_plan.hypotheses, comparisons)
    h1_eval = [h for h in hyp_evals if h.hypothesis_id == "H1"][0]
    assert h1_eval.supported is False  # RF was not higher than LR

    h2_eval = [h for h in hyp_evals if h.hypothesis_id == "H2"][0]
    assert h2_eval.supported is True  # LR was lower latency than RF


def test_zero_hallucination_refuses_to_invent_missing_metrics(sample_plan):
    # State has NO tool calls
    state = ResearchState(original_question=sample_plan.research_question)
    comparator = EvidenceComparator()

    claims = comparator.extract_evidence_claims(sample_plan, state)
    for c in claims:
        assert c.status == "UNAVAILABLE"
        assert c.value is None
        assert c.provenance is None

    comparisons = comparator.compare_models(sample_plan, claims)
    for comp in comparisons:
        assert comp.status == "UNAVAILABLE"
        assert comp.delta is None
        assert comp.superior_model is None

    hyp_evals = comparator.evaluate_hypotheses(sample_plan.hypotheses, comparisons)
    for h in hyp_evals:
        assert h.supported is None
        assert "UNAVAILABLE" in h.reasoning
