"""End-to-end integration tests for ResearchOrchestrator, MCP execution, and zero-hallucination evidence."""

import json
import pytest

from mcp_servers.ml_server import create_ml_server
from orchestrator.client import MCPToolClient
from orchestrator.orchestrator import ResearchOrchestrator
from orchestrator.state import BudgetConfig


@pytest.fixture(scope="module")
def shared_mcp_server():
    return create_ml_server()


@pytest.mark.anyio
async def test_orchestrator_end_to_end_research_question(shared_mcp_server):
    """Verifies end-to-end research flow on user's exact benchmark question:
    'Compare logistic regression and random forest on the breast cancer dataset
     and determine which model has better balanced accuracy while also considering inference latency.'
    """
    client = MCPToolClient(server=shared_mcp_server)
    orchestrator = ResearchOrchestrator(client=client)

    question = (
        "Compare logistic regression and random forest on the breast cancer dataset "
        "and determine which model has better balanced accuracy while also considering inference latency."
    )

    result = await orchestrator.run(question)

    # 1. Status and Plan Verification
    assert result.status == "COMPLETED"
    assert result.question == question
    assert result.plan.dataset_name == "breast_cancer"
    assert "logistic_regression" in result.plan.candidate_models
    assert "random_forest" in result.plan.candidate_models
    assert result.steps_executed >= 4
    assert result.total_duration_ms > 0.0

    # 2. Evidence Claims Verification (Zero Hallucination)
    assert len(result.evidence_claims) >= 4  # 2 models * 2 metrics
    for claim in result.evidence_claims:
        assert claim.status == "VERIFIED"
        assert claim.value is not None
        assert claim.provenance is not None
        assert claim.provenance.call_id is not None
        assert claim.provenance.tool_name in ("train_model", "measure_inference_latency")

    # 3. Model Comparisons Verification
    assert len(result.comparisons) >= 2
    comp_map = {c.metric_name: c for c in result.comparisons}

    assert "balanced_accuracy" in comp_map
    acc_comp = comp_map["balanced_accuracy"]
    assert acc_comp.status == "VERIFIED"
    assert acc_comp.value_a is not None
    assert acc_comp.value_b is not None
    assert acc_comp.delta == round(acc_comp.value_b - acc_comp.value_a, 6)
    assert acc_comp.superior_model in ("logistic_regression", "random_forest", "TIE")

    assert "p95_latency_ms" in comp_map
    lat_comp = comp_map["p95_latency_ms"]
    assert lat_comp.status == "VERIFIED"
    assert lat_comp.superior_model == "logistic_regression"  # Logistic regression is faster than RF

    # 4. Hypothesis Evaluations
    assert len(result.hypothesis_evaluations) >= 2
    for h_eval in result.hypothesis_evaluations:
        assert h_eval.supported in (True, False)
        assert len(h_eval.evidence_references) >= 2
        assert "call_id:" in h_eval.reasoning

    # 5. Output Conclusion & Limitations
    assert len(result.conclusion) > 50
    assert "balanced_accuracy" in result.conclusion
    assert len(result.limitations) >= 2

    # 6. JSON Serializability
    dumped = result.model_dump()
    json_str = json.dumps(dumped)
    assert len(json_str) > 0
    reloaded = json.loads(json_str)
    assert reloaded["research_id"] == result.research_id


@pytest.mark.anyio
async def test_orchestrator_halts_cleanly_on_budget_exhaustion(shared_mcp_server):
    """Verifies that orchestrator cleanly respects tool-call budgets and prevents runaway loops."""
    client = MCPToolClient(server=shared_mcp_server)
    # Restrict budget to only 2 tool calls
    tight_budget = BudgetConfig(max_tool_calls=2, max_iterations=2)
    orchestrator = ResearchOrchestrator(client=client, budget=tight_budget)

    question = "Compare logistic regression and random forest on breast cancer balanced accuracy."
    result = await orchestrator.run(question)

    assert result.status == "BUDGET_EXHAUSTED"
    assert result.steps_executed == 2
    # Check that it did not crash and returned structured limitations
    assert any("budget" in lim.lower() or "missing" in lim.lower() for lim in result.limitations)


@pytest.mark.anyio
async def test_orchestrator_routes_through_mcp_not_direct_python(shared_mcp_server):
    """Verifies that orchestrator dispatches tools through MCP protocol client."""
    client = MCPToolClient(server=shared_mcp_server)
    orchestrator = ResearchOrchestrator(client=client)

    # Spy on client.call_tool
    call_log = []
    original_call = client.call_tool

    async def spied_call(tool_name, arguments, state=None):
        call_log.append(tool_name)
        return await original_call(tool_name, arguments, state=state)

    client.call_tool = spied_call

    question = "Compare logistic regression and random forest on accuracy."
    result = await orchestrator.run(question)

    assert result.status == "COMPLETED"
    assert len(call_log) >= 3
    assert "inspect_dataset" in call_log
    assert "create_split" in call_log
    assert "train_model" in call_log
