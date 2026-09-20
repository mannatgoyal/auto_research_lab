"""Unit tests for ResearchState and BudgetConfig."""

import pytest
from orchestrator.state import BudgetConfig, ResearchState, ResearchStatus, ToolCallRecord


def test_research_state_initialization():
    state = ResearchState(original_question="What is the accuracy of random forest?")
    assert state.research_id.startswith("res_")
    assert state.original_question == "What is the accuracy of random forest?"
    assert state.status == ResearchStatus.INITIALIZED
    assert len(state.tool_calls) == 0
    assert state.iteration_count == 0
    assert state.failed_calls_count == 0


def test_state_tool_call_recording():
    state = ResearchState(original_question="Test question")
    record = ToolCallRecord(
        tool_name="inspect_dataset",
        arguments={"dataset_name": "breast_cancer"},
        result={"row_count": 569},
        success=True,
        duration_ms=12.5,
    )
    state.add_tool_call(record)
    assert len(state.tool_calls) == 1
    assert state.failed_calls_count == 0

    fail_record = ToolCallRecord(
        tool_name="nonexistent_tool",
        arguments={},
        success=False,
        error="Tool not found",
    )
    state.add_tool_call(fail_record)
    assert len(state.tool_calls) == 2
    assert state.failed_calls_count == 1
    assert len(state.errors) == 1


def test_budget_exceeded_checks():
    budget = BudgetConfig(max_tool_calls=2, max_iterations=3, max_failed_tool_calls=2)
    state = ResearchState(original_question="Budget check")

    # Initial state: within budget
    exceeded, reason = state.is_budget_exceeded(budget)
    assert exceeded is False

    # Add 2 tool calls -> max tool calls hit
    state.add_tool_call(ToolCallRecord(tool_name="t1", arguments={}, success=True))
    state.add_tool_call(ToolCallRecord(tool_name="t2", arguments={}, success=True))
    exceeded, reason = state.is_budget_exceeded(budget)
    assert exceeded is True
    assert "Max tool calls exceeded" in reason

    # Test failed calls budget
    fail_budget = BudgetConfig(max_tool_calls=10, max_iterations=5, max_failed_tool_calls=2)
    fresh_state = ResearchState(original_question="Fail budget")
    fresh_state.add_tool_call(ToolCallRecord(tool_name="t1", arguments={}, success=False, error="err1"))
    fresh_state.add_tool_call(ToolCallRecord(tool_name="t2", arguments={}, success=False, error="err2"))
    exceeded, reason = fresh_state.is_budget_exceeded(fail_budget)
    assert exceeded is True
    assert "Max failed tool calls exceeded" in reason
