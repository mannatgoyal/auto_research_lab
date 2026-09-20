"""Typed research context and state object for Phase 5 orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid
from pydantic import BaseModel, Field


class ResearchStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    COMPARING = "COMPARING"
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED = "FAILED"


class BudgetConfig(BaseModel):
    max_tool_calls: int = Field(default=15, ge=1, description="Maximum total tool calls permitted.")
    max_iterations: int = Field(default=10, ge=1, description="Maximum planning-execution loop iterations.")
    max_failed_tool_calls: int = Field(default=3, ge=1, description="Maximum failed tool calls before aborting.")
    timeout_seconds: float = Field(default=300.0, ge=5.0, description="Overall orchestrator runtime timeout.")


class ToolCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    success: bool = True
    duration_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ResearchState(BaseModel):
    research_id: str = Field(default_factory=lambda: f"res_{uuid.uuid4().hex[:12]}")
    original_question: str
    plan: Optional[Dict[str, Any]] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    evidence_store: Dict[str, Any] = Field(default_factory=dict)
    dataset_fingerprint: Optional[str] = None
    config_hashes: Dict[str, str] = Field(default_factory=dict)
    run_ids: Dict[str, str] = Field(default_factory=dict)
    iteration_count: int = 0
    failed_calls_count: int = 0
    status: ResearchStatus = ResearchStatus.INITIALIZED
    errors: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_tool_call(self, record: ToolCallRecord) -> None:
        """Appends a tool call record and updates metrics."""
        self.tool_calls.append(record)
        if not record.success:
            self.failed_calls_count += 1
            if record.error:
                self.errors.append(f"Tool '{record.tool_name}' failed: {record.error}")

    def is_budget_exceeded(self, budget: BudgetConfig) -> Tuple[bool, Optional[str]]:
        """Checks whether any guardrail budget has been exhausted."""
        if len(self.tool_calls) >= budget.max_tool_calls:
            return True, f"Max tool calls exceeded ({len(self.tool_calls)} >= {budget.max_tool_calls})"
        if self.iteration_count >= budget.max_iterations:
            return True, f"Max iterations exceeded ({self.iteration_count} >= {budget.max_iterations})"
        if self.failed_calls_count >= budget.max_failed_tool_calls:
            return True, f"Max failed tool calls exceeded ({self.failed_calls_count} >= {budget.max_failed_tool_calls})"
        return False, None

    def update_status(self, new_status: ResearchStatus) -> None:
        self.status = new_status
