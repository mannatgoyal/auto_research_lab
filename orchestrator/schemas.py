"""Structured schemas for research planning, evidence claims, and conclusions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    id: str = Field(description="Unique identifier for the hypothesis (e.g. 'H1').")
    statement: str = Field(description="Natural-language description of the hypothesis.")
    target_metric: str = Field(description="Deterministic metric to be evaluated (e.g. 'balanced_accuracy').")
    comparator: str = Field(default=">", description="Comparison operator ('>', '<', '==').")
    model_a: str = Field(description="First model name.")
    model_b: str = Field(description="Second model name.")
    predicted_winner: Optional[str] = Field(default=None, description="Expected winning model name.")


class ExperimentStep(BaseModel):
    step_id: int = Field(description="Sequential step index.")
    tool_name: str = Field(description="Target MCP tool name to call.")
    arguments: Dict[str, Any] = Field(description="Input arguments for the tool.")
    purpose: str = Field(description="Rationale for executing this step.")
    model_target: Optional[str] = Field(default=None, description="Associated model if applicable.")


class ResearchPlan(BaseModel):
    research_question: str
    dataset_name: str = "breast_cancer"
    candidate_models: List[str]
    primary_metric: str = "balanced_accuracy"
    secondary_metric: Optional[str] = "p95_latency_ms"
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    steps: List[ExperimentStep] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    required_tools: List[str] = Field(default_factory=list)


class ProvenanceRef(BaseModel):
    tool_name: str
    call_id: str
    run_id: Optional[str] = None
    property_path: str
    timestamp: str


class EvidenceClaim(BaseModel):
    metric_name: str
    model_name: str
    value: Optional[float] = None
    status: str = "VERIFIED"  # "VERIFIED", "UNAVAILABLE", "ERROR"
    unit: Optional[str] = None
    provenance: Optional[ProvenanceRef] = None


class ModelComparison(BaseModel):
    metric_name: str
    model_a: str
    model_b: str
    value_a: Optional[float] = None
    value_b: Optional[float] = None
    delta: Optional[float] = None
    superior_model: Optional[str] = None
    status: str = "VERIFIED"  # "VERIFIED", "UNAVAILABLE"
    provenance_a: Optional[ProvenanceRef] = None
    provenance_b: Optional[ProvenanceRef] = None


class HypothesisEvaluation(BaseModel):
    hypothesis_id: str
    statement: str
    supported: Optional[bool] = None
    reasoning: str
    evidence_references: List[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    research_id: str
    question: str
    status: str
    plan: ResearchPlan
    steps_executed: int
    evidence_claims: List[EvidenceClaim] = Field(default_factory=list)
    comparisons: List[ModelComparison] = Field(default_factory=list)
    hypothesis_evaluations: List[HypothesisEvaluation] = Field(default_factory=list)
    conclusion: str
    limitations: List[str] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
