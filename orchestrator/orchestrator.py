"""Research Orchestrator driving autonomous planning, MCP execution, and evidence evaluation."""

from __future__ import annotations

import time
from typing import Optional

from orchestrator.client import MCPToolClient
from orchestrator.comparator import EvidenceComparator
from orchestrator.planner import ResearchPlanner
from orchestrator.schemas import ResearchPlan, ResearchResult
from orchestrator.state import BudgetConfig, ResearchState, ResearchStatus


class ResearchOrchestrator:
    """Orchestrates machine learning research questions over the MCP tool layer.
    
    CRITICAL INVARIANT: The orchestrator connects to the deterministic ML engine
    exclusively via MCPToolClient (mcp_server.call_tool). It does not perform
    metric calculations, execute arbitrary Python, or bypass the MCP boundary.
    """

    def __init__(
        self,
        client: Optional[MCPToolClient] = None,
        budget: Optional[BudgetConfig] = None,
        planner: Optional[ResearchPlanner] = None,
        comparator: Optional[EvidenceComparator] = None,
    ):
        self.client = client or MCPToolClient()
        self.budget = budget or BudgetConfig()
        self.planner = planner or ResearchPlanner()
        self.comparator = comparator or EvidenceComparator()

    async def run(self, question: str) -> ResearchResult:
        """Executes the full research orchestration workflow for a user research question."""
        start_time = time.perf_counter()
        state = ResearchState(
            original_question=question,
            status=ResearchStatus.PLANNING,
        )

        # 1. Parse question and generate structured plan
        plan = self.planner.plan(question)
        state.plan = plan.model_dump()

        # 2. Validate tool availability against MCP server
        available_tools = await self.client.list_available_tools()
        missing_tools = [t for t in plan.required_tools if t not in available_tools]
        if missing_tools:
            state.update_status(ResearchStatus.FAILED)
            state.errors.append(f"Required tools missing on MCP server: {missing_tools}")
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return self._build_aborted_result(
                state=state,
                plan=plan,
                reason=f"Aborted: Required tools missing ({missing_tools})",
                duration_ms=duration_ms,
            )

        # 3. Transition to execution
        state.update_status(ResearchStatus.EXECUTING)
        steps_executed = 0

        for step in plan.steps:
            # Check budgets before dispatching
            budget_exceeded, reason = state.is_budget_exceeded(self.budget)
            if budget_exceeded:
                state.update_status(ResearchStatus.BUDGET_EXHAUSTED)
                state.errors.append(f"Execution halted: {reason}")
                break

            # Dynamic argument resolution for linked steps
            args = dict(step.arguments)
            if step.tool_name == "measure_inference_latency":
                if "run_id" not in args and step.model_target:
                    run_id = state.run_ids.get(step.model_target)
                    if run_id:
                        args["run_id"] = run_id
                    else:
                        # Cannot measure latency without a tracked run_id; skip step
                        continue

            # Execute via MCP Tool Client
            success, result_data, error_str = await self.client.call_tool(
                tool_name=step.tool_name,
                arguments=args,
                state=state,
            )
            steps_executed += 1
            state.iteration_count += 1

            # Extract lineage and child run identifiers for downstream steps
            if success and step.tool_name == "train_model" and isinstance(result_data, dict):
                tracking = result_data.get("tracking", {})
                child_id = tracking.get("child_run_id")
                if child_id and step.model_target:
                    state.run_ids[step.model_target] = child_id
                lineage = result_data.get("lineage", {})
                if "dataset_fingerprint" in lineage:
                    state.dataset_fingerprint = lineage["dataset_fingerprint"]
                if "config_hash_sha256" in lineage and step.model_target:
                    state.config_hashes[step.model_target] = lineage["config_hash_sha256"]

            elif success and step.tool_name == "inspect_dataset" and isinstance(result_data, dict):
                if "fingerprint_sha256" in result_data:
                    state.dataset_fingerprint = result_data["fingerprint_sha256"]

        # 4. Compare evidence & formulate conclusion
        if state.status not in (ResearchStatus.BUDGET_EXHAUSTED, ResearchStatus.FAILED):
            state.update_status(ResearchStatus.COMPARING)

        claims = self.comparator.extract_evidence_claims(plan, state)
        comparisons = self.comparator.compare_models(plan, claims)
        hyp_evals = self.comparator.evaluate_hypotheses(plan.hypotheses, comparisons)
        conclusion, limitations = self.comparator.synthesize_conclusion(
            question=question,
            comparisons=comparisons,
            hyp_evals=hyp_evals,
        )

        if state.status == ResearchStatus.COMPARING:
            state.update_status(ResearchStatus.COMPLETED)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ResearchResult(
            research_id=state.research_id,
            question=question,
            status=state.status.value,
            plan=plan,
            steps_executed=steps_executed,
            evidence_claims=claims,
            comparisons=comparisons,
            hypothesis_evaluations=hyp_evals,
            conclusion=conclusion,
            limitations=limitations,
            total_duration_ms=duration_ms,
        )

    def _build_aborted_result(
        self,
        state: ResearchState,
        plan: ResearchPlan,
        reason: str,
        duration_ms: float,
    ) -> ResearchResult:
        """Constructs a structured failure/aborted result without hallucinating metrics."""
        return ResearchResult(
            research_id=state.research_id,
            question=state.original_question,
            status=state.status.value,
            plan=plan,
            steps_executed=0,
            evidence_claims=[],
            comparisons=[],
            hypothesis_evaluations=[],
            conclusion=f"Research could not be completed: {reason}",
            limitations=["Execution was aborted prior to collecting experimental evidence."],
            total_duration_ms=duration_ms,
        )
