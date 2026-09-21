"""Deterministic evidence comparator enforcing the Zero-Hallucination Evidence Rule."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from orchestrator.schemas import (
    EvidenceClaim,
    Hypothesis,
    HypothesisEvaluation,
    ModelComparison,
    ProvenanceRef,
    ResearchPlan,
)
from orchestrator.state import ResearchState, ToolCallRecord


class EvidenceComparator:
    """Extracts, verifies, and compares deterministic model metrics from tool call records.
    
    ZERO-HALLUCINATION RULE:
    1. Every claim is bound to a concrete ToolCallRecord.
    2. Missing evidence produces status='UNAVAILABLE' rather than invented values.
    3. All arithmetic is deterministic and traceable to source tool data.
    """

    HIGHER_IS_BETTER: Set[str] = {
        "accuracy",
        "balanced_accuracy",
        "precision_binary",
        "recall_binary",
        "f1_binary",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "pr_auc",
    }

    LOWER_IS_BETTER: Set[str] = {
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_latency_ms",
        "log_loss_val",
    }

    def extract_evidence_claims(
        self,
        plan: ResearchPlan,
        state: ResearchState,
    ) -> List[EvidenceClaim]:
        """Extracts verified evidence claims directly from state tool call records."""
        claims: List[EvidenceClaim] = []
        metrics_to_collect = [plan.primary_metric]
        if plan.secondary_metric:
            metrics_to_collect.append(plan.secondary_metric)

        for model in plan.candidate_models:
            # 1. Search for train_model outputs
            train_record, train_data = self._find_tool_call(state, "train_model", model_target=model)
            # 2. Search for latency outputs
            latency_record, latency_data = self._find_tool_call(state, "measure_inference_latency", model_target=model)

            for metric in metrics_to_collect:
                val: Optional[float] = None
                prov: Optional[ProvenanceRef] = None
                unit: Optional[str] = None
                status = "UNAVAILABLE"

                if metric in ("accuracy", "balanced_accuracy", "f1_binary", "roc_auc", "precision_binary", "recall_binary"):
                    unit = "ratio"
                    if train_record and train_data and train_record.success:
                        test_metrics = train_data.get("test_metrics", {})
                        if metric in test_metrics:
                            val = float(test_metrics[metric])
                            status = "VERIFIED"
                            run_id = train_data.get("tracking", {}).get("child_run_id")
                            prov = ProvenanceRef(
                                tool_name="train_model",
                                call_id=train_record.call_id,
                                run_id=run_id,
                                property_path=f"test_metrics.{metric}",
                                timestamp=train_record.timestamp,
                            )

                elif "latency" in metric:
                    unit = "ms"
                    # Try explicit latency tool first, fallback to train_model benchmarks
                    if latency_record and latency_data and latency_record.success:
                        benchmarks = latency_data.get("benchmarks", [])
                        if benchmarks:
                            b0 = benchmarks[0]
                            if metric in b0:
                                val = float(b0[metric])
                                status = "VERIFIED"
                                prov = ProvenanceRef(
                                    tool_name="measure_inference_latency",
                                    call_id=latency_record.call_id,
                                    run_id=latency_data.get("run_id"),
                                    property_path=f"benchmarks[0].{metric}",
                                    timestamp=latency_record.timestamp,
                                )
                    elif train_record and train_data and train_record.success:
                        benchmarks = train_data.get("latency_benchmarks", [])
                        if benchmarks:
                            b0 = benchmarks[0]
                            if metric in b0:
                                val = float(b0[metric])
                                status = "VERIFIED"
                                prov = ProvenanceRef(
                                    tool_name="train_model",
                                    call_id=train_record.call_id,
                                    run_id=train_data.get("tracking", {}).get("child_run_id"),
                                    property_path=f"latency_benchmarks[0].{metric}",
                                    timestamp=train_record.timestamp,
                                )

                claims.append(
                    EvidenceClaim(
                        metric_name=metric,
                        model_name=model,
                        value=val,
                        status=status,
                        unit=unit,
                        provenance=prov,
                    )
                )

        return claims

    def compare_models(
        self,
        plan: ResearchPlan,
        claims: List[EvidenceClaim],
    ) -> List[ModelComparison]:
        """Compares models pairwise for each metric of interest using verified claims."""
        comparisons: List[ModelComparison] = []
        metrics_to_compare = [plan.primary_metric]
        if plan.secondary_metric:
            metrics_to_compare.append(plan.secondary_metric)

        claims_map: Dict[Tuple[str, str], EvidenceClaim] = {
            (c.model_name, c.metric_name): c for c in claims
        }

        model_a = plan.candidate_models[0]
        model_b = plan.candidate_models[1]

        for metric in metrics_to_compare:
            claim_a = claims_map.get((model_a, metric))
            claim_b = claims_map.get((model_b, metric))

            if claim_a and claim_b and claim_a.status == "VERIFIED" and claim_b.status == "VERIFIED":
                val_a = claim_a.value
                val_b = claim_b.value
                assert val_a is not None and val_b is not None
                delta = round(val_b - val_a, 6)

                superior = None
                if metric in self.HIGHER_IS_BETTER:
                    if delta > 0.0001:
                        superior = model_b
                    elif delta < -0.0001:
                        superior = model_a
                    else:
                        superior = "TIE"
                elif metric in self.LOWER_IS_BETTER:
                    if delta < -0.0001:
                        superior = model_b
                    elif delta > 0.0001:
                        superior = model_a
                    else:
                        superior = "TIE"

                comparisons.append(
                    ModelComparison(
                        metric_name=metric,
                        model_a=model_a,
                        model_b=model_b,
                        value_a=val_a,
                        value_b=val_b,
                        delta=delta,
                        superior_model=superior,
                        status="VERIFIED",
                        provenance_a=claim_a.provenance,
                        provenance_b=claim_b.provenance,
                    )
                )
            else:
                comparisons.append(
                    ModelComparison(
                        metric_name=metric,
                        model_a=model_a,
                        model_b=model_b,
                        value_a=claim_a.value if claim_a else None,
                        value_b=claim_b.value if claim_b else None,
                        delta=None,
                        superior_model=None,
                        status="UNAVAILABLE",
                        provenance_a=claim_a.provenance if claim_a else None,
                        provenance_b=claim_b.provenance if claim_b else None,
                    )
                )

        return comparisons

    def evaluate_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        comparisons: List[ModelComparison],
    ) -> List[HypothesisEvaluation]:
        """Evaluates formulated hypotheses against verified comparisons."""
        evals: List[HypothesisEvaluation] = []
        comp_map = {c.metric_name: c for c in comparisons}

        for h in hypotheses:
            comp = comp_map.get(h.target_metric)
            if not comp or comp.status != "VERIFIED" or comp.value_a is None or comp.value_b is None:
                evals.append(
                    HypothesisEvaluation(
                        hypothesis_id=h.id,
                        statement=h.statement,
                        supported=None,
                        reasoning=f"Evidence for metric '{h.target_metric}' is UNAVAILABLE.",
                        evidence_references=[],
                    )
                )
                continue

            val_a = comp.value_a
            val_b = comp.value_b

            supported = False
            if h.comparator == ">":
                supported = val_b > val_a
            elif h.comparator == "<":
                supported = val_a < val_b
            elif h.comparator == "==":
                supported = abs(val_a - val_b) < 1e-5

            ref_a = f"{comp.model_a}={val_a} (call_id: {comp.provenance_a.call_id[:8] if comp.provenance_a else 'n/a'})"
            ref_b = f"{comp.model_b}={val_b} (call_id: {comp.provenance_b.call_id[:8] if comp.provenance_b else 'n/a'})"

            reasoning = (
                f"Observed {h.target_metric}: {ref_b} vs {ref_a}. "
                f"Delta ({comp.model_b} - {comp.model_a}) is {comp.delta}. "
                f"Hypothesis that {h.statement} is {'SUPPORTED' if supported else 'NOT SUPPORTED'}."
            )

            evals.append(
                HypothesisEvaluation(
                    hypothesis_id=h.id,
                    statement=h.statement,
                    supported=supported,
                    reasoning=reasoning,
                    evidence_references=[ref_a, ref_b],
                )
            )

        return evals

    def synthesize_conclusion(
        self,
        question: str,
        comparisons: List[ModelComparison],
        hyp_evals: List[HypothesisEvaluation],
    ) -> Tuple[str, List[str]]:
        """Formulates an evidence-grounded research conclusion and lists limitations."""
        lines: List[str] = [f"Empirical evaluation for research question: '{question}'."]
        limitations: List[str] = [
            "Evaluated on a single deterministic seed (seed 42). Multi-seed variance and statistical significance tests were not assessed in this run.",
            "Default baseline hyperparameters used; automated hyperparameter tuning was not requested.",
            "Conclusions reflect performance on this specific dataset partition and preprocessing pipeline, and do not imply universal algorithmic superiority.",
        ]

        for c in comparisons:
            if c.status == "VERIFIED":
                lines.append(
                    f"- {c.metric_name}: {c.model_a}={c.value_a}, {c.model_b}={c.value_b} (delta={c.delta}). "
                    f"Higher-performing candidate on this partition: {c.superior_model}."
                )
            else:
                lines.append(f"- {c.metric_name}: Evidence was UNAVAILABLE.")
                limitations.append(f"Metric '{c.metric_name}' could not be evaluated due to missing tool execution evidence.")

        for h in hyp_evals:
            lines.append(f"Hypothesis {h.hypothesis_id} ({h.statement}): {'Supported' if h.supported else 'Not Supported'}.")

        conclusion = " ".join(lines)
        return conclusion, limitations

    def _find_tool_call(
        self,
        state: ResearchState,
        tool_name: str,
        model_target: Optional[str] = None,
    ) -> Tuple[Optional[ToolCallRecord], Optional[Dict[str, Any]]]:
        """Finds matching tool call record from state."""
        for call in reversed(state.tool_calls):
            if call.tool_name == tool_name and call.success and call.result:
                if model_target:
                    arg_model = call.arguments.get("model_name")
                    res_model = call.result.get("model_name")
                    if arg_model == model_target or res_model == model_target:
                        return call, call.result
                else:
                    return call, call.result
        return None, None
