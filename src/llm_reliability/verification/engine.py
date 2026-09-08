"""Counterfactual simulation and rerun verification engine."""

import copy
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import FailureCategory, Severity, SpanKind
from llm_reliability.models.execution import RetrievalStep, RetrievedDocument
from llm_reliability.models.trace import Run, Span, Trace
from llm_reliability.verification.models import (
    CounterfactualResult,
    MetricComparison,
    VerificationReport,
)

SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class VerificationEngine:
    """Simulates counterfactual fixes and verifies reliability improvements."""

    def __init__(self, diagnostic_engine: DiagnosticEngine | None = None) -> None:
        """Initialize verification engine with diagnostic orchestrator."""
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()

    def simulate_top_k_adjustment(
        self,
        run: Run,
        new_k: int | None = None,
        min_score_threshold: float | None = None,
    ) -> CounterfactualResult:
        """Simulate adjusting the top-k parameter or filtering low-relevance documents."""
        before_diag = self.diagnostic_engine.diagnose_run(run)
        simulated_run = copy.deepcopy(run)

        for span in simulated_run.spans:
            if span.kind == SpanKind.RETRIEVAL and span.retrieval:
                docs = list(span.retrieval.documents)
                if min_score_threshold is not None:
                    docs = [d for d in docs if d.score is None or d.score >= min_score_threshold]
                if new_k is not None:
                    docs = docs[:new_k]

                span.retrieval = RetrievalStep(
                    query=span.retrieval.query,
                    documents=docs,
                    top_k=new_k or span.retrieval.top_k,
                    retriever_name=span.retrieval.retriever_name,
                )

        after_diag = self.diagnostic_engine.diagnose_run(simulated_run)
        metric_comparisons = self._compare_metrics(before_diag, after_diag)

        before_failures = {f.category for f in before_diag.failures}
        if before_diag.primary_category != FailureCategory.NONE:
            before_failures.add(before_diag.primary_category)

        after_failures = {f.category for f in after_diag.failures}
        if after_diag.primary_category != FailureCategory.NONE:
            after_failures.add(after_diag.primary_category)

        resolved = list(before_failures - after_failures)
        remaining = list(after_failures)

        sev_before = SEVERITY_RANK.get(before_diag.severity, 0)
        sev_after = SEVERITY_RANK.get(after_diag.severity, 0)
        is_improved = (
            (len(resolved) > 0)
            or (sev_after < sev_before)
            or (len(after_diag.failures) < len(before_diag.failures))
        )

        summary = (
            f"Simulated Top-K/Threshold adjustment (new_k={new_k}, min_score={min_score_threshold}): "
            f"{'Improved reliability' if is_improved else 'No improvement detected'}. "
            f"Resolved {len(resolved)} failure(s)."
        )

        return CounterfactualResult(
            experiment_name="Top-K & Score Filtering Simulation",
            experiment_type="TOP_K_ADJUSTMENT",
            parameters={"new_k": new_k, "min_score_threshold": min_score_threshold},
            before_diagnosis=before_diag,
            after_diagnosis=after_diag,
            metric_comparisons=metric_comparisons,
            is_improved=is_improved,
            confidence=0.88,
            resolved_failures=resolved,
            remaining_failures=remaining,
            summary=summary,
        )

    def simulate_context_deduplication(
        self,
        run: Run,
        similarity_threshold: float = 0.85,
    ) -> CounterfactualResult:
        """Simulate near-duplicate chunk removal before LLM context construction."""
        before_diag = self.diagnostic_engine.diagnose_run(run)
        simulated_run = copy.deepcopy(run)

        for span in simulated_run.spans:
            if span.kind == SpanKind.RETRIEVAL and span.retrieval:
                deduped_docs = self._deduplicate_documents(
                    span.retrieval.documents, similarity_threshold
                )
                span.retrieval = RetrievalStep(
                    query=span.retrieval.query,
                    documents=deduped_docs,
                    top_k=span.retrieval.top_k,
                    retriever_name=span.retrieval.retriever_name,
                )

        after_diag = self.diagnostic_engine.diagnose_run(simulated_run)
        metric_comparisons = self._compare_metrics(before_diag, after_diag)

        before_failures = {f.category for f in before_diag.failures}
        if before_diag.primary_category != FailureCategory.NONE:
            before_failures.add(before_diag.primary_category)

        after_failures = {f.category for f in after_diag.failures}
        if after_diag.primary_category != FailureCategory.NONE:
            after_failures.add(after_diag.primary_category)

        resolved = list(before_failures - after_failures)
        remaining = list(after_failures)

        is_improved = len(resolved) > 0 or any(
            m.name == "retrieval_duplicate_ratio" and m.improved for m in metric_comparisons
        )

        summary = (
            f"Simulated pre-context deduplication (similarity_threshold={similarity_threshold}): "
            f"{'Reduced context duplication and improved efficiency' if is_improved else 'No change in duplication'}."
        )

        return CounterfactualResult(
            experiment_name="Pre-Context Deduplication Simulation",
            experiment_type="CONTEXT_DEDUPLICATION",
            parameters={"similarity_threshold": similarity_threshold},
            before_diagnosis=before_diag,
            after_diagnosis=after_diag,
            metric_comparisons=metric_comparisons,
            is_improved=is_improved,
            confidence=0.92,
            resolved_failures=resolved,
            remaining_failures=remaining,
            summary=summary,
        )

    def simulate_agent_loop_interception(
        self,
        run: Run,
        max_repetitions: int = 2,
    ) -> CounterfactualResult:
        """Simulate middleware loop interception breaking repetitive tool call cycles."""
        before_diag = self.diagnostic_engine.diagnose_run(run)
        simulated_run = copy.deepcopy(run)

        # Truncate repeated tool calls exceeding max_repetitions
        filtered_spans: list[Span] = []
        seen_calls: dict[str, int] = {}

        for span in simulated_run.spans:
            if span.kind == SpanKind.TOOL and span.tool_call:
                sig = f"{span.tool_call.tool_name}:{sorted(span.tool_call.arguments.items()) if isinstance(span.tool_call.arguments, dict) else str(span.tool_call.arguments)}"
                seen_calls[sig] = seen_calls.get(sig, 0) + 1
                if seen_calls[sig] <= max_repetitions:
                    filtered_spans.append(span)
            else:
                filtered_spans.append(span)

        simulated_run.spans = filtered_spans
        after_diag = self.diagnostic_engine.diagnose_run(simulated_run)
        metric_comparisons = self._compare_metrics(before_diag, after_diag)

        before_failures = {f.category for f in before_diag.failures}
        if before_diag.primary_category != FailureCategory.NONE:
            before_failures.add(before_diag.primary_category)

        after_failures = {f.category for f in after_diag.failures}
        if after_diag.primary_category != FailureCategory.NONE:
            after_failures.add(after_diag.primary_category)

        resolved = list(before_failures - after_failures)
        remaining = list(after_failures)

        is_improved = len(resolved) > 0 or len(after_diag.failures) < len(before_diag.failures)

        summary = (
            f"Simulated loop interception guardrail (max_repetitions={max_repetitions}): "
            f"{'Successfully intercepted agent infinite loop' if is_improved else 'No loop detected to intercept'}."
        )

        return CounterfactualResult(
            experiment_name="Agent Loop Guardrail Interception Simulation",
            experiment_type="AGENT_GUARDRAIL",
            parameters={"max_repetitions": max_repetitions},
            before_diagnosis=before_diag,
            after_diagnosis=after_diag,
            metric_comparisons=metric_comparisons,
            is_improved=is_improved,
            confidence=0.95,
            resolved_failures=resolved,
            remaining_failures=remaining,
            summary=summary,
        )

    def verify_fix(
        self,
        before_source: Trace | Run | str | Path | dict[str, Any] | list[Any] | TextIO,
        after_source: Trace | Run | str | Path | dict[str, Any] | list[Any] | TextIO,
    ) -> VerificationReport:
        """Perform comprehensive before vs after verification of an applied developer fix."""
        before_diag = self.diagnostic_engine.diagnose(before_source)
        after_diag = self.diagnostic_engine.diagnose(after_source)

        metric_comparisons = self._compare_metrics(before_diag, after_diag)

        before_failures = {f.category for f in before_diag.failures}
        if before_diag.primary_category != FailureCategory.NONE:
            before_failures.add(before_diag.primary_category)

        after_failures = {f.category for f in after_diag.failures}
        if after_diag.primary_category != FailureCategory.NONE:
            after_failures.add(after_diag.primary_category)

        resolved = list(before_failures - after_failures)
        new_regressions = list(after_failures - before_failures)

        sev_before_val = SEVERITY_RANK.get(before_diag.severity, 0)
        sev_after_val = SEVERITY_RANK.get(after_diag.severity, 0)

        is_verified = (
            len(resolved) > 0 and len(new_regressions) == 0 and sev_after_val <= sev_before_val
        ) or (
            len(before_failures) == 0
            and len(after_failures) == 0
            and sev_after_val <= sev_before_val
        )

        summary = (
            f"Fix Verification: {'VERIFIED IMPROVEMENT' if is_verified else 'NOT VERIFIED'}. "
            f"Resolved {len(resolved)} failure(s). New regressions: {len(new_regressions)}. "
            f"Severity changed from {before_diag.severity.value} to {after_diag.severity.value}."
        )

        return VerificationReport(
            is_verified=is_verified,
            overall_confidence=0.94,
            severity_before=before_diag.severity,
            severity_after=after_diag.severity,
            metric_comparisons=metric_comparisons,
            resolved_failures=resolved,
            new_regressions=new_regressions,
            summary=summary,
        )

    def _compare_metrics(
        self, before_diag: Diagnosis, after_diag: Diagnosis
    ) -> list[MetricComparison]:
        """Compute structured before vs after comparisons across diagnostic metrics."""
        before_map = {m.name: m for m in before_diag.metrics}
        after_map = {m.name: m for m in after_diag.metrics}

        all_names = sorted(set(before_map.keys()) | set(after_map.keys()))
        comparisons: list[MetricComparison] = []

        for name in all_names:
            b_metric = before_map.get(name)
            a_metric = after_map.get(name)

            b_val = b_metric.value if b_metric else None
            a_val = a_metric.value if a_metric else None

            delta: float | None = None
            pct_change: float | None = None

            b_passed = bool(b_metric.passed) if b_metric else False
            a_passed = bool(a_metric.passed) if a_metric else False

            if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float)):
                delta = round(float(a_val) - float(b_val), 4)
                if float(b_val) != 0.0:
                    pct_change = round((delta / float(b_val)) * 100.0, 2)

                # Determine if improved based on metric naming convention
                if "duplicate" in name or "error" in name or "loop" in name or "shortfall" in name:
                    improved = delta < 0
                elif "relevance" in name or "grounding" in name or "faithfulness" in name:
                    improved = delta > 0
                else:
                    improved = a_passed and not b_passed
            elif isinstance(b_val, bool) and isinstance(a_val, bool):
                improved = a_passed and not b_passed
            else:
                improved = a_passed and not b_passed

            comparisons.append(
                MetricComparison(
                    name=name,
                    before_value=b_val,
                    after_value=a_val,
                    delta=delta,
                    percent_change=pct_change,
                    improved=improved,
                    description=f"Comparison for metric {name}",
                )
            )

        return comparisons

    def _deduplicate_documents(
        self,
        documents: Sequence[RetrievedDocument],
        similarity_threshold: float,
    ) -> list[RetrievedDocument]:
        """Deduplicate a list of documents based on token Jaccard similarity."""
        unique_docs: list[RetrievedDocument] = []

        for doc in documents:
            doc_tokens = set(re.findall(r"\b\w+\b", doc.content.lower()))
            is_dup = False
            for u in unique_docs:
                u_tokens = set(re.findall(r"\b\w+\b", u.content.lower()))
                if not doc_tokens and not u_tokens:
                    is_dup = True
                    break
                union = doc_tokens | u_tokens
                if union:
                    jaccard = len(doc_tokens & u_tokens) / len(union)
                    if jaccard >= similarity_threshold:
                        is_dup = True
                        break
            if not is_dup:
                unique_docs.append(doc)

        return unique_docs
