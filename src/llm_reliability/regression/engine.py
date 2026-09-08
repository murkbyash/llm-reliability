"""Multi-run comparative regression engine analyzing batches across versions."""

import math
from collections.abc import Sequence
from typing import Any

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import FailureCategory
from llm_reliability.models.trace import Run, Trace
from llm_reliability.normalization.loader import load_trace
from llm_reliability.regression.models import (
    BatchComparisonReport,
    DistributionSummary,
    FailureRateSummary,
    RegressionVerdict,
)


class RegressionEngine:
    """Evaluates multi-run batches to detect statistical regressions, score drift, and new failure modes."""

    def __init__(self, diagnostic_engine: DiagnosticEngine | None = None) -> None:
        """Initialize regression engine with diagnostic analyzer."""
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()

    def compare_batches(
        self,
        baseline_source: Sequence[Run | dict[str, Any] | Any] | Trace,
        candidate_source: Sequence[Run | dict[str, Any] | Any] | Trace,
        failure_rate_tolerance: float = 0.0,
        latency_tolerance_ratio: float = 0.25,
    ) -> BatchComparisonReport:
        """Compare a baseline batch against a candidate batch across reliability metrics and latencies.

        Args:
            baseline_source: Collection of baseline runs, dicts, or Trace.
            candidate_source: Collection of candidate runs, dicts, or Trace.
            failure_rate_tolerance: Allowed increase in failure rate before flagging regression (default: 0.0).
            latency_tolerance_ratio: Allowed fractional increase in p95 latency before flagging regression (default: 0.25).

        Returns:
            BatchComparisonReport with aggregate failure rates, distributions, and verdict.
        """
        baseline_runs = self._extract_runs(baseline_source)
        candidate_runs = self._extract_runs(candidate_source)

        if not baseline_runs or not candidate_runs:
            return self._build_inconclusive_report(len(baseline_runs), len(candidate_runs))

        baseline_diagnoses = [self.diagnostic_engine.diagnose_run(r) for r in baseline_runs]
        candidate_diagnoses = [self.diagnostic_engine.diagnose_run(r) for r in candidate_runs]

        baseline_summary = self._compute_failure_summary(baseline_diagnoses)
        candidate_summary = self._compute_failure_summary(candidate_diagnoses)

        baseline_latencies = [self._extract_run_latency(r) for r in baseline_runs]
        candidate_latencies = [self._extract_run_latency(r) for r in candidate_runs]

        dist_baseline = self._compute_distribution(baseline_latencies)
        dist_candidate = self._compute_distribution(candidate_latencies)

        p95_delta = round(dist_candidate.p95 - dist_baseline.p95, 2)
        failure_delta = round(candidate_summary.failure_rate - baseline_summary.failure_rate, 4)

        baseline_cats = {
            cat for d in baseline_diagnoses for cat in self._get_diagnosis_categories(d)
        }
        candidate_cats = {
            cat for d in candidate_diagnoses for cat in self._get_diagnosis_categories(d)
        }

        new_cats = sorted(list(candidate_cats - baseline_cats), key=lambda x: x.value)
        resolved_cats = sorted(list(baseline_cats - candidate_cats), key=lambda x: x.value)

        # Compute regression score (0.0 to 1.0)
        regression_score = self._compute_regression_score(
            failure_delta=failure_delta,
            new_categories_count=len(new_cats),
            dist_baseline=dist_baseline,
            dist_candidate=dist_candidate,
            latency_tolerance_ratio=latency_tolerance_ratio,
        )

        # Determine Verdict
        if failure_delta > failure_rate_tolerance or len(new_cats) > 0 or regression_score >= 0.35:
            verdict = RegressionVerdict.REGRESSION
        elif failure_delta < -0.01 or (
            len(resolved_cats) > 0 and not new_cats and failure_delta <= 0
        ):
            verdict = RegressionVerdict.IMPROVED
        else:
            verdict = RegressionVerdict.PASSED

        summary = (
            f"Batch Comparison Verdict: {verdict.value.upper()}. "
            f"Failure rate changed from {int(baseline_summary.failure_rate * 100)}% "
            f"to {int(candidate_summary.failure_rate * 100)}% (delta: {round(failure_delta * 100, 1)}%). "
            f"New failure modes: {len(new_cats)}. Resolved failure modes: {len(resolved_cats)}. "
            f"p95 latency changed from {dist_baseline.p95}ms to {dist_candidate.p95}ms."
        )

        return BatchComparisonReport(
            baseline_runs_count=len(baseline_runs),
            candidate_runs_count=len(candidate_runs),
            baseline_failure_summary=baseline_summary,
            candidate_failure_summary=candidate_summary,
            failure_rate_delta=failure_delta,
            latency_baseline=dist_baseline,
            latency_candidate=dist_candidate,
            latency_p95_delta_ms=p95_delta,
            new_failure_categories=new_cats,
            resolved_failure_categories=resolved_cats,
            regression_score=regression_score,
            verdict=verdict,
            summary=summary,
        )

    def _extract_runs(self, source: Sequence[Any] | Trace) -> list[Run]:
        """Normalize various input types into a list of canonical Run models."""
        if isinstance(source, Trace):
            return list(source.runs)

        runs: list[Run] = []
        for item in source:
            if isinstance(item, Run):
                runs.append(item)
            elif isinstance(item, Trace):
                runs.extend(item.runs)
            else:
                trace = load_trace(item)
                runs.extend(trace.runs)
        return runs

    def _extract_run_latency(self, run: Run) -> float:
        """Extract or estimate execution latency in milliseconds for a run."""
        if run.duration_ms is not None and run.duration_ms > 0:
            return float(run.duration_ms)
        span_durations = [s.duration_ms for s in run.spans if s.duration_ms is not None]
        return float(sum(span_durations)) if span_durations else 0.0

    def _get_diagnosis_categories(self, diagnosis: Diagnosis) -> set[FailureCategory]:
        """Extract all failure categories identified in a diagnosis."""
        cats = {f.category for f in diagnosis.failures}
        if diagnosis.primary_category != FailureCategory.NONE:
            cats.add(diagnosis.primary_category)
        return cats

    def _compute_failure_summary(self, diagnoses: list[Diagnosis]) -> FailureRateSummary:
        """Aggregate failure rates and category distributions for a batch of diagnoses."""
        total = len(diagnoses)
        failed = sum(
            1 for d in diagnoses if d.primary_category != FailureCategory.NONE or d.failures
        )
        rate = round(failed / total, 4) if total > 0 else 0.0

        counts: dict[str, int] = {}
        for d in diagnoses:
            for f in d.failures:
                key = f.category.value
                counts[key] = counts.get(key, 0) + 1

        return FailureRateSummary(
            total_runs=total,
            failed_runs=failed,
            failure_rate=rate,
            category_counts=counts,
        )

    def _compute_distribution(self, values: list[float]) -> DistributionSummary:
        """Compute statistical percentiles (p50, p90, p95, p99) and mean for a series."""
        if not values:
            return DistributionSummary()

        clean_vals = sorted([float(v) for v in values if not math.isnan(v)])
        n = len(clean_vals)
        if n == 0:
            return DistributionSummary()

        mean_val = round(sum(clean_vals) / n, 2)
        min_val = round(clean_vals[0], 2)
        max_val = round(clean_vals[-1], 2)

        p50 = self._percentile(clean_vals, 0.50)
        p90 = self._percentile(clean_vals, 0.90)
        p95 = self._percentile(clean_vals, 0.95)
        p99 = self._percentile(clean_vals, 0.99)

        return DistributionSummary(
            count=n,
            mean=mean_val,
            min_val=min_val,
            max_val=max_val,
            p50=round(p50, 2),
            p90=round(p90, 2),
            p95=round(p95, 2),
            p99=round(p99, 2),
        )

    def _percentile(self, sorted_data: list[float], p: float) -> float:
        """Calculate p-th percentile using linear interpolation."""
        if not sorted_data:
            return 0.0
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return d0 + d1

    def _compute_regression_score(
        self,
        failure_delta: float,
        new_categories_count: int,
        dist_baseline: DistributionSummary,
        dist_candidate: DistributionSummary,
        latency_tolerance_ratio: float,
    ) -> float:
        """Compute normalized regression score between 0.0 and 1.0."""
        score = 0.0

        # Failure rate delta impact (0.0 to 0.60)
        if failure_delta > 0:
            score += min(failure_delta * 1.5, 0.60)

        # New failure categories penalty (0.0 to 0.30)
        if new_categories_count > 0:
            score += min(new_categories_count * 0.15, 0.30)

        # Latency regression penalty (0.0 to 0.20)
        if dist_baseline.p95 > 0 and dist_candidate.p95 > dist_baseline.p95:
            latency_ratio = (dist_candidate.p95 - dist_baseline.p95) / dist_baseline.p95
            if latency_ratio > latency_tolerance_ratio:
                score += min((latency_ratio - latency_tolerance_ratio) * 0.5, 0.20)

        return round(min(max(score, 0.0), 1.0), 3)

    def _build_inconclusive_report(
        self, baseline_count: int, candidate_count: int
    ) -> BatchComparisonReport:
        """Generate report when insufficient runs are provided."""
        return BatchComparisonReport(
            baseline_runs_count=baseline_count,
            candidate_runs_count=candidate_count,
            baseline_failure_summary=FailureRateSummary(),
            candidate_failure_summary=FailureRateSummary(),
            failure_rate_delta=0.0,
            latency_baseline=DistributionSummary(),
            latency_candidate=DistributionSummary(),
            latency_p95_delta_ms=0.0,
            regression_score=0.0,
            verdict=RegressionVerdict.INCONCLUSIVE,
            summary="Comparative analysis inconclusive: Empty batch provided for baseline or candidate.",
        )
