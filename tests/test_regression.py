"""Tests for multi-run comparative regression engine."""

import pytest

from llm_reliability import (
    BatchComparisonReport,
    DistributionSummary,
    RegressionEngine,
    RegressionVerdict,
)
from llm_reliability.models.enums import FailureCategory, SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    RetrievalStep,
    RetrievedDocument,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span


class TestRegressionEngine:
    """Deterministic tests for batch-level comparative regression detection."""

    @pytest.fixture
    def engine(self) -> RegressionEngine:
        return RegressionEngine()

    def _create_clean_run(self, run_id: str, duration_ms: float = 100.0) -> Run:
        doc = RetrievedDocument(
            doc_id="d1",
            content="Python is an open source programming language with clear syntax and high productivity.",
            score=0.95,
        )
        step = RetrievalStep(query="python", documents=[doc])
        span = Span(
            span_id=f"{run_id}-s1",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            duration_ms=duration_ms,
            retrieval=step,
        )
        return Run(
            run_id=run_id,
            trace_id=f"t-{run_id}",
            duration_ms=duration_ms,
            spans=[span],
            final_response=FinalResponse(
                text="Python is an open source programming language with clear syntax."
            ),
        )

    def _create_failed_run(
        self,
        run_id: str,
        category: FailureCategory,
        duration_ms: float = 150.0,
    ) -> Run:
        if category == FailureCategory.RETRIEVAL_FAILURE:
            step = RetrievalStep(query="missing", documents=[])
            span = Span(
                span_id=f"{run_id}-s1",
                name="retrieval",
                kind=SpanKind.RETRIEVAL,
                duration_ms=duration_ms,
                retrieval=step,
            )
            return Run(
                run_id=run_id,
                trace_id=f"t-{run_id}",
                duration_ms=duration_ms,
                spans=[span],
            )
        elif category == FailureCategory.TOOL_ERROR:
            span = Span(
                span_id=f"{run_id}-s1",
                name="api_call",
                kind=SpanKind.TOOL,
                status=SpanStatus.ERROR,
                duration_ms=duration_ms,
                error_message="HTTP 500 Server Error",
                tool_call=ToolCall(tool_name="api_call", arguments={"q": "test"}),
                tool_result=ToolResult(tool_name="api_call", error="HTTP 500 Server Error"),
            )
            return Run(
                run_id=run_id,
                trace_id=f"t-{run_id}",
                duration_ms=duration_ms,
                spans=[span],
            )
        else:
            # Agent loop
            spans = [
                Span(
                    span_id=f"{run_id}-s{i}",
                    name="search",
                    kind=SpanKind.TOOL,
                    duration_ms=duration_ms / 3,
                    tool_call=ToolCall(tool_name="search", arguments={"q": "same"}),
                )
                for i in range(3)
            ]
            return Run(
                run_id=run_id,
                trace_id=f"t-{run_id}",
                duration_ms=duration_ms,
                spans=spans,
            )

    def test_batch_comparison_clear_regression(self, engine: RegressionEngine) -> None:
        baseline = [self._create_clean_run(f"b{i}", duration_ms=100.0) for i in range(10)]
        # Candidate has 4 failures including a new category TOOL_ERROR
        candidate = [self._create_clean_run(f"c{i}", duration_ms=110.0) for i in range(6)] + [
            self._create_failed_run(f"c_fail_{i}", FailureCategory.TOOL_ERROR, duration_ms=200.0)
            for i in range(4)
        ]

        report: BatchComparisonReport = engine.compare_batches(baseline, candidate)

        assert report.verdict == RegressionVerdict.REGRESSION
        assert report.failure_rate_delta > 0.30
        assert FailureCategory.TOOL_ERROR in report.new_failure_categories
        assert report.regression_score > 0.40

    def test_batch_comparison_clear_improvement(self, engine: RegressionEngine) -> None:
        # Baseline has 50% failures
        baseline = [self._create_clean_run(f"b{i}", duration_ms=100.0) for i in range(5)] + [
            self._create_failed_run(
                f"b_fail_{i}", FailureCategory.RETRIEVAL_FAILURE, duration_ms=100.0
            )
            for i in range(5)
        ]
        # Candidate has 0% failures
        candidate = [self._create_clean_run(f"c{i}", duration_ms=90.0) for i in range(10)]

        report: BatchComparisonReport = engine.compare_batches(baseline, candidate)

        assert report.verdict == RegressionVerdict.IMPROVED
        assert report.failure_rate_delta <= -0.40
        assert FailureCategory.RETRIEVAL_FAILURE in report.resolved_failure_categories
        assert len(report.new_failure_categories) == 0
        assert report.regression_score == 0.0

    def test_batch_comparison_clean_passed(self, engine: RegressionEngine) -> None:
        baseline = [self._create_clean_run(f"b{i}", duration_ms=100.0) for i in range(5)]
        candidate = [self._create_clean_run(f"c{i}", duration_ms=105.0) for i in range(5)]

        report: BatchComparisonReport = engine.compare_batches(baseline, candidate)

        assert report.verdict == RegressionVerdict.PASSED
        assert report.failure_rate_delta == 0.0
        assert len(report.new_failure_categories) == 0

    def test_batch_comparison_inconclusive_empty(self, engine: RegressionEngine) -> None:
        baseline: list[Run] = []
        candidate = [self._create_clean_run("c1")]

        report: BatchComparisonReport = engine.compare_batches(baseline, candidate)

        assert report.verdict == RegressionVerdict.INCONCLUSIVE
        assert report.baseline_runs_count == 0

    def test_distribution_percentiles_calculation(self, engine: RegressionEngine) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        dist: DistributionSummary = engine._compute_distribution(latencies)

        assert dist.count == 10
        assert dist.mean == 55.0
        assert dist.min_val == 10.0
        assert dist.max_val == 100.0
        assert dist.p50 == 55.0
        assert dist.p90 >= 90.0
        assert dist.p95 >= 95.0

    def test_batch_comparison_with_raw_dicts(self, engine: RegressionEngine) -> None:
        baseline_dicts = [
            {
                "trace_id": f"t-b-{i}",
                "run_id": f"r-b-{i}",
                "duration_ms": 120.0,
                "spans": [
                    {
                        "span_id": f"s-{i}",
                        "name": "search",
                        "kind": "tool",
                        "status": "ok",
                        "tool_call": {"tool_name": "search", "arguments": {"q": f"item-{i}"}},
                    }
                ],
            }
            for i in range(5)
        ]

        candidate_dicts = [
            {
                "trace_id": f"t-c-{i}",
                "run_id": f"r-c-{i}",
                "duration_ms": 130.0,
                "spans": [
                    {
                        "span_id": f"s-{i}",
                        "name": "search",
                        "kind": "tool",
                        "status": "ok",
                        "tool_call": {"tool_name": "search", "arguments": {"q": f"item-{i}"}},
                    }
                ],
            }
            for i in range(5)
        ]

        report: BatchComparisonReport = engine.compare_batches(baseline_dicts, candidate_dicts)

        assert report.baseline_runs_count == 5
        assert report.candidate_runs_count == 5
        assert report.verdict == RegressionVerdict.PASSED
