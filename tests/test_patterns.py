"""Tests for Failure Signature & Pattern Detection Engine."""

import pytest

from llm_reliability import (
    PatternAnalysisReport,
    PatternDetectionEngine,
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


class TestPatternDetectionEngine:
    """Deterministic tests for structural failure pattern detection and clustering."""

    @pytest.fixture
    def engine(self) -> PatternDetectionEngine:
        return PatternDetectionEngine()

    def _create_tool_error_run(self, run_id: str, error_msg: str) -> Run:
        span = Span(
            span_id=f"{run_id}-s1",
            name="fetch_user",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message=error_msg,
            tool_call=ToolCall(tool_name="fetch_user", arguments={"id": "123"}),
            tool_result=ToolResult(tool_name="fetch_user", error=error_msg),
        )
        return Run(run_id=run_id, trace_id=f"t-{run_id}", spans=[span])

    def _create_retrieval_empty_run(self, run_id: str) -> Run:
        span = Span(
            span_id=f"{run_id}-s1",
            name="search",
            kind=SpanKind.RETRIEVAL,
            retrieval=RetrievalStep(query="missing topic", documents=[]),
        )
        return Run(run_id=run_id, trace_id=f"t-{run_id}", spans=[span])

    def _create_clean_run(self, run_id: str) -> Run:
        doc = RetrievedDocument(
            doc_id="d1",
            content="Python provides extensive standard library modules for data analysis.",
            score=0.95,
        )
        step = RetrievalStep(query="python", documents=[doc])
        span = Span(span_id=f"{run_id}-s1", name="search", kind=SpanKind.RETRIEVAL, retrieval=step)
        return Run(
            run_id=run_id,
            trace_id=f"t-{run_id}",
            spans=[span],
            final_response=FinalResponse(
                text="Python provides extensive standard library modules for data analysis."
            ),
        )

    def test_pattern_detection_clusters_identical_failures(
        self, engine: PatternDetectionEngine
    ) -> None:
        # 5 tool error runs with identical structural failure
        tool_runs = [
            self._create_tool_error_run(f"r_tool_{i}", "Connection timeout after 30s")
            for i in range(5)
        ]
        # 3 retrieval empty runs
        ret_runs = [self._create_retrieval_empty_run(f"r_ret_{i}") for i in range(3)]

        runs = tool_runs + ret_runs
        report: PatternAnalysisReport = engine.analyze_patterns(runs)

        assert report.total_runs_analyzed == 8
        assert report.failed_runs_count == 8
        assert report.unique_patterns_count == 2

        # Cluster 1 (Tool errors) must rank highest with 5 occurrences
        assert report.clusters[0].occurrence_count == 5
        assert report.clusters[0].signature.category == FailureCategory.TOOL_ERROR
        assert report.clusters[0].frequency_ratio == round(5 / 8, 4)

        # Cluster 2 (Retrieval errors)
        assert report.clusters[1].occurrence_count == 3
        assert report.clusters[1].signature.category == FailureCategory.RETRIEVAL_FAILURE

    def test_deterministic_fingerprint_invariance(self, engine: PatternDetectionEngine) -> None:
        # Errors differing only by UUIDs/numbers should yield identical fingerprints
        run1 = self._create_tool_error_run(
            "r1", "Failed request to user 12345 with id 12345678-1234-5678-1234-567812345678"
        )
        run2 = self._create_tool_error_run(
            "r2", "Failed request to user 98765 with id 87654321-4321-8765-4321-876543210987"
        )

        diag1 = engine.diagnostic_engine.diagnose_run(run1)
        diag2 = engine.diagnostic_engine.diagnose_run(run2)

        sigs1 = engine.generate_signatures(diag1, run1)
        sigs2 = engine.generate_signatures(diag2, run2)

        assert len(sigs1) >= 1
        assert len(sigs2) >= 1
        assert sigs1[0].fingerprint == sigs2[0].fingerprint

    def test_pattern_detection_all_clean(self, engine: PatternDetectionEngine) -> None:
        runs = [self._create_clean_run(f"clean_{i}") for i in range(10)]
        report: PatternAnalysisReport = engine.analyze_patterns(runs)

        assert report.total_runs_analyzed == 10
        assert report.failed_runs_count == 0
        assert report.unique_patterns_count == 0
        assert len(report.clusters) == 0

    def test_pattern_detection_empty_input(self, engine: PatternDetectionEngine) -> None:
        report: PatternAnalysisReport = engine.analyze_patterns([])
        assert report.total_runs_analyzed == 0
        assert report.failed_runs_count == 0
        assert len(report.clusters) == 0

    def test_representative_trace_selection(self, engine: PatternDetectionEngine) -> None:
        run1 = self._create_tool_error_run("r1", "Database timeout")
        # Add an additional span to run2 to verify representative selection
        run2 = self._create_tool_error_run("r2", "Database timeout")
        run2.spans.append(
            Span(
                span_id="r2-s2",
                name="cleanup",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="cleanup", arguments={}),
            )
        )

        report: PatternAnalysisReport = engine.analyze_patterns([run1, run2])
        assert len(report.clusters) == 1
        assert report.clusters[0].representative_run_id == "r2"
