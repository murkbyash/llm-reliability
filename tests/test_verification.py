"""Tests for counterfactual simulation and rerun verification engine."""

import pytest

from llm_reliability import (
    DiagnosticEngine,
    VerificationEngine,
)
from llm_reliability.models.enums import FailureCategory, Severity, SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    RetrievalStep,
    RetrievedDocument,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span
from llm_reliability.verification.models import CounterfactualResult, VerificationReport


class TestVerificationEngine:
    """Deterministic tests for counterfactual simulation and verification."""

    @pytest.fixture
    def verification_engine(self) -> VerificationEngine:
        return VerificationEngine()

    def test_simulate_top_k_score_filtering(self, verification_engine: VerificationEngine) -> None:
        docs = [
            RetrievedDocument(
                doc_id="d1",
                content="Relevant document containing specific answer facts about quantum mechanics.",
                score=0.92,
            ),
            RetrievedDocument(
                doc_id="d2",
                content="Irrelevant noise document mentioning baking recipes.",
                score=0.25,
            ),
        ]
        step = RetrievalStep(query="quantum mechanics", documents=docs)
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        result: CounterfactualResult = verification_engine.simulate_top_k_adjustment(
            run, min_score_threshold=0.70
        )

        assert result.is_improved or len(result.after_diagnosis.failures) <= len(
            result.before_diagnosis.failures
        )
        assert result.parameters["min_score_threshold"] == 0.70

    def test_simulate_context_deduplication(self, verification_engine: VerificationEngine) -> None:
        docs = [
            RetrievedDocument(
                doc_id="d1",
                content="Identical paragraph text repeated for user documentation index.",
                score=0.88,
            ),
            RetrievedDocument(
                doc_id="d2",
                content="Identical paragraph text repeated for user documentation index.",
                score=0.87,
            ),
            RetrievedDocument(
                doc_id="d3",
                content="Identical paragraph text repeated for user documentation index.",
                score=0.86,
            ),
        ]
        step = RetrievalStep(query="documentation", documents=docs)
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        result: CounterfactualResult = verification_engine.simulate_context_deduplication(
            run, similarity_threshold=0.85
        )

        assert result.is_improved is True
        assert any(
            m.name == "retrieval_duplicate_ratio" and m.improved for m in result.metric_comparisons
        )

    def test_simulate_agent_loop_interception(
        self, verification_engine: VerificationEngine
    ) -> None:
        spans = [
            Span(
                span_id=f"s{i}",
                name="lookup",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="lookup", arguments={"item": "x"}),
            )
            for i in range(4)
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)

        result: CounterfactualResult = verification_engine.simulate_agent_loop_interception(
            run, max_repetitions=1
        )

        assert result.is_improved is True
        assert FailureCategory.AGENT_LOOP in result.resolved_failures or len(
            result.after_diagnosis.failures
        ) < len(result.before_diagnosis.failures)

    def test_verify_fix_successful_resolution(
        self, verification_engine: VerificationEngine
    ) -> None:
        # Before: tool error failure
        before_span = Span(
            span_id="s1",
            name="fetch_user",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="Database disconnected",
            tool_call=ToolCall(tool_name="fetch_user", arguments={"id": "1"}),
            tool_result=ToolResult(tool_name="fetch_user", error="Database disconnected"),
        )
        before_run = Run(run_id="r1", trace_id="t1", spans=[before_span])

        # After: successful tool execution
        after_span = Span(
            span_id="s1",
            name="fetch_user",
            kind=SpanKind.TOOL,
            status=SpanStatus.SUCCESS,
            tool_call=ToolCall(tool_name="fetch_user", arguments={"id": "1"}),
            tool_result=ToolResult(tool_name="fetch_user", output={"name": "Alice"}),
        )
        after_run = Run(run_id="r1", trace_id="t1", spans=[after_span])

        report: VerificationReport = verification_engine.verify_fix(before_run, after_run)

        assert report.is_verified is True
        assert FailureCategory.TOOL_ERROR in report.resolved_failures
        assert len(report.new_regressions) == 0
        assert report.severity_after in (Severity.INFO, Severity.LOW)

    def test_verify_fix_detects_regression(self, verification_engine: VerificationEngine) -> None:
        # Before: clean retrieval
        doc_before = RetrievedDocument(
            doc_id="d1",
            content="Solar panels convert sunlight directly into electrical energy cleanly.",
            score=0.95,
        )
        step_before = RetrievalStep(query="solar energy", documents=[doc_before])
        before_span = Span(
            span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step_before
        )
        before_run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[before_span],
            final_response=FinalResponse(
                text="Solar panels convert sunlight directly into electrical energy."
            ),
        )

        # After: new tool execution error introduced
        after_span = Span(
            span_id="s2",
            name="external_api",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="HTTP 500 Service Unavailable",
            tool_call=ToolCall(tool_name="external_api", arguments={"action": "query"}),
            tool_result=ToolResult(tool_name="external_api", error="HTTP 500 Service Unavailable"),
        )
        after_run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[before_span, after_span],
        )

        report: VerificationReport = verification_engine.verify_fix(before_run, after_run)

        assert report.is_verified is False
        assert FailureCategory.TOOL_ERROR in report.new_regressions

    def test_metric_comparison_computation(self, verification_engine: VerificationEngine) -> None:
        diag1 = DiagnosticEngine().diagnose(
            Run(
                run_id="r1",
                trace_id="t1",
                spans=[
                    Span(
                        span_id="s1",
                        name="ret",
                        kind=SpanKind.RETRIEVAL,
                        retrieval=RetrievalStep(
                            query="q",
                            documents=[
                                RetrievedDocument(doc_id="d1", content="text a", score=0.5),
                                RetrievedDocument(doc_id="d2", content="text b", score=0.5),
                            ],
                        ),
                    )
                ],
            )
        )
        diag2 = DiagnosticEngine().diagnose(
            Run(
                run_id="r1",
                trace_id="t1",
                spans=[
                    Span(
                        span_id="s1",
                        name="ret",
                        kind=SpanKind.RETRIEVAL,
                        retrieval=RetrievalStep(
                            query="q",
                            documents=[
                                RetrievedDocument(doc_id="d1", content="text a", score=0.95),
                                RetrievedDocument(doc_id="d2", content="text b", score=0.90),
                            ],
                        ),
                    )
                ],
            )
        )

        diffs = verification_engine._compare_metrics(diag1, diag2)
        assert len(diffs) > 0
        rel_diff = next((d for d in diffs if d.name == "retrieval_relevance_ratio"), None)
        if rel_diff and rel_diff.delta is not None:
            assert rel_diff.delta >= 0
