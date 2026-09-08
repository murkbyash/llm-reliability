"""Tests for unified root cause diagnostic engine."""

import pytest

from llm_reliability import DiagnosticEngine, diagnose
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import FailureCategory, Severity, SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    LLMCall,
    RetrievalStep,
    RetrievedDocument,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span, Trace


class TestDiagnosticEngine:
    """Deterministic tests for root cause diagnostic orchestrator."""

    @pytest.fixture
    def engine(self) -> DiagnosticEngine:
        return DiagnosticEngine()

    def test_diagnose_clean_run(self, engine: DiagnosticEngine) -> None:
        doc = RetrievedDocument(
            doc_id="d1",
            content="Rust is a systems programming language focusing on safety and performance.",
            score=0.92,
        )
        step = RetrievalStep(query="What is Rust?", documents=[doc])
        retrieval_span = Span(
            span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step
        )
        llm_span = Span(
            span_id="s2",
            name="generate",
            kind=SpanKind.LLM,
            llm_call=LLMCall(
                model="gpt-4o",
                prompt="What is Rust?",
                response="Rust is a systems programming language focusing on safety.",
            ),
        )
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[retrieval_span, llm_span],
            final_response=FinalResponse(
                text="Rust is a systems programming language focusing on safety."
            ),
        )

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.NONE
        assert diagnosis.severity == Severity.INFO
        assert len(diagnosis.failures) == 0
        assert len(diagnosis.hypotheses) == 0
        assert len(diagnosis.metrics) > 0

    def test_diagnose_empty_retrieval_failure(self, engine: DiagnosticEngine) -> None:
        step = RetrievalStep(query="Find tax documents", documents=[])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.RETRIEVAL_FAILURE
        assert diagnosis.severity == Severity.HIGH
        assert len(diagnosis.failures) == 1
        assert diagnosis.hypotheses[0].rank == 1
        assert diagnosis.hypotheses[0].confidence >= 0.90
        assert len(diagnosis.evidence) >= 1

    def test_diagnose_grounding_contradiction(self, engine: DiagnosticEngine) -> None:
        doc = RetrievedDocument(
            doc_id="d1",
            content="The release deployment succeeded and all servers are healthy.",
            score=0.88,
        )
        step = RetrievalStep(query="status", documents=[doc])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(
                text="The release deployment failed and servers are not healthy."
            ),
        )

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.HALLUCINATION
        assert diagnosis.severity == Severity.HIGH
        assert len(diagnosis.failures) >= 1
        assert any(h.category == FailureCategory.HALLUCINATION for h in diagnosis.hypotheses)

    def test_diagnose_grounding_unsupported(self, engine: DiagnosticEngine) -> None:
        doc = RetrievedDocument(
            doc_id="d1",
            content="Water freezes at 0 degrees Celsius.",
            score=0.85,
        )
        step = RetrievalStep(query="water", documents=[doc])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(
                text="Tesla stock jumped 45% following stellar Q3 earnings report."
            ),
        )

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.GROUNDING_FAILURE
        assert diagnosis.severity == Severity.HIGH
        assert any(h.category == FailureCategory.GROUNDING_FAILURE for h in diagnosis.hypotheses)

    def test_diagnose_agent_loop_failure(self, engine: DiagnosticEngine) -> None:
        spans = [
            Span(
                span_id=f"s{i}",
                name="search_users",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search_users", arguments={"q": "john"}),
            )
            for i in range(3)
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.AGENT_LOOP
        assert diagnosis.severity in (Severity.HIGH, Severity.CRITICAL)
        assert len(diagnosis.failures) >= 1
        assert diagnosis.hypotheses[0].category == FailureCategory.AGENT_LOOP

    def test_diagnose_tool_execution_failure(self, engine: DiagnosticEngine) -> None:
        span = Span(
            span_id="s1",
            name="send_payment",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="Gateway timeout",
            tool_call=ToolCall(tool_name="send_payment", arguments={"amount": 500}),
            tool_result=ToolResult(tool_name="send_payment", error="Gateway timeout"),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.TOOL_ERROR
        assert diagnosis.severity == Severity.HIGH
        assert any(h.category == FailureCategory.TOOL_ERROR for h in diagnosis.hypotheses)

    def test_diagnose_tool_argument_schema_violation(self, engine: DiagnosticEngine) -> None:
        span = Span(
            span_id="s1",
            name="execute_sql",
            kind=SpanKind.TOOL,
            tool_call=ToolCall(tool_name="execute_sql", arguments="{ bad json syntax "),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.SCHEMA_VIOLATION
        assert diagnosis.severity == Severity.HIGH
        assert any(h.category == FailureCategory.SCHEMA_VIOLATION for h in diagnosis.hypotheses)

    def test_diagnose_llm_span_error(self, engine: DiagnosticEngine) -> None:
        span = Span(
            span_id="s1",
            name="chat_completion",
            kind=SpanKind.LLM,
            status=SpanStatus.ERROR,
            error_message="Rate limit exceeded: 429 Too Many Requests",
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = engine.diagnose(run)

        assert diagnosis.primary_category == FailureCategory.LLM_CALL_FAILURE
        assert diagnosis.severity == Severity.CRITICAL
        assert len(diagnosis.failures) == 1

    def test_diagnose_multi_run_trace_aggregation(self, engine: DiagnosticEngine) -> None:
        # Run 1: clean
        run1 = Run(
            run_id="r1",
            trace_id="t1",
            spans=[Span(span_id="s1", name="step", kind=SpanKind.CUSTOM)],
        )
        # Run 2: empty retrieval failure
        step = RetrievalStep(query="missing", documents=[])
        run2 = Run(
            run_id="r2",
            trace_id="t1",
            spans=[Span(span_id="s2", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)],
        )
        trace = Trace(trace_id="t1", runs=[run1, run2])

        diagnosis: Diagnosis = engine.diagnose(trace)

        assert diagnosis.primary_category == FailureCategory.RETRIEVAL_FAILURE
        assert diagnosis.severity == Severity.HIGH
        assert len(diagnosis.failures) >= 1
        assert "Analyzed 2 runs" in diagnosis.summary

    def test_diagnose_convenience_function_with_dict(self) -> None:
        payload = {
            "trace_id": "trace-dict-1",
            "spans": [
                {
                    "span_id": "sp1",
                    "name": "payment_gateway",
                    "kind": "tool",
                    "status": "error",
                    "error_message": "Insufficient funds error",
                    "tool_call": {"tool_name": "payment_gateway", "arguments": {"user": "u1"}},
                }
            ],
        }

        diagnosis: Diagnosis = diagnose(payload)

        assert diagnosis.trace_id == "trace-dict-1"
        assert diagnosis.primary_category == FailureCategory.TOOL_ERROR
        assert diagnosis.severity == Severity.HIGH
        assert len(diagnosis.failures) == 1
