"""Tests for developer recommendation engine."""

import pytest

from llm_reliability import DiagnosticEngine, RecommendationEngine, diagnose
from llm_reliability.models.diagnosis import Diagnosis, Recommendation
from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    RetrievalStep,
    RetrievedDocument,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span


class TestRecommendationEngine:
    """Deterministic tests for developer remediation recommendations."""

    @pytest.fixture
    def engine(self) -> RecommendationEngine:
        return RecommendationEngine()

    @pytest.fixture
    def diagnostic_engine(self) -> DiagnosticEngine:
        return DiagnosticEngine()

    def test_rag_empty_retrieval_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        step = RetrievalStep(query="tax form 1099", documents=[])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert len(recs) >= 1
        assert any(r.action_type == "RETRIEVER_CONFIG" for r in recs)
        assert any(
            "index" in r.description.lower() or "filter" in r.description.lower() for r in recs
        )

    def test_rag_duplicate_context_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        docs = [
            RetrievedDocument(
                doc_id="d1", content="Identical policy paragraph text for users.", score=0.85
            ),
            RetrievedDocument(
                doc_id="d2", content="Identical policy paragraph text for users.", score=0.84
            ),
            RetrievedDocument(
                doc_id="d3", content="Identical policy paragraph text for users.", score=0.83
            ),
        ]
        step = RetrievalStep(query="policy", documents=docs)
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert any(r.action_type == "CHUNKING_CONFIG" for r in recs)
        assert any("deduplicat" in r.description.lower() for r in recs)

    def test_grounding_unsupported_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        doc = RetrievedDocument(doc_id="d1", content="Apples grow on apple trees.", score=0.9)
        step = RetrievalStep(query="apples", documents=[doc])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(text="Oranges contain 500mg of Vitamin C per unit."),
        )

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert any(r.action_type == "PROMPT_TUNING" for r in recs)
        assert any(
            "grounding" in r.title.lower() or "refusal" in r.description.lower() for r in recs
        )

    def test_grounding_contradiction_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        doc = RetrievedDocument(
            doc_id="d1", content="The flight departure was approved.", score=0.9
        )
        step = RetrievalStep(query="flight", documents=[doc])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(text="The flight departure was denied and failed."),
        )

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert any(r.action_type == "PROMPT_TUNING" for r in recs)
        assert any(
            "temperature" in r.description.lower() or "strict" in r.title.lower() for r in recs
        )

    def test_agent_loop_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        spans = [
            Span(
                span_id=f"s{i}",
                name="search",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search", arguments={"q": "same"}),
            )
            for i in range(3)
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert any(r.action_type == "AGENT_GUARDRAIL" for r in recs)
        assert any(
            "step limit" in r.description.lower() or "guardrail" in r.title.lower() for r in recs
        )

    def test_tool_execution_error_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        span = Span(
            span_id="s1",
            name="send_sms",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="HTTP 500 Internal Error",
            tool_call=ToolCall(tool_name="send_sms", arguments={"phone": "123"}),
            tool_result=ToolResult(tool_name="send_sms", error="HTTP 500 Internal Error"),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert any(r.action_type == "TOOL_RETRY_POLICY" for r in recs)
        assert any("backoff" in r.description.lower() for r in recs)

    def test_tool_schema_violation_recommendation(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        span = Span(
            span_id="s1",
            name="sql_query",
            kind=SpanKind.TOOL,
            tool_call=ToolCall(tool_name="sql_query", arguments="{ unclosed json "),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert any(r.action_type == "TOOL_SCHEMA_DEFINITION" for r in recs)
        assert any("pydantic" in r.description.lower() or "schema" in r.title.lower() for r in recs)

    def test_clean_run_empty_recommendations(
        self, diagnostic_engine: DiagnosticEngine, engine: RecommendationEngine
    ) -> None:
        doc = RetrievedDocument(doc_id="d1", content="Python is popular.", score=0.95)
        step = RetrievalStep(query="python", documents=[doc])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(text="Python is popular."),
        )

        diagnosis: Diagnosis = diagnostic_engine.diagnose(run)
        recs: list[Recommendation] = engine.generate_recommendations(diagnosis)

        assert len(recs) == 0

    def test_recommendation_priority_sorting(self, engine: RecommendationEngine) -> None:
        rec1 = Recommendation(title="Low priority action", description="Desc 1", priority=3)
        rec2 = Recommendation(title="High priority action", description="Desc 2", priority=1)
        rec3 = Recommendation(title="Medium priority action", description="Desc 3", priority=2)

        sorted_recs = sorted([rec1, rec2, rec3], key=lambda r: r.priority)
        assert sorted_recs[0].priority == 1
        assert sorted_recs[1].priority == 2
        assert sorted_recs[2].priority == 3

    def test_diagnose_integration_attaches_recommendations(self) -> None:
        payload = {
            "trace_id": "t-rec-test",
            "spans": [
                {
                    "span_id": "sp1",
                    "name": "search_db",
                    "kind": "tool",
                    "status": "error",
                    "error_message": "Connection refused",
                    "tool_call": {"tool_name": "search_db", "arguments": {"db": "prod"}},
                }
            ],
        }

        diagnosis: Diagnosis = diagnose(payload)

        assert len(diagnosis.recommendations) >= 1
        assert diagnosis.recommendations[0].priority == 1
        assert diagnosis.recommendations[0].rationale is not None
