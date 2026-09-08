"""Tests for grounding, faithfulness, and answer support analysis."""

import pytest

from llm_reliability.grounding import (
    GroundingAnalyzer,
    SupportStatus,
)
from llm_reliability.models.enums import EvidenceType, SpanKind
from llm_reliability.models.execution import FinalResponse, RetrievalStep, RetrievedDocument
from llm_reliability.models.trace import Run, Span, Trace


class TestGroundingAnalyzer:
    """Deterministic tests for answer grounding against context."""

    @pytest.fixture
    def analyzer(self) -> GroundingAnalyzer:
        return GroundingAnalyzer(supported_threshold=0.60, partial_threshold=0.30)

    def test_fully_supported_answer(self, analyzer: GroundingAnalyzer) -> None:
        context = [
            "Python 3.13 introduces an experimental JIT compiler to speed up execution.",
            "It also allows disabling the Global Interpreter Lock (GIL) via free-threading build mode.",
        ]
        response = (
            "Python 3.13 introduces an experimental JIT compiler. "
            "Additionally, users can disable the GIL in free-threading mode."
        )

        metrics = analyzer.analyze_response(response, context)

        assert metrics.status == SupportStatus.SUPPORTED
        assert metrics.grounding_score >= 0.70
        assert metrics.total_claims == 2
        assert metrics.supported_claims >= 1
        assert metrics.unsupported_claims == 0
        assert metrics.contradicted_claims == 0
        assert len(metrics.hallucinated_entities) == 0

    def test_unsupported_hallucinated_answer(self, analyzer: GroundingAnalyzer) -> None:
        context = [
            "The solar system consists of the Sun and eight primary planets orbiting it.",
            "Jupiter is the largest planet, followed by Saturn.",
        ]
        response = (
            "Acme Corporation reported annual revenue of $500M in 2025. "
            "Their quarterly profit margin increased by 45%."
        )

        metrics = analyzer.analyze_response(response, context)

        assert metrics.status == SupportStatus.UNSUPPORTED
        assert metrics.grounding_score < 0.30
        assert metrics.total_claims == 2
        assert metrics.supported_claims == 0
        assert metrics.unsupported_claims == 2
        assert len(metrics.hallucinated_entities) > 0

    def test_partially_supported_answer(self, analyzer: GroundingAnalyzer) -> None:
        context = [
            "PostgreSQL is an open-source object-relational database system.",
        ]
        response = (
            "PostgreSQL is an open-source object-relational database system. "
            "It was originally created in 1842 by Victorian railroad engineers."
        )

        metrics = analyzer.analyze_response(response, context)

        assert metrics.status == SupportStatus.PARTIALLY_SUPPORTED
        assert 0.30 <= metrics.grounding_score < 0.85
        assert metrics.total_claims == 2
        assert metrics.supported_claims == 1
        assert metrics.unsupported_claims == 1
        assert "1842" in metrics.hallucinated_entities

    def test_contradiction_detection_polarity(self, analyzer: GroundingAnalyzer) -> None:
        context = "The database migration completed successfully and all tables passed validation."
        response = "The database migration failed and tables were not passed."

        metrics = analyzer.analyze_response(response, context)

        assert metrics.status == SupportStatus.CONTRADICTED
        assert metrics.contradicted_claims >= 1

    def test_contradiction_detection_numeric(self, analyzer: GroundingAnalyzer) -> None:
        context = "The server query latency was 25ms during peak traffic."
        response = "The server query latency was 4500ms during peak traffic."

        metrics = analyzer.analyze_response(response, context)

        assert metrics.status == SupportStatus.CONTRADICTED
        assert metrics.contradicted_claims >= 1

    def test_empty_context_handling(self, analyzer: GroundingAnalyzer) -> None:
        response = "The earth revolves around the sun."
        metrics = analyzer.analyze_response(response, context=[])

        assert metrics.status == SupportStatus.UNSUPPORTED
        assert metrics.grounding_score == 0.0
        assert metrics.total_claims == 1

    def test_empty_response_handling(self, analyzer: GroundingAnalyzer) -> None:
        metrics = analyzer.analyze_response("", context="Some context text.")
        assert metrics.status == SupportStatus.SUPPORTED
        assert metrics.total_claims == 0
        assert metrics.grounding_score == 1.0

    def test_analyze_trace_grounding(self, analyzer: GroundingAnalyzer) -> None:
        doc = RetrievedDocument(
            doc_id="d1",
            content="Redis is an in-memory key-value data store used as a database and cache.",
        )
        step = RetrievalStep(query="What is Redis?", documents=[doc])
        span = Span(span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL, retrieval=step)

        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(
                text="Redis is an in-memory key-value data store used as a cache."
            ),
        )
        trace = Trace(trace_id="t1", runs=[run])

        results = analyzer.analyze_trace(trace)
        assert len(results) == 1
        assert results[0].status == SupportStatus.SUPPORTED
        assert results[0].grounding_score >= 0.70

    def test_to_diagnostic_metrics_and_evidence(self, analyzer: GroundingAnalyzer) -> None:
        context = "The capital of France is Paris."
        response = "The capital of France is London with 9000 residents."
        metrics = analyzer.analyze_response(response, context)

        # Metrics conversion
        diag_metrics = metrics.to_diagnostic_metrics()
        assert len(diag_metrics) == 4
        metric_names = [m.name for m in diag_metrics]
        assert "answer_grounding_score" in metric_names
        assert "answer_unsupported_claims_count" in metric_names
        assert "answer_contradictions_count" in metric_names
        assert "answer_hallucinated_entities_count" in metric_names

        # Evidence conversion
        evidence = metrics.to_evidence(span_id="span-test-1")
        assert len(evidence) >= 1
        assert any(
            e.evidence_type
            in (
                EvidenceType.GROUNDING_DEFICIT,
                EvidenceType.CONTRADICTION,
                EvidenceType.HALLUCINATION_OVERLAP,
            )
            for e in evidence
        )
        for ev in evidence:
            assert ev.span_id == "span-test-1"
