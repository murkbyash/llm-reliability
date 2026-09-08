"""Comprehensive tests for core data models: validation, serialization, deserialization, and edge cases."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llm_reliability.models import (
    Diagnosis,
    Evidence,
    EvidenceType,
    Failure,
    FailureCategory,
    FinalResponse,
    Hypothesis,
    LLMCall,
    Metric,
    Recommendation,
    RetrievalStep,
    RetrievedDocument,
    Run,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    TokenUsage,
    ToolCall,
    ToolResult,
    Trace,
)


class TestExecutionModels:
    """Tests for granular execution models (TokenUsage, RetrievedDocument, RetrievalStep, LLMCall, ToolCall, ToolResult, FinalResponse)."""

    def test_token_usage_valid(self) -> None:
        usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.002
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost_usd == 0.002

    def test_token_usage_invalid_negative(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-5, completion_tokens=10, total_tokens=5)

    def test_retrieved_document_valid(self) -> None:
        doc = RetrievedDocument(
            doc_id="doc-123",
            content="Vector embeddings are numerical representations of text.",
            score=0.88,
            rank=1,
            metadata={"source": "wiki.pdf", "page": 4},
        )
        assert doc.doc_id == "doc-123"
        assert doc.score == 0.88
        assert doc.rank == 1
        assert doc.metadata["page"] == 4

    def test_retrieval_step_valid(self) -> None:
        step = RetrievalStep(
            query="What is vector search?",
            documents=[
                RetrievedDocument(doc_id="d1", content="Chunk 1", score=0.9),
                RetrievedDocument(doc_id="d2", content="Chunk 2", score=0.7),
            ],
            top_k=2,
            retriever_name="hybrid-bm25-dense",
            latency_ms=45.2,
        )
        assert len(step.documents) == 2
        assert step.top_k == 2
        assert step.retriever_name == "hybrid-bm25-dense"

    def test_llm_call_valid(self) -> None:
        call = LLMCall(
            model="llama-3-8b",
            prompt="Explain RAG in one sentence.",
            response="RAG combines retrieval with generative LLMs.",
            temperature=0.7,
            token_usage=TokenUsage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
            latency_ms=320.5,
            raw_parameters={"top_p": 0.95},
        )
        assert call.model == "llama-3-8b"
        assert call.token_usage is not None
        assert call.token_usage.total_tokens == 25
        assert call.raw_parameters["top_p"] == 0.95

    def test_tool_call_and_result_valid(self) -> None:
        call = ToolCall(
            tool_name="database_lookup",
            arguments={"query": "SELECT * FROM users WHERE id=1"},
            call_id="call-42",
        )
        result = ToolResult(
            tool_name="database_lookup",
            output={"user_id": 1, "name": "Alice"},
            call_id="call-42",
            status=SpanStatus.SUCCESS,
            latency_ms=12.0,
        )
        assert call.tool_name == "database_lookup"
        assert result.output["name"] == "Alice"
        assert result.status == SpanStatus.SUCCESS

    def test_final_response_valid(self) -> None:
        resp = FinalResponse(
            text="The user name is Alice.",
            grounded=True,
            confidence=0.95,
            metadata={"latency_ms": 350},
        )
        assert resp.text == "The user name is Alice."
        assert resp.grounded is True
        assert resp.confidence == 0.95


class TestTraceHierarchyModels:
    """Tests for Span, Run, and Trace hierarchy and helper methods."""

    def test_span_root_and_error_properties(self) -> None:
        root_span = Span(
            span_id="span-root",
            name="pipeline_execution",
            kind=SpanKind.ROOT,
            status=SpanStatus.SUCCESS,
        )
        assert root_span.is_root is True
        assert root_span.is_error is False

        child_error_span = Span(
            span_id="span-retrieval",
            parent_span_id="span-root",
            name="retrieval_step",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.ERROR,
            error_message="Connection timeout to vector store",
        )
        assert child_error_span.is_root is False
        assert child_error_span.is_error is True

    def test_run_span_queries(self) -> None:
        span1 = Span(span_id="s1", name="root", kind=SpanKind.ROOT)
        span2 = Span(span_id="s2", parent_span_id="s1", name="retrieval", kind=SpanKind.RETRIEVAL)
        span3 = Span(span_id="s3", parent_span_id="s1", name="llm", kind=SpanKind.LLM)
        span4 = Span(span_id="s4", parent_span_id="s3", name="tool", kind=SpanKind.TOOL)

        run = Run(
            run_id="run-001",
            trace_id="trace-001",
            spans=[span1, span2, span3, span4],
            input_query="Test query",
        )

        assert run.get_span("s2") == span2
        assert run.get_span("non-existent") is None
        assert len(run.get_spans_by_kind(SpanKind.LLM)) == 1
        assert run.get_spans_by_kind(SpanKind.LLM)[0].span_id == "s3"
        assert len(run.get_root_spans()) == 1
        assert run.get_root_spans()[0].span_id == "s1"
        assert len(run.get_child_spans("s1")) == 2

    def test_trace_aggregations(self) -> None:
        now = datetime.now(timezone.utc)
        llm_call_1 = LLMCall(
            model="m1",
            prompt="p1",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.001
            ),
        )
        llm_call_2 = LLMCall(
            model="m2",
            prompt="p2",
            token_usage=TokenUsage(
                prompt_tokens=40, completion_tokens=30, total_tokens=70, cost_usd=0.002
            ),
        )
        retrieval = RetrievalStep(query="q", top_k=5)
        tool = ToolCall(tool_name="search", arguments={"q": "rag"})

        span_llm1 = Span(span_id="s1", name="llm1", kind=SpanKind.LLM, llm_call=llm_call_1)
        span_ret = Span(span_id="s2", name="ret", kind=SpanKind.RETRIEVAL, retrieval=retrieval)
        span_tool = Span(span_id="s3", name="tool", kind=SpanKind.TOOL, tool_call=tool)
        span_llm2 = Span(span_id="s4", name="llm2", kind=SpanKind.LLM, llm_call=llm_call_2)

        run1 = Run(run_id="r1", trace_id="t1", spans=[span_llm1, span_ret])
        run2 = Run(run_id="r2", trace_id="t1", spans=[span_tool, span_llm2])

        trace = Trace(trace_id="t1", runs=[run1, run2], created_at=now)

        assert len(trace.get_all_spans()) == 4
        assert len(trace.get_llm_calls()) == 2
        assert len(trace.get_retrievals()) == 1
        assert len(trace.get_tool_calls()) == 1

        usage = trace.aggregate_token_usage()
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 100
        assert usage.cost_usd == pytest.approx(0.003)


class TestDiagnosticModels:
    """Tests for Metric, Evidence, Failure, Recommendation, Hypothesis, and Diagnosis models."""

    def test_metric_and_evidence_valid(self) -> None:
        metric = Metric(
            name="retrieval_relevance_ratio",
            value=0.25,
            threshold=0.6,
            unit="ratio",
            passed=False,
            details={"total_retrieved": 4, "relevant_retrieved": 1},
        )
        evidence = Evidence(
            evidence_type=EvidenceType.RELEVANCE_SCORE,
            description="Only 1 of 4 retrieved chunks was relevant to query.",
            metric=metric,
            supporting_data={"doc_ids": ["doc-1"], "query": "What is RAG?"},
            span_id="span-retrieval-1",
        )
        assert evidence.evidence_type == EvidenceType.RELEVANCE_SCORE
        assert evidence.metric is not None
        assert evidence.metric.passed is False

    def test_failure_and_recommendation_valid(self) -> None:
        rec = Recommendation(
            title="Increase retrieval candidate count",
            description="Increase top_k from 3 to 10 and consider reranking.",
            action_type="RETRIEVER_CONFIG",
            priority=1,
            rationale="Low relevance ratio detected in initial candidate set.",
        )
        failure = Failure(
            category=FailureCategory.RETRIEVAL_FAILURE,
            severity=Severity.HIGH,
            message="Retriever failed to surface necessary documents.",
            span_id="span-ret-1",
            evidence=[],
        )
        assert failure.category == FailureCategory.RETRIEVAL_FAILURE
        assert failure.severity == Severity.HIGH
        assert rec.priority == 1

    def test_diagnosis_valid_and_confidence_bounds(self) -> None:
        diag = Diagnosis(
            root_cause=FailureCategory.RETRIEVAL_FAILURE,
            confidence=0.88,
            summary="Retrieval failed due to low semantic match with query.",
            uncertainty_note="Evidence indicates low similarity scores across all retrieved documents.",
            failures=[],
            evidence=[],
            alternative_hypotheses=[
                Hypothesis(
                    category=FailureCategory.CONTEXT_FAILURE,
                    confidence=0.10,
                    rationale="Context window limit",
                ),
            ],
            recommendations=[
                Recommendation(
                    title="Evaluate reranking",
                    description="Add cross-encoder reranker.",
                    action_type="RERANKING",
                )
            ],
            metrics=[Metric(name="relevance", value=0.2)],
            trace_id="trace-123",
        )
        assert diag.root_cause == FailureCategory.RETRIEVAL_FAILURE
        assert diag.confidence == 0.88
        assert len(diag.alternative_hypotheses) == 1
        assert len(diag.recommendations) == 1

    def test_diagnosis_invalid_confidence(self) -> None:
        with pytest.raises(ValidationError):
            Diagnosis(
                root_cause=FailureCategory.RETRIEVAL_FAILURE,
                confidence=1.5,  # Must be <= 1.0
                summary="Invalid",
            )
        with pytest.raises(ValidationError):
            Diagnosis(
                root_cause=FailureCategory.RETRIEVAL_FAILURE,
                confidence=-0.1,  # Must be >= 0.0
                summary="Invalid",
            )


class TestSerializationAndDeserialization:
    """Tests for JSON serialization, deserialization, round-trips, and backward compatibility."""

    def test_trace_json_roundtrip(self) -> None:
        now = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)
        original_trace = Trace(
            trace_id="trace-roundtrip-1",
            name="rag_qa_pipeline",
            runs=[
                Run(
                    run_id="run-1",
                    trace_id="trace-roundtrip-1",
                    input_query="How does indexing work?",
                    spans=[
                        Span(
                            span_id="span-1",
                            name="vector_lookup",
                            kind=SpanKind.RETRIEVAL,
                            retrieval=RetrievalStep(
                                query="How does indexing work?",
                                documents=[
                                    RetrievedDocument(
                                        doc_id="d1",
                                        content="Indexing creates an inverted index.",
                                        score=0.85,
                                    ),
                                ],
                                top_k=1,
                            ),
                        ),
                        Span(
                            span_id="span-2",
                            parent_span_id="span-1",
                            name="generate_answer",
                            kind=SpanKind.LLM,
                            llm_call=LLMCall(
                                model="gpt-4-turbo",
                                prompt="Summarize indexing.",
                                response="Indexing organizes content for rapid retrieval.",
                                token_usage=TokenUsage(
                                    prompt_tokens=20, completion_tokens=15, total_tokens=35
                                ),
                            ),
                        ),
                    ],
                    final_response=FinalResponse(
                        text="Indexing organizes content for rapid retrieval.", grounded=True
                    ),
                )
            ],
            created_at=now,
        )

        json_str = original_trace.model_dump_json()
        assert isinstance(json_str, str)

        restored_trace = Trace.model_validate_json(json_str)
        assert restored_trace.trace_id == original_trace.trace_id
        assert len(restored_trace.runs) == 1
        assert len(restored_trace.runs[0].spans) == 2
        assert restored_trace.runs[0].spans[0].retrieval is not None
        assert restored_trace.runs[0].spans[0].retrieval.documents[0].score == 0.85
        assert restored_trace.runs[0].spans[1].llm_call is not None
        assert restored_trace.runs[0].spans[1].llm_call.model == "gpt-4-turbo"
        assert restored_trace.created_at == now

    def test_backward_compatibility_with_extra_fields(self) -> None:
        """Verify models accept unknown/extra fields gracefully without crashing."""
        raw_data = {
            "span_id": "span-extra-1",
            "name": "custom_step",
            "kind": "custom",
            "future_unrecognized_field": {"arbitrary": "data"},
            "telemetry_flag": True,
        }
        span = Span.model_validate(raw_data)
        assert span.span_id == "span-extra-1"
        assert getattr(span, "future_unrecognized_field", None) == {"arbitrary": "data"}

    def test_diagnosis_json_roundtrip(self) -> None:
        diag = Diagnosis(
            root_cause=FailureCategory.GROUNDING_FAILURE,
            confidence=0.92,
            summary="LLM hallucinated facts not present in context.",
            uncertainty_note="Grounding overlap is below 0.3.",
            failures=[
                Failure(
                    category=FailureCategory.GROUNDING_FAILURE,
                    severity=Severity.CRITICAL,
                    message="Answer claims the moon is made of cheese.",
                    evidence=[
                        Evidence(
                            evidence_type=EvidenceType.HALLUCINATION_OVERLAP,
                            description="Extracted key entities absent from retrieved context.",
                        )
                    ],
                )
            ],
            recommendations=[
                Recommendation(
                    title="Instruct LLM to cite context explicitly",
                    description="Update prompt with citation constraints.",
                    action_type="PROMPT_TUNING",
                    priority=1,
                )
            ],
        )

        json_str = diag.model_dump_json()
        restored = Diagnosis.model_validate_json(json_str)
        assert restored.root_cause == FailureCategory.GROUNDING_FAILURE
        assert restored.confidence == 0.92
        assert len(restored.failures) == 1
        assert restored.failures[0].evidence[0].evidence_type == EvidenceType.HALLUCINATION_OVERLAP
        assert len(restored.recommendations) == 1
