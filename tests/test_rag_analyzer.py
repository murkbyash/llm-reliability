"""Comprehensive tests for the RAG Analysis Engine."""

import pytest

from llm_reliability.models.enums import SpanKind
from llm_reliability.models.execution import RetrievalStep, RetrievedDocument
from llm_reliability.models.trace import Run, Span, Trace
from llm_reliability.rag import RAGAnalyzer, RetrievalMetrics


class TestRAGAnalyzer:
    """Tests for RAGAnalyzer deterministic metric extraction."""

    @pytest.fixture
    def analyzer(self) -> RAGAnalyzer:
        return RAGAnalyzer(relevance_threshold=0.70, duplicate_similarity_threshold=0.85)

    def test_empty_retrieval(self, analyzer: RAGAnalyzer) -> None:
        step = RetrievalStep(query="What is vector indexing?", documents=[], top_k=5)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.total_retrieved == 0
        assert metrics.requested_k == 5
        assert metrics.k_shortfall is True
        assert metrics.is_empty_retrieval is True
        assert metrics.is_low_relevance is True
        assert metrics.is_insufficient_context is True
        assert metrics.has_scores is False
        assert metrics.relevance_ratio == 0.0
        assert metrics.duplicate_count == 0

    def test_high_quality_retrieval(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(
                doc_id="d1",
                content="Vector search computes cosine similarity between dense embeddings.",
                score=0.95,
                rank=1,
            ),
            RetrievedDocument(
                doc_id="d2",
                content="Approximate Nearest Neighbors indexes enable sub-millisecond retrieval.",
                score=0.88,
                rank=2,
            ),
            RetrievedDocument(
                doc_id="d3",
                content="HNSW graphs construct hierarchical proximity layers for fast vector lookup.",
                score=0.82,
                rank=3,
            ),
        ]
        step = RetrievalStep(query="How does vector search work?", documents=docs, top_k=3)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.total_retrieved == 3
        assert metrics.k_shortfall is False
        assert metrics.is_empty_retrieval is False
        assert metrics.has_scores is True
        assert metrics.max_score == 0.95
        assert metrics.min_score == 0.82
        assert metrics.mean_score == pytest.approx(0.8833, abs=1e-3)
        assert metrics.relevance_ratio == 1.0
        assert metrics.relevant_documents == 3
        assert metrics.has_relevant_documents is True
        assert metrics.is_low_relevance is False
        assert metrics.duplicate_count == 0
        assert metrics.duplicate_ratio == 0.0
        assert metrics.context_characters > 100
        assert metrics.context_tokens_estimate > 20

    def test_low_relevance_retrieval(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(
                doc_id="d1", content="Random facts about medieval history.", score=0.45
            ),
            RetrievedDocument(
                doc_id="d2", content="Baking recipes for chocolate cookies.", score=0.35
            ),
            RetrievedDocument(doc_id="d3", content="How to change car engine oil.", score=0.25),
        ]
        step = RetrievalStep(
            query="How does transformer self-attention work?", documents=docs, top_k=3
        )
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.total_retrieved == 3
        assert metrics.max_score == 0.45
        assert metrics.relevant_documents == 0
        assert metrics.relevance_ratio == 0.0
        assert metrics.has_relevant_documents is False
        assert metrics.is_low_relevance is True

    def test_score_variance_and_dropoff_distribution(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(doc_id="d1", content="Directly relevant document chunk.", score=0.90),
            RetrievedDocument(
                doc_id="d2", content="Somewhat related background chunk.", score=0.60
            ),
            RetrievedDocument(doc_id="d3", content="Completely unrelated noise chunk.", score=0.30),
        ]
        step = RetrievalStep(query="Search query", documents=docs, top_k=3)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.mean_score == 0.60
        assert metrics.max_score == 0.90
        assert metrics.min_score == 0.30
        assert metrics.score_spread == 0.60
        assert metrics.score_dropoff_ratio == pytest.approx(0.6667, abs=1e-3)
        assert metrics.score_variance is not None
        assert metrics.score_variance > 0.0

    def test_top_k_shortfall_detection(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(doc_id="d1", content="Only one chunk returned.", score=0.85),
        ]
        step = RetrievalStep(query="Query", documents=docs, top_k=10)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.total_retrieved == 1
        assert metrics.requested_k == 10
        assert metrics.k_shortfall is True

    def test_exact_and_near_duplicate_detection(self, analyzer: RAGAnalyzer) -> None:
        content_a = (
            "Retrieval Augmented Generation combines retrieval systems with large language models."
        )
        content_exact = (
            "Retrieval Augmented Generation combines retrieval systems with large language models."
        )
        content_near = "Retrieval Augmented Generation combines retrieval systems with large language models efficiently."
        content_unique = (
            "Convolutional neural networks are specialized for 2D visual grid processing."
        )

        docs = [
            RetrievedDocument(doc_id="d1", content=content_a, score=0.95),
            RetrievedDocument(doc_id="d2", content=content_exact, score=0.95),
            RetrievedDocument(doc_id="d3", content=content_near, score=0.92),
            RetrievedDocument(doc_id="d4", content=content_unique, score=0.75),
        ]
        step = RetrievalStep(query="What is RAG?", documents=docs, top_k=4)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.total_retrieved == 4
        assert metrics.duplicate_count == 2
        assert metrics.duplicate_ratio == 0.50
        assert metrics.is_high_duplicate_ratio is True

    def test_empty_and_blank_document_chunks(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(doc_id="d1", content="Valid content chunk.", score=0.80),
            RetrievedDocument(doc_id="d2", content="", score=0.10),
            RetrievedDocument(doc_id="d3", content="   \n\t  ", score=0.15),
        ]
        step = RetrievalStep(query="Query", documents=docs, top_k=3)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.total_retrieved == 3
        assert metrics.empty_documents_count == 2

    def test_insufficient_context_detection(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(doc_id="d1", content="Tiny text", score=0.85),
        ]
        step = RetrievalStep(query="Query", documents=docs, top_k=1)
        metrics = analyzer.analyze_retrieval_step(step)

        assert metrics.context_characters == len("Tiny text")
        assert metrics.is_insufficient_context is True

    def test_analyze_trace_with_multiple_retrievals(self, analyzer: RAGAnalyzer) -> None:
        step1 = RetrievalStep(
            query="Query 1",
            documents=[
                RetrievedDocument(doc_id="d1", content="Valid chunk for query 1", score=0.90)
            ],
            top_k=1,
        )
        step2 = RetrievalStep(
            query="Query 2",
            documents=[
                RetrievedDocument(doc_id="d2", content="Valid chunk for query 2", score=0.85)
            ],
            top_k=1,
        )

        span1 = Span(span_id="s1", name="retrieval_1", kind=SpanKind.RETRIEVAL, retrieval=step1)
        span2 = Span(span_id="s2", name="retrieval_2", kind=SpanKind.RETRIEVAL, retrieval=step2)

        trace = Trace(
            trace_id="t-multi-ret",
            runs=[Run(run_id="r1", trace_id="t-multi-ret", spans=[span1, span2])],
        )

        results = analyzer.analyze_trace(trace)
        assert len(results) == 2
        assert results[0].max_score == 0.90
        assert results[1].max_score == 0.85

    def test_to_diagnostic_metrics_conversion(self, analyzer: RAGAnalyzer) -> None:
        docs = [
            RetrievedDocument(
                doc_id="d1", content="Good content chunk for diagnostic test.", score=0.85
            ),
            RetrievedDocument(doc_id="d2", content="Another good content chunk.", score=0.80),
        ]
        step = RetrievalStep(query="Testing diagnostics", documents=docs, top_k=2, latency_ms=45.0)
        ret_metrics: RetrievalMetrics = analyzer.analyze_retrieval_step(step)
        diagnostic_metrics = ret_metrics.to_diagnostic_metrics()

        assert len(diagnostic_metrics) >= 4
        metric_names = [m.name for m in diagnostic_metrics]
        assert "retrieval_total_count" in metric_names
        assert "retrieval_relevance_ratio" in metric_names
        assert "retrieval_duplicate_ratio" in metric_names
        assert "retrieval_context_tokens" in metric_names
        assert "retrieval_mean_score" in metric_names
        assert "retrieval_latency_ms" in metric_names

        for m in diagnostic_metrics:
            assert m.passed is True
