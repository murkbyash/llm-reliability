"""RAG analysis engine computing deterministic statistical and structural metrics on retrieval traces."""

import math
import re
import statistics

from llm_reliability.models.enums import SpanKind
from llm_reliability.models.execution import RetrievalStep, RetrievedDocument
from llm_reliability.models.trace import Trace
from llm_reliability.rag.metrics import RetrievalMetrics


class RAGAnalyzer:
    """Deterministic analyzer evaluating retrieval quality, scores, top-k behavior, and deduplication."""

    def __init__(
        self,
        relevance_threshold: float = 0.70,
        duplicate_similarity_threshold: float = 0.85,
        min_context_characters: int = 50,
        max_duplicate_ratio_threshold: float = 0.30,
        min_relevance_ratio_threshold: float = 0.40,
    ) -> None:
        """Initialize analyzer with configurable heuristic thresholds.

        Args:
            relevance_threshold: Minimum similarity score to consider a document relevant (default: 0.70).
            duplicate_similarity_threshold: Jaccard similarity threshold for near-duplicate detection (default: 0.85).
            min_context_characters: Minimum characters expected in retrieved context (default: 50).
            max_duplicate_ratio_threshold: Warning threshold for duplicate content proportion (default: 0.30).
            min_relevance_ratio_threshold: Warning threshold for low relevance proportion (default: 0.40).
        """
        self.relevance_threshold = relevance_threshold
        self.duplicate_similarity_threshold = duplicate_similarity_threshold
        self.min_context_characters = min_context_characters
        self.max_duplicate_ratio_threshold = max_duplicate_ratio_threshold
        self.min_relevance_ratio_threshold = min_relevance_ratio_threshold

    def analyze_retrieval_step(self, step: RetrievalStep) -> RetrievalMetrics:
        """Compute comprehensive metrics for a single retrieval step.

        Args:
            step: The RetrievalStep execution model.

        Returns:
            Structured RetrievalMetrics instance.
        """
        docs = step.documents
        total_retrieved = len(docs)
        requested_k = step.top_k
        k_shortfall = bool(requested_k is not None and total_retrieved < requested_k)
        is_empty_retrieval = total_retrieved == 0

        # Score calculations
        scores: list[float] = [d.score for d in docs if d.score is not None]
        has_scores = len(scores) > 0

        mean_score: float | None = None
        max_score: float | None = None
        min_score: float | None = None
        score_variance: float | None = None
        score_spread: float | None = None
        score_dropoff_ratio: float | None = None

        if has_scores:
            mean_score = round(statistics.mean(scores), 4)
            max_score = round(max(scores), 4)
            min_score = round(min(scores), 4)
            score_spread = round(max_score - min_score, 4)
            score_variance = round(statistics.pvariance(scores), 6) if len(scores) > 1 else 0.0

            # Dropoff from first document to last document
            if len(scores) >= 2 and scores[0] > 0.0:
                score_dropoff_ratio = round(max(0.0, (scores[0] - scores[-1]) / scores[0]), 4)
            else:
                score_dropoff_ratio = 0.0

        # Relevance calculation
        relevant_docs_count = 0
        if has_scores:
            relevant_docs_count = sum(1 for s in scores if s >= self.relevance_threshold)
        else:
            # If no scores present, non-empty docs default to candidate pool
            relevant_docs_count = sum(1 for d in docs if bool(d.content and d.content.strip()))

        relevance_ratio = (
            round(relevant_docs_count / total_retrieved, 4) if total_retrieved > 0 else 0.0
        )
        has_relevant_documents = relevant_docs_count > 0

        # Deduplication and redundancy
        duplicate_count = self._detect_duplicates(docs)
        duplicate_ratio = (
            round(duplicate_count / total_retrieved, 4) if total_retrieved > 0 else 0.0
        )

        # Context volume and token estimation
        context_chars = sum(len(d.content) for d in docs if d.content)
        context_tokens = sum(self._estimate_tokens(d.content) for d in docs if d.content)
        empty_docs_count = sum(1 for d in docs if not d.content or not d.content.strip())

        # Quality flags
        is_low_relevance = (
            is_empty_retrieval
            or (has_scores and max_score is not None and max_score < self.relevance_threshold)
            or (total_retrieved > 0 and relevance_ratio < self.min_relevance_ratio_threshold)
        )
        is_high_duplicate_ratio = duplicate_ratio >= self.max_duplicate_ratio_threshold
        is_insufficient_context = is_empty_retrieval or (
            context_chars < self.min_context_characters
        )

        return RetrievalMetrics(
            total_retrieved=total_retrieved,
            requested_k=requested_k,
            k_shortfall=k_shortfall,
            has_scores=has_scores,
            mean_score=mean_score,
            max_score=max_score,
            min_score=min_score,
            score_variance=score_variance,
            score_spread=score_spread,
            score_dropoff_ratio=score_dropoff_ratio,
            relevance_threshold=self.relevance_threshold,
            relevant_documents=relevant_docs_count,
            relevance_ratio=relevance_ratio,
            has_relevant_documents=has_relevant_documents,
            duplicate_count=duplicate_count,
            duplicate_ratio=duplicate_ratio,
            context_characters=context_chars,
            context_tokens_estimate=context_tokens,
            empty_documents_count=empty_docs_count,
            is_empty_retrieval=is_empty_retrieval,
            is_low_relevance=is_low_relevance,
            is_high_duplicate_ratio=is_high_duplicate_ratio,
            is_insufficient_context=is_insufficient_context,
            latency_ms=step.latency_ms,
            retriever_name=step.retriever_name,
        )

    def analyze_trace(self, trace: Trace) -> list[RetrievalMetrics]:
        """Analyze all retrieval operations detected within a trace."""
        results: list[RetrievalMetrics] = []
        for retrieval_step in trace.get_retrievals():
            results.append(self.analyze_retrieval_step(retrieval_step))

        # Check spans that might have retrieval kinds without typed payload
        if not results:
            for span in trace.get_all_spans():
                if span.kind == SpanKind.RETRIEVAL and span.retrieval is None:
                    # Synthesize basic step from attributes if available
                    query = str(span.attributes.get("query") or "")
                    step = RetrievalStep(
                        query=query,
                        documents=[],
                        latency_ms=span.duration_ms,
                    )
                    results.append(self.analyze_retrieval_step(step))

        return results

    def _detect_duplicates(self, documents: list[RetrievedDocument]) -> int:
        """Count identical or near-duplicate documents using Jaccard word n-gram similarity."""
        if len(documents) <= 1:
            return 0

        duplicates_found = 0
        seen_exact: set[str] = set()
        token_sets: list[set[str]] = []

        for doc in documents:
            text = (doc.content or "").strip().lower()
            if not text:
                continue

            # Exact duplicate check
            if text in seen_exact:
                duplicates_found += 1
                continue
            seen_exact.add(text)

            # Near duplicate check using word-level set comparison
            words = set(re.findall(r"\w+", text))
            if not words:
                continue

            is_near_duplicate = False
            for existing_words in token_sets:
                intersection = len(words & existing_words)
                union = len(words | existing_words)
                if union > 0 and (intersection / union) >= self.duplicate_similarity_threshold:
                    is_near_duplicate = True
                    break

            if is_near_duplicate:
                duplicates_found += 1
            else:
                token_sets.append(words)

        return duplicates_found

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count deterministically without requiring an external tokenizer library."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        char_estimate = math.ceil(chars / 4.0)
        return max(1, max(words, char_estimate))
