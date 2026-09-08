"""Structured RAG retrieval metrics model."""

from pydantic import BaseModel, ConfigDict, Field

from llm_reliability.models.diagnosis import Metric


class RetrievalMetrics(BaseModel):
    """Structured metrics computed from a RAG retrieval step."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Retrieval counts
    total_retrieved: int = Field(default=0, ge=0, description="Total number of documents returned")
    requested_k: int | None = Field(default=None, ge=1, description="Requested top-k parameter")
    k_shortfall: bool = Field(
        default=False, description="True if retrieved count is less than requested_k"
    )

    # Score statistics
    has_scores: bool = Field(
        default=False, description="Whether documents included similarity scores"
    )
    mean_score: float | None = Field(default=None, description="Average similarity/relevance score")
    max_score: float | None = Field(default=None, description="Highest similarity score")
    min_score: float | None = Field(default=None, description="Lowest similarity score")
    score_variance: float | None = Field(
        default=None, ge=0.0, description="Variance across document scores"
    )
    score_spread: float | None = Field(
        default=None, description="Difference between max and min score"
    )
    score_dropoff_ratio: float | None = Field(
        default=None,
        description="Relative dropoff between top-ranked and bottom-ranked document score",
    )

    # Relevance & noise
    relevance_threshold: float = Field(
        default=0.70, description="Threshold used to determine relevance"
    )
    relevant_documents: int = Field(
        default=0, ge=0, description="Count of documents meeting relevance threshold"
    )
    relevance_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Ratio of relevant documents"
    )
    has_relevant_documents: bool = Field(
        default=False, description="True if at least one document is relevant"
    )

    # Deduplication
    duplicate_count: int = Field(
        default=0, ge=0, description="Number of duplicate or near-duplicate chunks"
    )
    duplicate_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Ratio of duplicate chunks"
    )

    # Context volume
    context_characters: int = Field(
        default=0, ge=0, description="Total characters across retrieved documents"
    )
    context_tokens_estimate: int = Field(default=0, ge=0, description="Estimated total token count")
    empty_documents_count: int = Field(
        default=0, ge=0, description="Number of empty or whitespace-only documents"
    )

    # Quality indicators
    is_empty_retrieval: bool = Field(
        default=False, description="True if zero documents were retrieved"
    )
    is_low_relevance: bool = Field(
        default=False, description="True if relevance ratio or max score is low"
    )
    is_high_duplicate_ratio: bool = Field(
        default=False, description="True if duplicate ratio exceeds threshold"
    )
    is_insufficient_context: bool = Field(
        default=False, description="True if context volume is too small"
    )

    # Latency
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="Retrieval latency in milliseconds"
    )
    retriever_name: str | None = Field(default=None, description="Name or identifier of retriever")

    def to_diagnostic_metrics(self) -> list[Metric]:
        """Convert structured RAG metrics into standard diagnostic Metric instances."""
        metrics: list[Metric] = [
            Metric(
                name="retrieval_total_count",
                value=self.total_retrieved,
                threshold=float(self.requested_k) if self.requested_k else 1.0,
                unit="count",
                passed=not self.is_empty_retrieval and not self.k_shortfall,
                details={"requested_k": self.requested_k, "k_shortfall": self.k_shortfall},
            ),
            Metric(
                name="retrieval_relevance_ratio",
                value=self.relevance_ratio,
                threshold=0.50,
                unit="ratio",
                passed=not self.is_low_relevance,
                details={
                    "relevant_documents": self.relevant_documents,
                    "relevance_threshold": self.relevance_threshold,
                    "max_score": self.max_score,
                },
            ),
            Metric(
                name="retrieval_duplicate_ratio",
                value=self.duplicate_ratio,
                threshold=0.30,
                unit="ratio",
                passed=not self.is_high_duplicate_ratio,
                details={"duplicate_count": self.duplicate_count},
            ),
            Metric(
                name="retrieval_context_tokens",
                value=self.context_tokens_estimate,
                threshold=50.0,
                unit="tokens",
                passed=not self.is_insufficient_context,
                details={
                    "context_characters": self.context_characters,
                    "empty_documents_count": self.empty_documents_count,
                },
            ),
        ]

        if self.mean_score is not None:
            metrics.append(
                Metric(
                    name="retrieval_mean_score",
                    value=self.mean_score,
                    threshold=self.relevance_threshold,
                    unit="score",
                    passed=self.mean_score >= self.relevance_threshold,
                    details={
                        "max_score": self.max_score,
                        "min_score": self.min_score,
                        "variance": self.score_variance,
                        "spread": self.score_spread,
                    },
                )
            )

        if self.latency_ms is not None:
            metrics.append(
                Metric(
                    name="retrieval_latency_ms",
                    value=self.latency_ms,
                    threshold=1000.0,
                    unit="ms",
                    passed=self.latency_ms <= 1000.0,
                    details={"retriever_name": self.retriever_name},
                )
            )

        return metrics
