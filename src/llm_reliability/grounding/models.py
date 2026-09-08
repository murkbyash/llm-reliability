"""Models for sentence-level claim support and aggregate grounding metrics."""

from pydantic import BaseModel, ConfigDict, Field

from llm_reliability.grounding.enums import SupportStatus
from llm_reliability.models.diagnosis import Evidence, Metric
from llm_reliability.models.enums import EvidenceType


class ClaimSupport(BaseModel):
    """Factual support evaluation for an individual claim or sentence in an LLM answer."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    claim_text: str = Field(
        ..., min_length=1, description="Text of the individual claim or sentence"
    )
    status: SupportStatus = Field(..., description="Support status for this specific claim")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in support determination"
    )
    overlap_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Lexical/semantic overlap with context"
    )
    matching_context_snippet: str | None = Field(
        default=None,
        description="Best matching snippet from context supporting or contradicting claim",
    )
    contradiction_reason: str | None = Field(
        default=None,
        description="Explanation if this claim contradicts context",
    )
    missing_entities: list[str] = Field(
        default_factory=list,
        description="Key entities (numbers, names, dates) in claim not found in context",
    )


class GroundingMetrics(BaseModel):
    """Aggregate grounding and faithfulness metrics for an LLM generated response."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    status: SupportStatus = Field(..., description="Overall answer support status")
    grounding_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of answer supported by context (0.0 to 1.0)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in overall grounding assessment",
    )
    total_claims: int = Field(default=0, ge=0, description="Total number of evaluated claims")
    supported_claims: int = Field(default=0, ge=0, description="Count of supported claims")
    unsupported_claims: int = Field(default=0, ge=0, description="Count of unsupported claims")
    contradicted_claims: int = Field(default=0, ge=0, description="Count of contradicted claims")
    claims: list[ClaimSupport] = Field(
        default_factory=list,
        description="Detailed evaluations for each claim",
    )
    hallucinated_entities: list[str] = Field(
        default_factory=list,
        description="Key factual entities in answer not present in context",
    )
    evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Key snippets extracted from context",
    )

    def to_diagnostic_metrics(self) -> list[Metric]:
        """Convert grounding measurements into standard diagnostic Metric instances."""
        return [
            Metric(
                name="answer_grounding_score",
                value=self.grounding_score,
                threshold=0.70,
                unit="ratio",
                passed=self.grounding_score >= 0.70,
                details={
                    "status": self.status.value,
                    "supported_claims": self.supported_claims,
                    "total_claims": self.total_claims,
                },
            ),
            Metric(
                name="answer_unsupported_claims_count",
                value=float(self.unsupported_claims),
                threshold=0.0,
                unit="count",
                passed=self.unsupported_claims == 0,
                details={"unsupported_claims": self.unsupported_claims},
            ),
            Metric(
                name="answer_contradictions_count",
                value=float(self.contradicted_claims),
                threshold=0.0,
                unit="count",
                passed=self.contradicted_claims == 0,
                details={"contradicted_claims": self.contradicted_claims},
            ),
            Metric(
                name="answer_hallucinated_entities_count",
                value=float(len(self.hallucinated_entities)),
                threshold=0.0,
                unit="count",
                passed=len(self.hallucinated_entities) == 0,
                details={"hallucinated_entities": self.hallucinated_entities},
            ),
        ]

    def to_evidence(self, span_id: str | None = None) -> list[Evidence]:
        """Generate structured Evidence objects for unsupported, contradicted, or hallucinated claims."""
        evidence_list: list[Evidence] = []

        if self.status == SupportStatus.CONTRADICTED or self.contradicted_claims > 0:
            for claim in self.claims:
                if claim.status == SupportStatus.CONTRADICTED:
                    evidence_list.append(
                        Evidence(
                            evidence_type=EvidenceType.CONTRADICTION,
                            description=f"Contradiction detected in answer: '{claim.claim_text}'",
                            supporting_data={
                                "claim": claim.claim_text,
                                "contradiction_reason": claim.contradiction_reason,
                                "matching_context": claim.matching_context_snippet,
                            },
                            span_id=span_id,
                        )
                    )

        if self.unsupported_claims > 0 or self.status == SupportStatus.UNSUPPORTED:
            for claim in self.claims:
                if claim.status == SupportStatus.UNSUPPORTED:
                    evidence_list.append(
                        Evidence(
                            evidence_type=EvidenceType.GROUNDING_DEFICIT,
                            description=f"Unsupported claim in answer: '{claim.claim_text}'",
                            supporting_data={
                                "claim": claim.claim_text,
                                "overlap_score": claim.overlap_score,
                                "missing_entities": claim.missing_entities,
                            },
                            span_id=span_id,
                        )
                    )

        if self.hallucinated_entities:
            evidence_list.append(
                Evidence(
                    evidence_type=EvidenceType.HALLUCINATION_OVERLAP,
                    description=f"Extracted {len(self.hallucinated_entities)} hallucinated entities not found in context",
                    supporting_data={"entities": self.hallucinated_entities},
                    span_id=span_id,
                )
            )

        return evidence_list
