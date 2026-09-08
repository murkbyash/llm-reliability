"""Diagnostic output models including metrics, evidence, failures, recommendations, and root-cause diagnoses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_reliability.models.enums import EvidenceType, FailureCategory, Severity


class Metric(BaseModel):
    """A computed diagnostic or performance metric."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = Field(
        description="Name/identifier of the metric (e.g., 'relevance_ratio', 'grounding_score')"
    )
    value: float | int | bool | str = Field(description="Computed value of the metric")
    threshold: float | None = Field(
        default=None, description="Reference threshold used for evaluation"
    )
    unit: str | None = Field(default=None, description="Measurement unit (e.g. '%', 'ms', 'count')")
    passed: bool | None = Field(default=None, description="Whether metric met acceptable criteria")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Underlying calculation details"
    )


class Evidence(BaseModel):
    """Specific piece of evidence supporting a failure hypothesis."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    evidence_type: EvidenceType = Field(description="Classification of evidence type")
    description: str = Field(description="Human-readable explanation of the evidence")
    metric: Metric | None = Field(default=None, description="Associated metric if applicable")
    supporting_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured supporting data points or extracted snippets",
    )
    span_id: str | None = Field(default=None, description="Span ID where evidence was detected")


class Failure(BaseModel):
    """A detected point of failure or degradation in the execution trace."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    failure_id: str | None = Field(default=None, description="Unique failure identifier")
    category: FailureCategory = Field(description="Classification of failure")
    severity: Severity = Field(default=Severity.MEDIUM, description="Impact severity")
    title: str | None = Field(default=None, description="Short failure title")
    description: str | None = Field(default=None, description="Detailed failure description")
    message: str = Field(default="", description="Summary message describing the failure")
    span_id: str | None = Field(default=None, description="Specific span where failure occurred")
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence items supporting this failure"
    )

    @model_validator(mode="after")
    def populate_message(self) -> "Failure":
        """Synchronize message, title, and description."""
        if not self.message and self.description:
            self.message = self.description
        elif not self.message and self.title:
            self.message = self.title
        if not self.title and self.message:
            self.title = self.message
        if not self.description and self.message:
            self.description = self.message
        return self


class Recommendation(BaseModel):
    """An actionable remediation step for the developer."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    title: str = Field(description="Concise action title")
    description: str = Field(description="Detailed instructions on what to investigate or change")
    action_type: str = Field(
        default="GENERAL",
        description="Category of action (e.g., 'RETRIEVER_CONFIG', 'PROMPT_TUNING', 'RERANKING')",
    )
    priority: int = Field(default=1, ge=1, description="Priority ordering (1 = Highest)")
    rationale: str | None = Field(
        default=None, description="Why this recommendation is suggested based on evidence"
    )


class Hypothesis(BaseModel):
    """A candidate root-cause hypothesis with assigned confidence."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    category: FailureCategory = Field(description="Hypothesized root-cause category")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence probability (0.0 to 1.0)"
    )
    title: str | None = Field(default=None, description="Short hypothesis title")
    description: str | None = Field(default=None, description="Detailed hypothesis explanation")
    rationale: str | None = Field(
        default=None, description="Reasoning behind this candidate hypothesis"
    )
    rank: int | None = Field(
        default=None, ge=1, description="Rank ordering among alternative hypotheses"
    )
    evidence: list[Evidence] = Field(default_factory=list, description="Supporting evidence items")
    contributing_factors: list[str] = Field(
        default_factory=list, description="Contributing environmental or algorithmic factors"
    )

    @model_validator(mode="after")
    def sync_rationale(self) -> "Hypothesis":
        """Synchronize description and rationale."""
        if not self.rationale and self.description:
            self.rationale = self.description
        elif not self.description and self.rationale:
            self.description = self.rationale
        return self


class Diagnosis(BaseModel):
    """Top-level diagnostic result synthesizing failures, evidence, root cause, and recommendations."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    root_cause: FailureCategory = Field(
        default=FailureCategory.NONE, description="Primary diagnosed root cause"
    )
    primary_category: FailureCategory = Field(
        default=FailureCategory.NONE, description="Primary failure category"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in primary diagnosis (0.0 to 1.0)",
    )
    severity: Severity = Field(
        default=Severity.INFO, description="Overall severity assessment of diagnosis"
    )
    summary: str = Field(description="Concise narrative summary of the diagnostic finding")
    uncertainty_note: str | None = Field(
        default=None,
        description="Explicit communication of uncertainty or limitations in evidence",
    )
    failures: list[Failure] = Field(
        default_factory=list, description="All identified failure points"
    )
    evidence: list[Evidence] = Field(default_factory=list, description="Compiled evidence items")
    hypotheses: list[Hypothesis] = Field(
        default_factory=list, description="Ranked root-cause hypotheses"
    )
    alternative_hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Ranked alternative root-cause hypotheses",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Prioritized remediation recommendations",
    )
    metrics: list[Metric] = Field(
        default_factory=list, description="Computed metrics extracted during analysis"
    )
    trace_id: str | None = Field(default=None, description="ID of analyzed trace")
    run_id: str | None = Field(default=None, description="ID of analyzed run")

    @model_validator(mode="after")
    def sync_categories(self) -> "Diagnosis":
        """Ensure root_cause and primary_category are synchronized."""
        if (
            self.primary_category != FailureCategory.NONE
            and self.root_cause == FailureCategory.NONE
        ):
            self.root_cause = self.primary_category
        elif (
            self.root_cause != FailureCategory.NONE
            and self.primary_category == FailureCategory.NONE
        ):
            self.primary_category = self.root_cause
        if not self.alternative_hypotheses and len(self.hypotheses) > 1:
            self.alternative_hypotheses = self.hypotheses[1:]
        elif not self.hypotheses and self.alternative_hypotheses:
            self.hypotheses = self.alternative_hypotheses
        return self
