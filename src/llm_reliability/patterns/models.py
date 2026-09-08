"""Data models for structural failure signatures, recurrent pattern fingerprinting, and clustering."""

from pydantic import BaseModel, Field

from llm_reliability.models.enums import FailureCategory, Severity


class FailureSignature(BaseModel):
    """Normalized structural signature and fingerprint of a diagnosed failure mode."""

    fingerprint: str = Field(
        description="Deterministic cryptographic/normalized hash of the failure structure"
    )
    category: FailureCategory = Field(description="Primary failure classification category")
    pattern_name: str = Field(description="Human-readable name or label of the failure pattern")
    structural_key: str = Field(
        description="Invariant structural key representing the failure path and error signature"
    )
    severity: Severity = Field(
        default=Severity.MEDIUM, description="Inherent severity of the signature"
    )
    description: str = Field(description="Detailed explanation of the failure signature")


class FailureCluster(BaseModel):
    """Cluster of recurring execution traces sharing an identical or near-identical failure signature."""

    cluster_id: str = Field(description="Unique identifier for the failure cluster")
    signature: FailureSignature = Field(
        description="Common failure signature characterizing the cluster"
    )
    occurrence_count: int = Field(default=1, ge=1, description="Number of runs in this cluster")
    frequency_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of total failed runs exhibiting this signature",
    )
    representative_run_id: str | None = Field(
        default=None, description="Run ID of the most representative run for root cause inspection"
    )
    representative_trace_id: str | None = Field(
        default=None, description="Trace ID associated with the representative run"
    )
    run_ids: list[str] = Field(
        default_factory=list, description="All run IDs grouped in this cluster"
    )
    trace_ids: list[str] = Field(
        default_factory=list, description="All trace IDs grouped in this cluster"
    )
    common_contributing_factors: list[str] = Field(
        default_factory=list, description="Aggregated contributing factors across traces in cluster"
    )
    remediation_summary: str | None = Field(
        default=None, description="Consolidated remediation advice for this cluster"
    )


class PatternAnalysisReport(BaseModel):
    """Aggregate analysis report summarizing recurrent failure patterns and clusters across a trace corpus."""

    total_runs_analyzed: int = Field(
        default=0, ge=0, description="Total number of execution runs analyzed"
    )
    failed_runs_count: int = Field(
        default=0, ge=0, description="Total number of runs exhibiting failures"
    )
    unique_patterns_count: int = Field(
        default=0, ge=0, description="Number of distinct failure patterns discovered"
    )
    clusters: list[FailureCluster] = Field(
        default_factory=list,
        description="Ranked list of failure clusters sorted by occurrence frequency",
    )
    summary: str = Field(description="Executive summary of failure pattern analysis")
