"""Data models for multi-run batch comparative regression analysis."""

from enum import Enum

from pydantic import BaseModel, Field

from llm_reliability.models.enums import FailureCategory


class RegressionVerdict(str, Enum):
    """Overall verdict of a multi-run comparative regression analysis."""

    PASSED = "passed"
    IMPROVED = "improved"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


class DistributionSummary(BaseModel):
    """Statistical summary of a continuous numerical distribution."""

    count: int = Field(default=0, ge=0, description="Total number of observations")
    mean: float = Field(default=0.0, description="Arithmetic mean")
    min_val: float = Field(default=0.0, description="Minimum observed value")
    max_val: float = Field(default=0.0, description="Maximum observed value")
    p50: float = Field(default=0.0, description="Median (50th percentile)")
    p90: float = Field(default=0.0, description="90th percentile")
    p95: float = Field(default=0.0, description="95th percentile")
    p99: float = Field(default=0.0, description="99th percentile")


class FailureRateSummary(BaseModel):
    """Summary of execution failure rates and category breakdowns for a batch."""

    total_runs: int = Field(default=0, ge=0, description="Total number of runs in the batch")
    failed_runs: int = Field(
        default=0, ge=0, description="Number of runs with at least one failure"
    )
    failure_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Proportion of failed runs (0.0 to 1.0)"
    )
    category_counts: dict[str, int] = Field(
        default_factory=dict, description="Count of occurrences per failure category"
    )


class BatchComparisonReport(BaseModel):
    """Comparative report analyzing two batches of execution runs (e.g. Baseline vs Candidate)."""

    baseline_runs_count: int = Field(description="Number of runs in baseline batch")
    candidate_runs_count: int = Field(description="Number of runs in candidate batch")

    baseline_failure_summary: FailureRateSummary = Field(
        description="Failure rate breakdown for baseline"
    )
    candidate_failure_summary: FailureRateSummary = Field(
        description="Failure rate breakdown for candidate"
    )
    failure_rate_delta: float = Field(
        description="Change in failure rate (candidate_rate - baseline_rate)"
    )

    latency_baseline: DistributionSummary = Field(
        description="Latency distribution for baseline (ms)"
    )
    latency_candidate: DistributionSummary = Field(
        description="Latency distribution for candidate (ms)"
    )
    latency_p95_delta_ms: float = Field(
        description="Difference in p95 latency (candidate - baseline) in ms"
    )

    new_failure_categories: list[FailureCategory] = Field(
        default_factory=list, description="Failure categories newly observed in candidate batch"
    )
    resolved_failure_categories: list[FailureCategory] = Field(
        default_factory=list,
        description="Failure categories present in baseline but absent in candidate",
    )

    regression_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calculated regression severity score from 0.0 (improved/clean) to 1.0 (severe regression)",
    )
    verdict: RegressionVerdict = Field(description="Final comparative assessment verdict")
    summary: str = Field(description="Human-readable synthesis of comparative findings")
