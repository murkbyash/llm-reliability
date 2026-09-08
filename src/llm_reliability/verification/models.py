"""Data models for counterfactual simulation, rerun verification, and before/after comparisons."""

from typing import Any

from pydantic import BaseModel, Field

from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import FailureCategory, Severity


class MetricComparison(BaseModel):
    """Before vs after comparison for a single diagnostic metric."""

    name: str = Field(description="Name of the evaluated metric")
    before_value: float | int | bool | str | None = Field(
        default=None, description="Metric value before applying fix"
    )
    after_value: float | int | bool | str | None = Field(
        default=None, description="Metric value after applying fix"
    )
    delta: float | None = Field(
        default=None, description="Absolute numerical difference (after - before) if applicable"
    )
    percent_change: float | None = Field(
        default=None,
        description="Percentage change ((after - before) / before * 100) if applicable",
    )
    improved: bool = Field(
        default=False,
        description="True if the metric change represents an improvement in reliability",
    )
    description: str | None = Field(
        default=None, description="Contextual explanation of the change"
    )


class CounterfactualResult(BaseModel):
    """Result of simulating a counterfactual fix or configuration adjustment."""

    experiment_name: str = Field(description="Identifier or name of the simulated fix")
    experiment_type: str = Field(
        description="Type of intervention: TOP_K, DEDUPLICATION, AGENT_GUARDRAIL, etc."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters applied during the simulation"
    )
    before_diagnosis: Diagnosis = Field(description="Original diagnosis before simulated fix")
    after_diagnosis: Diagnosis = Field(description="Diagnosis after simulated fix")
    metric_comparisons: list[MetricComparison] = Field(
        default_factory=list, description="Detailed before/after metric comparisons"
    )
    is_improved: bool = Field(
        default=False,
        description="True if the simulated fix resolved failures or improved key metrics",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in the counterfactual simulation result",
    )
    resolved_failures: list[FailureCategory] = Field(
        default_factory=list, description="List of failure categories resolved by the fix"
    )
    remaining_failures: list[FailureCategory] = Field(
        default_factory=list, description="List of failure categories still present after the fix"
    )
    summary: str = Field(description="Human-readable summary of counterfactual simulation outcome")


class VerificationReport(BaseModel):
    """Comprehensive report verifying whether an applied fix improved overall system reliability."""

    is_verified: bool = Field(
        description="True if the fix is verified to improve reliability without introducing regressions"
    )
    overall_confidence: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Statistical confidence in verification outcome"
    )
    severity_before: Severity = Field(description="Original severity level")
    severity_after: Severity = Field(description="Post-fix severity level")
    metric_comparisons: list[MetricComparison] = Field(
        default_factory=list, description="Full suite of before vs after metric comparisons"
    )
    resolved_failures: list[FailureCategory] = Field(
        default_factory=list, description="Failures successfully resolved"
    )
    new_regressions: list[FailureCategory] = Field(
        default_factory=list,
        description="New failure categories introduced by the change (regressions)",
    )
    summary: str = Field(description="Detailed verification summary")
