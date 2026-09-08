"""Data models for CI/CD evaluation harness and reliability gatekeeper."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_reliability.models.enums import FailureCategory


class GatekeeperConfig(BaseModel):
    """Configuration criteria and thresholds for the CI evaluation gatekeeper."""

    max_failure_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Maximum allowed ratio of failing runs in candidate trace (0.0 to 1.0)",
    )
    max_regression_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable regression score when compared against baseline (0.0 to 1.0)",
    )
    max_latency_p95_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permissible 95th percentile latency in milliseconds",
    )
    min_grounding_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable grounding / faithfulness score (0.0 to 1.0)",
    )
    disallowed_categories: list[FailureCategory] = Field(
        default_factory=list,
        description="List of failure categories strictly forbidden from appearing in candidate runs",
    )
    policy_file: Path | None = Field(
        default=None,
        description="Optional path to custom diagnostic policy file (YAML/JSON)",
    )


class GatekeeperReport(BaseModel):
    """Structured evaluation report containing gate verdict, violations, and PR summary."""

    verdict: str = Field(description="Gate outcome: 'PASSED' or 'FAILED'")
    passed: bool = Field(description="True if all gatekeeper criteria were satisfied")
    violations: list[str] = Field(
        default_factory=list, description="List of unmet threshold violations"
    )
    candidate_summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics of candidate execution"
    )
    comparison_summary: dict[str, Any] | None = Field(
        default=None, description="Comparative summary vs baseline if baseline provided"
    )
    markdown_summary: str = Field(
        default="", description="Formatted GitHub pull request markdown comment"
    )
