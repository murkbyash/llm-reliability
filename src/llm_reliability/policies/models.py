"""Data models for custom diagnostic policies, configurable thresholds, and rule evaluation."""

from typing import Any

from pydantic import BaseModel, Field

from llm_reliability.models.enums import FailureCategory, Severity


class ThresholdConfig(BaseModel):
    """Configurable numerical and structural thresholds for reliability diagnostics."""

    min_relevance_score: float | None = Field(
        default=0.50, ge=0.0, le=1.0, description="Minimum acceptable retrieval relevance score"
    )
    min_supported_ratio: float | None = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum acceptable grounding claim support ratio"
    )
    max_context_tokens: int | None = Field(
        default=4000, ge=0, description="Maximum allowable context tokens before flagging bloat"
    )
    max_duplicate_ratio: float | None = Field(
        default=0.30, ge=0.0, le=1.0, description="Maximum allowable duplicate chunk ratio"
    )
    max_latency_ms: float | None = Field(
        default=None, ge=0.0, description="Maximum acceptable total run execution duration (ms)"
    )
    max_agent_steps: int | None = Field(
        default=10, ge=1, description="Maximum allowable agent tool execution steps"
    )
    disallowed_tools: list[str] = Field(
        default_factory=list, description="List of forbidden tool names (e.g. execute_raw_sql)"
    )
    required_metadata_keys: list[str] = Field(
        default_factory=list, description="Mandatory metadata keys that must exist on traces/runs"
    )


class CustomRule(BaseModel):
    """User-defined diagnostic rule or policy invariant."""

    rule_id: str = Field(description="Unique identifier for the custom rule")
    name: str = Field(description="Human-readable rule name")
    category: str | FailureCategory = Field(
        default="CUSTOM_POLICY_VIOLATION",
        description="Failure category associated with violations of this rule",
    )
    severity: Severity = Field(
        default=Severity.MEDIUM, description="Severity assigned when rule is violated"
    )
    metric_name: str | None = Field(
        default=None, description="Diagnostic metric to evaluate (e.g. retrieval_relevance_ratio)"
    )
    operator: str = Field(
        default="<",
        description="Comparison operator: '<', '<=', '>', '>=', '==', '!=', 'in', 'not_in'",
    )
    threshold_value: Any = Field(
        default=None, description="Target value or threshold to evaluate metric against"
    )
    description: str = Field(default="", description="Explanation of the rule and rationale")
    remediation: str | None = Field(
        default=None, description="Recommended remediation action when violated"
    )


class DiagnosticPolicy(BaseModel):
    """Complete diagnostic policy container defining thresholds, rules, and enforcement modes."""

    policy_id: str = Field(default="default-policy", description="Unique policy identifier")
    name: str = Field(default="Default Diagnostic Policy", description="Policy name")
    version: str = Field(default="1.0", description="Policy version string")
    thresholds: ThresholdConfig = Field(
        default_factory=ThresholdConfig, description="Configured diagnostic thresholds"
    )
    custom_rules: list[CustomRule] = Field(
        default_factory=list, description="List of user-defined custom rules"
    )
    enforce_strict_grounding: bool = Field(
        default=False, description="Whether to treat any ungrounded claim as a critical violation"
    )
    fail_on_warning: bool = Field(
        default=False,
        description="Whether to mark policy as failed on low/info severity violations",
    )


class PolicyViolation(BaseModel):
    """Instance of a detected policy rule or threshold breach."""

    rule_id: str = Field(description="ID of the violated rule or threshold")
    name: str = Field(description="Name of the violated policy check")
    severity: Severity = Field(description="Severity of the violation")
    metric_name: str | None = Field(default=None, description="Metric evaluated, if applicable")
    observed_value: Any = Field(default=None, description="Observed runtime value")
    expected_threshold: Any = Field(default=None, description="Configured policy limit")
    message: str = Field(description="Detailed explanation of the policy violation")
    remediation: str | None = Field(
        default=None, description="Actionable recommendation to resolve the violation"
    )


class PolicyEvaluationResult(BaseModel):
    """Overall outcome of evaluating an execution run or trace against a DiagnosticPolicy."""

    policy_id: str = Field(description="Identifier of the evaluated policy")
    policy_name: str = Field(description="Name of the evaluated policy")
    passed: bool = Field(description="True if zero unacceptable policy violations were detected")
    violations_count: int = Field(default=0, ge=0, description="Total number of violations")
    violations: list[PolicyViolation] = Field(
        default_factory=list, description="List of detailed violation records"
    )
    summary: str = Field(description="Human-readable policy evaluation summary")
