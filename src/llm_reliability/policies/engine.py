"""Policy evaluation engine enforcing configurable diagnostic rules and custom thresholds."""

import json
from pathlib import Path
from typing import Any

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import Severity, SpanKind
from llm_reliability.models.trace import Run, Trace
from llm_reliability.normalization.loader import load_trace
from llm_reliability.policies.models import (
    DiagnosticPolicy,
    PolicyEvaluationResult,
    PolicyViolation,
)


class PolicyEngine:
    """Evaluates execution runs against user-defined diagnostic policies and custom thresholds."""

    def __init__(
        self,
        policy: DiagnosticPolicy | None = None,
        diagnostic_engine: DiagnosticEngine | None = None,
    ) -> None:
        """Initialize policy engine with policy configuration and diagnostic analyzer."""
        self.policy = policy or DiagnosticPolicy()
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()

    def evaluate(
        self,
        source: Run | Trace | Diagnosis | dict[str, Any] | str | Path | Any,
    ) -> PolicyEvaluationResult:
        """Evaluate a run, trace, or pre-computed diagnosis against the active policy.

        Args:
            source: Run instance, Trace, Diagnosis, dict, or file path.

        Returns:
            PolicyEvaluationResult with detailed violations and pass/fail verdict.
        """
        run: Run | None = None
        diagnosis: Diagnosis

        if isinstance(source, Diagnosis):
            diagnosis = source
        elif isinstance(source, Run):
            run = source
            diagnosis = self.diagnostic_engine.diagnose_run(run)
        elif isinstance(source, Trace):
            diagnosis = self.diagnostic_engine.diagnose_trace(source)
            if source.runs:
                run = source.runs[0]
        else:
            trace = load_trace(source)
            diagnosis = self.diagnostic_engine.diagnose_trace(trace)
            if trace.runs:
                run = trace.runs[0]

        violations: list[PolicyViolation] = []

        # 1. Evaluate Structural Thresholds on Run
        if run is not None:
            self._evaluate_run_thresholds(run, violations)

        # 2. Evaluate Diagnostic Metrics Against Policy Thresholds
        self._evaluate_metric_thresholds(diagnosis, violations)

        # 3. Evaluate User Custom Rules
        self._evaluate_custom_rules(diagnosis, violations)

        # 4. Strict Grounding Check
        if self.policy.enforce_strict_grounding:
            self._evaluate_strict_grounding(diagnosis, violations)

        # Determine pass/fail verdict
        if not violations:
            passed = True
        elif self.policy.fail_on_warning:
            passed = False
        else:
            # Pass only if all violations are INFO or LOW
            passed = all(v.severity in (Severity.INFO, Severity.LOW) for v in violations)

        summary = (
            f"Policy '{self.policy.name}' evaluation: {'PASSED' if passed else 'FAILED'}. "
            f"Detected {len(violations)} violation(s)."
        )

        return PolicyEvaluationResult(
            policy_id=self.policy.policy_id,
            policy_name=self.policy.name,
            passed=passed,
            violations_count=len(violations),
            violations=violations,
            summary=summary,
        )

    def _evaluate_run_thresholds(self, run: Run, violations: list[PolicyViolation]) -> None:
        """Evaluate run-level properties such as latency, step count, and disallowed tools."""
        thresh = self.policy.thresholds

        # Max Latency
        if thresh.max_latency_ms is not None:
            latency = float(run.duration_ms or sum(s.duration_ms or 0.0 for s in run.spans))
            if latency > thresh.max_latency_ms:
                violations.append(
                    PolicyViolation(
                        rule_id="thresh-max-latency",
                        name="Execution Latency Threshold Exceeded",
                        severity=Severity.HIGH,
                        metric_name="run_duration_ms",
                        observed_value=latency,
                        expected_threshold=thresh.max_latency_ms,
                        message=f"Total run latency ({latency}ms) exceeded maximum allowable threshold ({thresh.max_latency_ms}ms).",
                        remediation="Optimize retriever query latency, parallelize tool calls, or switch to faster LLM inference endpoint.",
                    )
                )

        # Max Agent Steps
        if thresh.max_agent_steps is not None:
            tool_steps = [s for s in run.spans if s.kind == SpanKind.TOOL and s.tool_call]
            if len(tool_steps) > thresh.max_agent_steps:
                violations.append(
                    PolicyViolation(
                        rule_id="thresh-max-agent-steps",
                        name="Agent Step Limit Exceeded",
                        severity=Severity.MEDIUM,
                        metric_name="agent_tool_step_count",
                        observed_value=len(tool_steps),
                        expected_threshold=thresh.max_agent_steps,
                        message=f"Agent performed {len(tool_steps)} tool steps, exceeding the maximum allowed limit of {thresh.max_agent_steps}.",
                        remediation="Constrain agent planning prompt or configure hard step limit guardrails.",
                    )
                )

        # Disallowed Tools
        if thresh.disallowed_tools:
            for s in run.spans:
                if s.kind == SpanKind.TOOL and s.tool_call:
                    t_name = s.tool_call.tool_name
                    if t_name in thresh.disallowed_tools:
                        violations.append(
                            PolicyViolation(
                                rule_id="thresh-disallowed-tool",
                                name="Disallowed Tool Execution",
                                severity=Severity.CRITICAL,
                                metric_name="disallowed_tool_used",
                                observed_value=t_name,
                                expected_threshold=f"Forbidden tools: {thresh.disallowed_tools}",
                                message=f"Disallowed tool '{t_name}' was invoked during execution.",
                                remediation=f"Remove '{t_name}' from agent accessible tool registry or enforce schema guardrails.",
                            )
                        )

        # Required Metadata Keys
        if thresh.required_metadata_keys:
            metadata = run.metadata or {}
            for key in thresh.required_metadata_keys:
                if key not in metadata:
                    violations.append(
                        PolicyViolation(
                            rule_id="thresh-required-metadata",
                            name="Missing Required Metadata Key",
                            severity=Severity.LOW,
                            metric_name="metadata_keys",
                            observed_value=list(metadata.keys()),
                            expected_threshold=f"Required: {key}",
                            message=f"Run metadata is missing required key '{key}'.",
                            remediation=f"Ensure application telemetry attaches '{key}' to trace context.",
                        )
                    )

    def _evaluate_metric_thresholds(
        self, diagnosis: Diagnosis, violations: list[PolicyViolation]
    ) -> None:
        """Evaluate diagnostic metrics against configured thresholds."""
        thresh = self.policy.thresholds
        metric_map = {m.name: m for m in diagnosis.metrics}

        # Min Relevance Score
        if thresh.min_relevance_score is not None:
            m = metric_map.get("retrieval_relevance_ratio")
            if m and isinstance(m.value, (int, float)) and m.value < thresh.min_relevance_score:
                violations.append(
                    PolicyViolation(
                        rule_id="thresh-min-relevance",
                        name="Retrieval Relevance Below Policy Threshold",
                        severity=Severity.MEDIUM,
                        metric_name="retrieval_relevance_ratio",
                        observed_value=m.value,
                        expected_threshold=thresh.min_relevance_score,
                        message=f"Retrieval relevance ratio ({m.value}) fell below policy minimum threshold ({thresh.min_relevance_score}).",
                        remediation="Tune retriever similarity score cutoff or incorporate reranking.",
                    )
                )

        # Min Supported Ratio (Grounding)
        if thresh.min_supported_ratio is not None:
            m = metric_map.get("grounding_supported_ratio")
            if m and isinstance(m.value, (int, float)) and m.value < thresh.min_supported_ratio:
                violations.append(
                    PolicyViolation(
                        rule_id="thresh-min-grounding",
                        name="Answer Grounding Support Ratio Below Threshold",
                        severity=Severity.HIGH,
                        metric_name="grounding_supported_ratio",
                        observed_value=m.value,
                        expected_threshold=thresh.min_supported_ratio,
                        message=f"Grounding support ratio ({m.value}) fell below policy threshold ({thresh.min_supported_ratio}).",
                        remediation="Strengthen system prompt grounding constraints or penalize unsupported claims.",
                    )
                )

        # Max Context Tokens
        if thresh.max_context_tokens is not None:
            m = metric_map.get("retrieval_context_tokens")
            if m and isinstance(m.value, (int, float)) and m.value > thresh.max_context_tokens:
                violations.append(
                    PolicyViolation(
                        rule_id="thresh-max-context-tokens",
                        name="Retrieval Context Token Bloat",
                        severity=Severity.LOW,
                        metric_name="retrieval_context_tokens",
                        observed_value=m.value,
                        expected_threshold=thresh.max_context_tokens,
                        message=f"Retrieved context size ({m.value} tokens) exceeded policy limit ({thresh.max_context_tokens} tokens).",
                        remediation="Reduce top-k or decrease chunk size.",
                    )
                )

        # Max Duplicate Ratio
        if thresh.max_duplicate_ratio is not None:
            m = metric_map.get("retrieval_duplicate_ratio")
            if m and isinstance(m.value, (int, float)) and m.value > thresh.max_duplicate_ratio:
                violations.append(
                    PolicyViolation(
                        rule_id="thresh-max-duplicate-ratio",
                        name="Excessive Context Duplicate Ratio",
                        severity=Severity.LOW,
                        metric_name="retrieval_duplicate_ratio",
                        observed_value=m.value,
                        expected_threshold=thresh.max_duplicate_ratio,
                        message=f"Context duplicate ratio ({m.value}) exceeded limit ({thresh.max_duplicate_ratio}).",
                        remediation="Enable pre-context deduplication filter before prompt injection.",
                    )
                )

    def _evaluate_custom_rules(
        self, diagnosis: Diagnosis, violations: list[PolicyViolation]
    ) -> None:
        """Evaluate custom user-defined rules against diagnosis metrics."""
        metric_map = {m.name: m.value for m in diagnosis.metrics}

        for rule in self.policy.custom_rules:
            if not rule.metric_name:
                continue

            obs_val = metric_map.get(rule.metric_name)
            if obs_val is None:
                continue

            is_violation = self._check_rule_violation(obs_val, rule.operator, rule.threshold_value)
            if is_violation:
                violations.append(
                    PolicyViolation(
                        rule_id=rule.rule_id,
                        name=rule.name,
                        severity=rule.severity,
                        metric_name=rule.metric_name,
                        observed_value=obs_val,
                        expected_threshold=f"{rule.operator} {rule.threshold_value}",
                        message=f"Custom rule '{rule.name}' violated: observed {rule.metric_name}={obs_val}, expected {rule.operator} {rule.threshold_value}.",
                        remediation=rule.remediation or rule.description,
                    )
                )

    def _check_rule_violation(self, val: Any, operator: str, target: Any) -> bool:
        """Evaluate boolean condition for custom rule violation."""
        try:
            if operator in ("<", "lt"):
                return bool(float(val) < float(target))
            elif operator in ("<=", "lte"):
                return bool(float(val) <= float(target))
            elif operator in (">", "gt"):
                return bool(float(val) > float(target))
            elif operator in (">=", "gte"):
                return bool(float(val) >= float(target))
            elif operator in ("==", "eq"):
                return bool(val == target)
            elif operator in ("!=", "neq"):
                return bool(val != target)
            elif operator == "in":
                return bool(val in target)
            elif operator == "not_in":
                return bool(val not in target)
            return False
        except (ValueError, TypeError):
            return False

    def _evaluate_strict_grounding(
        self, diagnosis: Diagnosis, violations: list[PolicyViolation]
    ) -> None:
        """Enforce zero-tolerance ungrounded claim validation."""
        unsupported = 0.0
        contradictions = 0.0
        for m in diagnosis.metrics:
            if m.name in ("answer_unsupported_claims_count", "grounding_unsupported_claims"):
                if isinstance(m.value, (int, float)):
                    unsupported += float(m.value)
            elif m.name in ("answer_contradictions_count", "grounding_contradicted_claims"):
                if isinstance(m.value, (int, float)):
                    contradictions += float(m.value)
            elif m.name in ("answer_grounding_score", "grounding_supported_ratio"):
                if isinstance(m.value, (int, float)) and m.value < 0.99:
                    unsupported += 1.0

        if unsupported > 0 or contradictions > 0:
            violations.append(
                PolicyViolation(
                    rule_id="strict-grounding-unsupported",
                    name="Strict Grounding Violation: Unsupported Claims Detected",
                    severity=Severity.CRITICAL,
                    metric_name="answer_unsupported_claims_count",
                    observed_value=unsupported,
                    expected_threshold=0,
                    message=f"Strict grounding policy rejected response: {int(unsupported)} unsupported claim(s) and {int(contradictions)} contradiction(s) detected.",
                    remediation="Require 100% lexical or entity grounding for all output sentences.",
                )
            )

    @classmethod
    def load_policy_from_dict(cls, data: dict[str, Any]) -> DiagnosticPolicy:
        """Instantiate DiagnosticPolicy from Python dictionary."""
        return DiagnosticPolicy.model_validate(data)

    @classmethod
    def load_policy_from_json(cls, content_or_path: str | Path) -> DiagnosticPolicy:
        """Load DiagnosticPolicy from JSON string or file path."""
        path = Path(content_or_path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = str(content_or_path)
        return cls.load_policy_from_dict(json.loads(text))

    @classmethod
    def load_policy_from_yaml(cls, content_or_path: str | Path) -> DiagnosticPolicy:
        """Load DiagnosticPolicy from simple YAML/JSON text or file path (pure-Python stdlib parser)."""
        path = Path(content_or_path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = str(content_or_path)

        # Try JSON first
        try:
            return cls.load_policy_from_dict(json.loads(text))
        except json.JSONDecodeError:
            # Fallback to simple YAML dictionary parsing
            parsed_dict = cls._parse_simple_yaml(text)
            return cls.load_policy_from_dict(parsed_dict)

    @staticmethod
    def _parse_simple_yaml(yaml_text: str) -> dict[str, Any]:
        """Simple pure-Python YAML subset parser for policy configurations."""
        result: dict[str, Any] = {}
        current_section: str | None = None

        for raw_line in yaml_text.splitlines():
            line = raw_line.rstrip()
            if not line or line.strip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            if indent == 0 and ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()
                if not val:
                    current_section = key
                    result[current_section] = {}
                else:
                    current_section = None
                    result[key] = _parse_yaml_value(val)
            elif indent > 0 and current_section and ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()
                result[current_section][key] = _parse_yaml_value(val)

        return result


def _parse_yaml_value(val_str: str) -> Any:
    """Helper to cast scalar YAML values."""
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False
    if val_str.lower() in ("null", "none", "~"):
        return None
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str.strip('"').strip("'")
