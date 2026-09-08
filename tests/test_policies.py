"""Tests for Custom Diagnostic Rules & Policy Engine."""

import json
from pathlib import Path

from llm_reliability import (
    CustomRule,
    DiagnosticPolicy,
    PolicyEngine,
    PolicyEvaluationResult,
    Severity,
    ThresholdConfig,
)
from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    RetrievalStep,
    RetrievedDocument,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span


class TestPolicyEngine:
    """Deterministic tests for user-defined policies, threshold rules, and evaluation engine."""

    def _create_clean_run(self) -> Run:
        doc = RetrievedDocument(
            doc_id="d1",
            content="PostgreSQL is a powerful, open-source object-relational database system.",
            score=0.95,
        )
        step = RetrievalStep(query="postgresql", documents=[doc])
        span = Span(
            span_id="s1",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            duration_ms=150.0,
            retrieval=step,
        )
        return Run(
            run_id="r1",
            trace_id="t1",
            duration_ms=150.0,
            metadata={"tenant_id": "tenant-123"},
            spans=[span],
            final_response=FinalResponse(
                text="PostgreSQL is an open-source object-relational database system."
            ),
        )

    def test_policy_threshold_max_latency(self) -> None:
        policy = DiagnosticPolicy(
            policy_id="p-latency",
            name="Latency SLA Policy",
            thresholds=ThresholdConfig(max_latency_ms=100.0),
        )
        engine = PolicyEngine(policy)
        run = self._create_clean_run()

        res: PolicyEvaluationResult = engine.evaluate(run)

        assert res.passed is False
        assert res.violations_count >= 1
        assert any(v.rule_id == "thresh-max-latency" for v in res.violations)

    def test_policy_threshold_disallowed_tools(self) -> None:
        policy = DiagnosticPolicy(
            policy_id="p-security",
            name="Tool Security Policy",
            thresholds=ThresholdConfig(disallowed_tools=["execute_raw_sql", "shell_exec"]),
        )
        engine = PolicyEngine(policy)

        forbidden_span = Span(
            span_id="s1",
            name="execute_raw_sql",
            kind=SpanKind.TOOL,
            status=SpanStatus.SUCCESS,
            tool_call=ToolCall(
                tool_name="execute_raw_sql", arguments={"query": "DROP TABLE users;"}
            ),
            tool_result=ToolResult(tool_name="execute_raw_sql", output={"rows": 0}),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[forbidden_span])

        res: PolicyEvaluationResult = engine.evaluate(run)

        assert res.passed is False
        assert any(v.severity == Severity.CRITICAL for v in res.violations)
        assert any(v.rule_id == "thresh-disallowed-tool" for v in res.violations)

    def test_policy_custom_rules_evaluation(self) -> None:
        custom_rule = CustomRule(
            rule_id="custom-duplicate-check",
            name="Strict Context Deduplication Invariant",
            metric_name="retrieval_duplicate_ratio",
            operator=">",
            threshold_value=0.10,
            severity=Severity.HIGH,
            description="Context duplicate ratio must not exceed 10%",
        )
        policy = DiagnosticPolicy(
            policy_id="p-custom",
            name="Custom Rule Policy",
            custom_rules=[custom_rule],
        )
        engine = PolicyEngine(policy)

        doc1 = RetrievedDocument(
            doc_id="d1", content="same exact content repeated for index", score=0.9
        )
        doc2 = RetrievedDocument(
            doc_id="d2", content="same exact content repeated for index", score=0.9
        )
        span = Span(
            span_id="s1",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            retrieval=RetrievalStep(query="test", documents=[doc1, doc2]),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])

        res: PolicyEvaluationResult = engine.evaluate(run)

        assert res.passed is False
        assert any(v.rule_id == "custom-duplicate-check" for v in res.violations)

    def test_policy_strict_grounding_rejection(self) -> None:
        policy = DiagnosticPolicy(
            policy_id="p-grounding",
            name="Zero Hallucination Policy",
            enforce_strict_grounding=True,
        )
        engine = PolicyEngine(policy)

        doc = RetrievedDocument(
            doc_id="d1",
            content="The Moon is Earth's only natural satellite, orbiting at 384,400 km.",
            score=0.9,
        )
        span = Span(
            span_id="s1",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            retrieval=RetrievalStep(query="moon", documents=[doc]),
        )
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(
                text="The tech index rallied today and quantum computing shares surged 50%."
            ),
        )

        res: PolicyEvaluationResult = engine.evaluate(run)

        assert res.passed is False
        assert any(v.rule_id == "strict-grounding-unsupported" for v in res.violations)

    def test_policy_load_from_json_and_yaml(self, tmp_path: Path) -> None:
        policy_data = {
            "policy_id": "test-json-policy",
            "name": "JSON Configured Policy",
            "thresholds": {
                "min_relevance_score": 0.85,
                "max_latency_ms": 1000.0,
            },
        }

        json_file = tmp_path / "policy.json"
        json_file.write_text(json.dumps(policy_data), encoding="utf-8")

        loaded_json_policy = PolicyEngine.load_policy_from_json(json_file)
        assert loaded_json_policy.policy_id == "test-json-policy"
        assert loaded_json_policy.thresholds.min_relevance_score == 0.85

        yaml_text = """
policy_id: test-yaml-policy
name: YAML Configured Policy
thresholds:
  min_relevance_score: 0.80
  max_latency_ms: 2500.0
"""
        yaml_file = tmp_path / "policy.yaml"
        yaml_file.write_text(yaml_text, encoding="utf-8")

        loaded_yaml_policy = PolicyEngine.load_policy_from_yaml(yaml_file)
        assert loaded_yaml_policy.policy_id == "test-yaml-policy"
        assert loaded_yaml_policy.thresholds.min_relevance_score == 0.80

    def test_policy_clean_pass(self) -> None:
        policy = DiagnosticPolicy(
            policy_id="p-default",
            name="Standard Policy",
            thresholds=ThresholdConfig(
                max_latency_ms=1000.0,
                min_relevance_score=0.50,
                min_supported_ratio=0.50,
            ),
        )
        engine = PolicyEngine(policy)
        run = self._create_clean_run()

        res: PolicyEvaluationResult = engine.evaluate(run)

        assert res.passed is True
        assert res.violations_count == 0
        assert len(res.violations) == 0
