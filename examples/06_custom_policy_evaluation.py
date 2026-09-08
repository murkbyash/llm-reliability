"""Example 06: Custom Diagnostic Policies and Strict SLA Rules."""

from llm_reliability import (
    CustomRule,
    DiagnosticPolicy,
    PolicyEngine,
    Severity,
    SyntheticTraceGenerator,
    ThresholdConfig,
)


def main() -> None:
    print("=" * 70)
    print("Example 06: Custom Diagnostic Policies & Production SLA Gates")
    print("=" * 70)

    # 1. Define custom SLA policy
    policy = DiagnosticPolicy(
        policy_id="enterprise_rag_sla",
        name="Production Enterprise RAG SLA",
        thresholds=ThresholdConfig(
            min_supported_ratio=0.85,
            min_relevance_score=0.70,
            max_duplicate_ratio=0.15,
            max_latency_ms=1200.0,
            disallowed_tools=["eval_python", "raw_sql_exec"],
        ),
        custom_rules=[
            CustomRule(
                rule_id="strict_faithfulness_check",
                name="Strict Grounding Faithfulness Check",
                description="Grounding faithfulness score must exceed 0.85",
                metric_name="answer_grounding_score",
                operator=">=",
                threshold_value=0.85,
                severity=Severity.HIGH,
            )
        ],
    )

    print(f"\n[1] Configured Policy: {policy.name}")
    print(f"    * Min Grounding Score: {policy.thresholds.min_supported_ratio}")
    print(f"    * Disallowed Tools   : {policy.thresholds.disallowed_tools}")

    # 2. Test policy against a clean trace vs a failing trace
    generator = SyntheticTraceGenerator()
    clean_trace = generator.generate_clean_trace()
    hallucination_trace = generator.generate_hallucination_trace()

    engine = PolicyEngine(policy=policy)

    clean_result = engine.evaluate(clean_trace)
    print("\n[2] Clean Trace Evaluation:")
    print(f"    * Passed Policy: {clean_result.passed}")
    print(f"    * Violations   : {len(clean_result.violations)}")

    failing_result = engine.evaluate(hallucination_trace)
    print("\n[3] Hallucination Trace Evaluation:")
    print(f"    * Passed Policy: {failing_result.passed}")
    print(f"    * Violations   : {len(failing_result.violations)}")
    for v in failing_result.violations:
        print(f"      [!] [{v.severity.value}] {v.name}: {v.message}")
    print("=" * 70)


if __name__ == "__main__":
    main()
