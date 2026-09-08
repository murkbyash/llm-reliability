"""Example 02: Diagnosing Agent Trajectory Loops and Tool Failures."""

from llm_reliability import (
    DiagnosticEngine,
    RecommendationEngine,
    SyntheticTraceGenerator,
)


def main() -> None:
    print("=" * 70)
    print("Example 02: Diagnosing AI Agent Trajectory Loops & Tool Failures")
    print("=" * 70)

    # 1. Generate an agent execution trace trapped in an infinite retry loop
    generator = SyntheticTraceGenerator()
    agent_trace = generator.generate_agent_loop_trace()

    print(f"\n[1] Generated Agent Trace: {agent_trace.trace_id}")
    print(f"    Executed Steps: {len(agent_trace.runs[0].spans)}")

    # 2. Run Root Cause Diagnosis
    engine = DiagnosticEngine()
    diagnosis = engine.diagnose(agent_trace)

    print("\n[2] Diagnostic Results:")
    print(f"    Primary Root Cause : {diagnosis.primary_category.value}")
    print(f"    Confidence         : {diagnosis.confidence * 100:.1f}%")
    print(f"    Severity           : {diagnosis.severity.value.upper()}")
    print(f"    Summary            : {diagnosis.summary}")

    # 3. Inspect Failure Evidence
    print("\n[3] Extracted Evidence:")
    for ev in diagnosis.evidence:
        print(f"    * [{ev.evidence_type.value}] {ev.description}")

    # 4. Generate Remediation
    recs = RecommendationEngine().recommend(diagnosis)
    print("\n[4] Agent Guardrail Recommendations:")
    for r in recs:
        print(f"    [P{r.priority}] {r.title} ({r.action_type}):")
        print(f"        {r.description}")
    print("=" * 70)


if __name__ == "__main__":
    main()
