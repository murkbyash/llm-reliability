"""Example 01: Diagnosing RAG Failures and Generating Remediation Guidance."""

from llm_reliability import (
    DiagnosticEngine,
    RecommendationEngine,
    SyntheticTraceGenerator,
    save_html_report,
)


def main() -> None:
    print("=" * 70)
    print("Example 01: Diagnosing RAG Failures (Local-First, $0 Cloud Cost)")
    print("=" * 70)

    # 1. Generate synthetic RAG trace exhibiting hallucination and retrieval deficit
    generator = SyntheticTraceGenerator()
    rag_trace = generator.generate_hallucination_trace()

    print(f"\n[1] Generated RAG Trace: {rag_trace.trace_id}")
    print(f"    Total Runs: {len(rag_trace.runs)}, Spans: {len(rag_trace.runs[0].spans)}")

    # 2. Run Root Cause Diagnosis
    engine = DiagnosticEngine()
    diagnosis = engine.diagnose(rag_trace)

    print("\n[2] Diagnostic Results:")
    print(f"    Primary Root Cause : {diagnosis.primary_category.value}")
    print(f"    Confidence         : {diagnosis.confidence * 100:.1f}%")
    print(f"    Severity           : {diagnosis.severity.value.upper()}")
    print(f"    Summary            : {diagnosis.summary}")

    print("\n[3] Ranked Root Cause Hypotheses:")
    for i, hyp in enumerate(diagnosis.hypotheses, 1):
        print(f"    #{i} [{hyp.confidence * 100:.0f}% Confidence] {hyp.title}")
        print(f"        Rationale: {hyp.description}")
        if hyp.contributing_factors:
            print(f"        Factors  : {', '.join(hyp.contributing_factors)}")

    # 3. Generate Actionable Developer Remediation
    rec_engine = RecommendationEngine()
    recommendations = rec_engine.recommend(diagnosis)

    print(f"\n[4] Prioritized Developer Remediation Actions ({len(recommendations)}):")
    for rec in recommendations:
        print(f"    [P{rec.priority}] {rec.title} ({rec.action_type})")
        print(f"        Fix: {rec.description}")

    # 4. Generate Interactive Standalone HTML Report
    html_path = "rag_diagnostic_report.html"
    save_html_report(diagnosis, html_path, trace=rag_trace, title="RAG Diagnostic Report")
    print(f"\n[5] Saved Interactive Offline HTML Report to: {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
