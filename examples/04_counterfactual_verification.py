"""Example 04: Counterfactual Fix Simulation & Before/After Verification."""

from llm_reliability import SyntheticTraceGenerator, VerificationEngine


def main() -> None:
    print("=" * 70)
    print("Example 04: Counterfactual Simulation & Fix Verification Engine")
    print("=" * 70)

    generator = SyntheticTraceGenerator()

    # 1. Baseline failing trace (high duplicate chunk redundancy)
    baseline_trace = generator.generate_duplicate_chunks_trace()
    # 2. Candidate fixed trace (deduplicated clean RAG context)
    candidate_trace = generator.generate_clean_trace()

    print(f"\n[1] Baseline Trace  : {baseline_trace.trace_id}")
    print(f"    Candidate Trace : {candidate_trace.trace_id}")

    # 3. Verify Fix
    engine = VerificationEngine()
    report = engine.verify_fix(baseline_trace, candidate_trace)

    print("\n[2] Verification Outcome:")
    print(f"    * Fix Is Verified     : {report.is_verified}")
    print(
        f"    * Severity Transition : {report.severity_before.value.upper()} -> {report.severity_after.value.upper()}"
    )
    print(f"    * Resolved Failures   : {[f.value for f in report.resolved_failures]}")
    print(f"    * New Regressions     : {[f.value for f in report.new_regressions]}")
    print(f"\n[3] Executive Summary:\n    {report.summary}")

    print("\n[4] Diagnostic Metric Transitions:")
    for mc in report.metric_comparisons[:5]:
        pct_str = f"({mc.percent_change:+.1f}%)" if mc.percent_change is not None else ""
        print(
            f"    * {mc.name:30s} : {str(mc.before_value):6s} -> {str(mc.after_value):6s} {pct_str}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
