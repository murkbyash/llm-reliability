"""Example 05: Multi-Run Batch Comparative Regression Engine."""

from llm_reliability import RegressionEngine, SyntheticTraceGenerator


def main() -> None:
    print("=" * 70)
    print("Example 05: Multi-Run Batch Comparative Regression Analysis")
    print("=" * 70)

    generator = SyntheticTraceGenerator()

    # 1. Generate baseline batch and candidate batch
    baseline_batch = [s.trace for s in generator.generate_benchmark_suite(samples_per_category=2)]
    candidate_batch = [s.trace for s in generator.generate_benchmark_suite(samples_per_category=2)]

    print("\n[1] Comparing Two Execution Batches:")
    print(f"    * Baseline Traces  : {len(baseline_batch)}")
    print(f"    * Candidate Traces : {len(candidate_batch)}")

    # 2. Run Comparative Regression
    engine = RegressionEngine()
    report = engine.compare_batches(baseline_batch, candidate_batch)

    print("\n[2] Regression Verdict & Score:")
    print(f"    * Verdict           : {report.verdict.value.upper()}")
    print(f"    * Regression Score  : {report.regression_score:.2f} (0.0=Clean, 1.0=Severe)")

    print("\n[3] Failure Rate Shift:")
    print(f"    * Baseline Failures : {report.baseline_failure_summary.failure_rate * 100:.1f}%")
    print(f"    * Candidate Failures: {report.candidate_failure_summary.failure_rate * 100:.1f}%")

    print("\n[4] Latency Percentiles (Baseline vs Candidate):")
    print(
        f"    * p50 Median Latency: {report.latency_baseline.p50:.1f}ms -> {report.latency_candidate.p50:.1f}ms"
    )
    print(
        f"    * p95 Tail Latency  : {report.latency_baseline.p95:.1f}ms -> {report.latency_candidate.p95:.1f}ms"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
