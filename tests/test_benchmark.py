"""Tests for Synthetic Failure Trace Generator & Benchmark Suite."""

from llm_reliability import (
    BenchmarkReport,
    BenchmarkRunner,
    BenchmarkSample,
    FailureCategory,
    SyntheticTraceGenerator,
    Trace,
    run_benchmark,
)


class TestBenchmarkSuite:
    """Deterministic validation of synthetic failure generators and diagnostic accuracy benchmarking."""

    def test_synthetic_generators_produce_valid_traces(self) -> None:
        gen = SyntheticTraceGenerator()

        traces = [
            gen.generate_clean_trace(),
            gen.generate_empty_retrieval_trace(),
            gen.generate_low_relevance_trace(),
            gen.generate_duplicate_chunks_trace(),
            gen.generate_hallucination_trace(),
            gen.generate_agent_loop_trace(),
            gen.generate_tool_error_trace(),
            gen.generate_tool_argument_error_trace(),
            gen.generate_llm_api_error_trace(),
        ]

        for t in traces:
            assert isinstance(t, Trace)
            assert len(t.runs) >= 1
            assert len(t.runs[0].spans) >= 1
            assert t.trace_id != ""

    def test_synthetic_benchmark_suite_generation(self) -> None:
        gen = SyntheticTraceGenerator()
        samples = gen.generate_benchmark_suite(samples_per_category=3)

        assert len(samples) == 3 * 9  # 9 categories * 3 samples = 27 samples
        for s in samples:
            assert isinstance(s, BenchmarkSample)
            assert s.sample_id != ""
            assert isinstance(s.trace, Trace)
            assert isinstance(s.ground_truth_category, FailureCategory)

    def test_benchmark_runner_accuracy_target(self) -> None:
        runner = BenchmarkRunner()
        report: BenchmarkReport = runner.run(min_accuracy_target=0.90)

        assert report.total_samples > 0
        assert report.overall_accuracy >= 0.90
        assert report.macro_precision >= 0.90
        assert report.macro_recall >= 0.90
        assert report.macro_f1 >= 0.90
        assert report.passed is True
        assert len(report.category_metrics) >= 7

    def test_benchmark_runner_empty_input(self) -> None:
        runner = BenchmarkRunner()
        report = runner.run(samples=[])
        assert report.total_samples == 0
        assert report.passed is True

    def test_run_benchmark_convenience_function(self) -> None:
        report = run_benchmark(samples_per_category=2)
        assert isinstance(report, BenchmarkReport)
        assert report.total_samples == 18
        assert report.passed is True
        assert "Benchmark Evaluation: PASSED" in report.summary
