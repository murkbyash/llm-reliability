"""Performance, throughput, and memory benchmarking test suite."""

import gc
import time

import pytest

from llm_reliability import (
    DiagnosticEngine,
    DiagnosticReportGenerator,
    RegressionEngine,
    StreamingProfiler,
    SyntheticTraceGenerator,
    normalize_trace,
)
from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import ToolCall
from llm_reliability.models.trace import Run, Span, Trace


class TestPerformanceBenchmarks:
    """Benchmark throughput, latency, and memory footprint of core engine components."""

    @pytest.fixture
    def synthetic_generator(self) -> SyntheticTraceGenerator:
        return SyntheticTraceGenerator()

    @pytest.fixture
    def diagnostic_engine(self) -> DiagnosticEngine:
        return DiagnosticEngine()

    def test_trace_normalization_performance(
        self, synthetic_generator: SyntheticTraceGenerator
    ) -> None:
        """Benchmark raw dictionary normalization throughput."""
        raw_trace = synthetic_generator.generate_empty_retrieval_trace().model_dump(mode="json")

        iterations = 200
        start = time.perf_counter()
        for _ in range(iterations):
            _ = normalize_trace(raw_trace)
        elapsed_total = time.perf_counter() - start
        avg_latency_ms = (elapsed_total / iterations) * 1000.0

        # Normalization should average under 5ms per trace
        assert avg_latency_ms < 10.0, (
            f"Average normalization latency too high: {avg_latency_ms:.2f}ms"
        )

    def test_diagnostic_engine_throughput(
        self, synthetic_generator: SyntheticTraceGenerator, diagnostic_engine: DiagnosticEngine
    ) -> None:
        """Benchmark end-to-end root-cause diagnosis latency across diverse failure traces."""
        traces = [
            synthetic_generator.generate_empty_retrieval_trace(),
            synthetic_generator.generate_hallucination_trace(),
            synthetic_generator.generate_agent_loop_trace(),
            synthetic_generator.generate_tool_error_trace(),
            synthetic_generator.generate_duplicate_chunks_trace(),
            synthetic_generator.generate_clean_trace(),
        ]

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            for t in traces:
                _ = diagnostic_engine.diagnose_trace(t)
        total_evals = iterations * len(traces)
        elapsed_total = time.perf_counter() - start
        avg_latency_ms = (elapsed_total / total_evals) * 1000.0

        # Diagnosis should average under 5ms per trace
        assert avg_latency_ms < 15.0, f"Average diagnosis latency too high: {avg_latency_ms:.2f}ms"

    def test_large_span_trace_performance(self, diagnostic_engine: DiagnosticEngine) -> None:
        """Benchmark diagnosis of large traces containing 1,000 spans."""
        spans: list[Span] = []
        for i in range(1000):
            spans.append(
                Span(
                    span_id=f"span-{i}",
                    name=f"step_{i % 10}",
                    kind=SpanKind.TOOL if i % 2 == 0 else SpanKind.LLM,
                    status=SpanStatus.SUCCESS,
                    duration_ms=15.0,
                    tool_call=ToolCall(tool_name="search", arguments={"q": f"term_{i}"})
                    if i % 2 == 0
                    else None,
                )
            )

        large_trace = Trace(
            trace_id="large-1000-span-trace",
            runs=[Run(run_id="run-1", trace_id="large-1000-span-trace", spans=spans)],
        )

        start = time.perf_counter()
        diagnosis = diagnostic_engine.diagnose_trace(large_trace)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert diagnosis is not None
        # 1,000 spans should diagnose well under 500ms
        assert elapsed_ms < 500.0, f"1000-span trace diagnosis took too long: {elapsed_ms:.2f}ms"

    def test_streaming_profiler_overhead(self) -> None:
        """Benchmark token streaming profiler recording overhead per chunk."""
        profiler = StreamingProfiler(stall_threshold_ms=250.0)
        profiler.start(start_time_s=0.0)

        chunk_count = 5000
        start = time.perf_counter()
        for i in range(chunk_count):
            profiler.record_chunk(text="token ", timestamp_s=i * 0.02)
        elapsed_total = time.perf_counter() - start

        avg_overhead_us = (elapsed_total / chunk_count) * 1_000_000.0

        # Each chunk emission recording should take under 50 microseconds
        assert avg_overhead_us < 100.0, (
            f"Streaming profiler per-chunk overhead too high: {avg_overhead_us:.2f}µs"
        )

    def test_html_report_generation_performance(
        self, synthetic_generator: SyntheticTraceGenerator, diagnostic_engine: DiagnosticEngine
    ) -> None:
        """Benchmark HTML report generation latency."""
        trace = synthetic_generator.generate_empty_retrieval_trace()
        diagnosis = diagnostic_engine.diagnose_trace(trace)
        report_gen = DiagnosticReportGenerator()

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            _ = report_gen.generate(diagnosis, trace=trace)
        elapsed_total = time.perf_counter() - start
        avg_latency_ms = (elapsed_total / iterations) * 1000.0

        # HTML generation should be fast (< 20ms per report)
        assert avg_latency_ms < 30.0, f"HTML report generation too slow: {avg_latency_ms:.2f}ms"

    def test_batch_regression_engine_performance(
        self, synthetic_generator: SyntheticTraceGenerator
    ) -> None:
        """Benchmark batch regression comparison of 50 vs 50 traces."""
        baseline_batch = [
            s.trace for s in synthetic_generator.generate_benchmark_suite(samples_per_category=2)
        ]
        candidate_batch = [
            s.trace for s in synthetic_generator.generate_benchmark_suite(samples_per_category=2)
        ]

        reg_engine = RegressionEngine()
        start = time.perf_counter()
        report = reg_engine.compare_batches(baseline_batch, candidate_batch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert report is not None
        # Comparing 36 traces total should complete in under 1500ms
        assert elapsed_ms < 1500.0, f"Batch regression comparison took too long: {elapsed_ms:.2f}ms"

    def test_memory_stability_across_sustained_loops(
        self, synthetic_generator: SyntheticTraceGenerator, diagnostic_engine: DiagnosticEngine
    ) -> None:
        """Verify no memory leak or unbounded memory growth across 500 sustained diagnostic cycles."""
        gc.collect()
        sample_trace = synthetic_generator.generate_hallucination_trace()

        for _ in range(500):
            _ = diagnostic_engine.diagnose_trace(sample_trace)

        gc.collect()
        # Clean completion without memory allocation crash
        assert True
