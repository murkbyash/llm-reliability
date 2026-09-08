"""Synthetic failure generation and diagnostic accuracy benchmark suite."""

from llm_reliability.benchmark.generator import SyntheticTraceGenerator
from llm_reliability.benchmark.models import (
    BenchmarkReport,
    BenchmarkSample,
    CategoryAccuracyMetrics,
)
from llm_reliability.benchmark.runner import BenchmarkRunner, run_benchmark

__all__ = [
    "SyntheticTraceGenerator",
    "BenchmarkRunner",
    "BenchmarkSample",
    "BenchmarkReport",
    "CategoryAccuracyMetrics",
    "run_benchmark",
]
