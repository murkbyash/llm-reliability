# API Reference

Complete reference for all public classes, functions, enums, and data models exported by `llm_reliability`.

---

## 1. Top-Level Functions

### `diagnose(source)`
Diagnoses an execution trace, run dictionary, OTel payload, or trace file.
- **Parameters:** `source: Trace | Run | str | Path | dict | list | TextIO`
- **Returns:** `Diagnosis`

### `load_trace(source)`
Loads and normalizes traces from JSON, JSONL, dictionaries, OpenTelemetry, or LangSmith formats.
- **Parameters:** `source: str | Path | dict | list | TextIO`
- **Returns:** `Trace`

### `normalize_trace(raw_data)`
Converts raw dictionary or span list payloads into a canonical `Trace` instance.
- **Parameters:** `raw_data: dict | list`
- **Returns:** `Trace`

### `run_benchmark(samples_per_category=5)`
Executes the diagnostic accuracy benchmark suite across 9 taxonomy categories.
- **Parameters:** `samples_per_category: int = 5`
- **Returns:** `BenchmarkReport`

### `render_html_report(diagnosis, trace=None, title="LLM Reliability Diagnostic Report")`
Renders a `Diagnosis` and optional `Trace` hierarchy into an interactive HTML string.
- **Returns:** `str`

### `save_html_report(diagnosis, output_path, trace=None, title="LLM Reliability Diagnostic Report")`
Generates and writes a self-contained interactive HTML report to disk.
- **Returns:** `Path`

---

## 2. Core Engines

- `DiagnosticEngine()`: Multi-layer root cause diagnosis engine.
- `RecommendationEngine()`: Developer remediation advice generator.
- `VerificationEngine()`: Fix verification and counterfactual simulation engine.
- `RegressionEngine()`: Batch comparative regression engine.
- `PatternDetectionEngine()`: Failure clustering and signature engine.
- `PolicyEngine(policy=None)`: Custom policy and threshold evaluation engine.
- `SyntheticTraceGenerator(seed=42)`: Synthetic failure trace generator.
- `BenchmarkRunner()`: Benchmark evaluation runner.
- `OTelImporter`: OpenTelemetry, OpenInference, and LangSmith ingestion adapter.
- `StreamingProfiler(stall_threshold_ms=500.0)`: Token latency, TTFT, and stalling profiler.
- `Tracer()`: Live in-memory tracing SDK.

---

## 3. Decorators & Wrappers

- `@trace(name=None, kind=SpanKind.CUSTOM)`: Decorator for custom spans.
- `@trace_llm(model="unknown", name=None)`: Decorator for LLM completions.
- `@trace_tool(tool_name=None)`: Decorator for tool functions.
- `@trace_retrieval(query_param="query")`: Decorator for retrieval functions.
- `wrap_openai(client)`: Wraps an OpenAI client instance.
- `wrap_anthropic(client)`: Wraps an Anthropic client instance.
- `wrap_tool(tool_fn, tool_name=None)`: Wraps a tool callable.
- `wrap_stream(generator, on_complete=None, stall_threshold_ms=500.0)`: Synchronous stream profiler wrapper.
- `wrap_async_stream(async_generator, on_complete=None, stall_threshold_ms=500.0)`: Asynchronous stream profiler wrapper.

---

## 4. Primary Data Models & Enums

- **Enums:** `FailureCategory`, `Severity`, `SpanKind`, `SpanStatus`, `EvidenceType`, `SupportStatus`, `RegressionVerdict`.
- **Trace Hierarchy:** `Trace`, `Run`, `Span`.
- **Execution Units:** `LLMCall`, `RetrievalStep`, `RetrievedDocument`, `ToolCall`, `ToolResult`, `TokenUsage`, `FinalResponse`.
- **Diagnostic Models:** `Diagnosis`, `Hypothesis`, `Failure`, `Evidence`, `Metric`, `Recommendation`.
- **Verification & Regression:** `VerificationReport`, `CounterfactualResult`, `MetricComparison`, `BatchComparisonReport`, `DistributionSummary`, `FailureRateSummary`.
- **Patterns & Policies:** `FailureSignature`, `FailureCluster`, `PatternAnalysisReport`, `DiagnosticPolicy`, `ThresholdConfig`, `CustomRule`, `PolicyViolation`, `PolicyEvaluationResult`.
- **Streaming Models:** `TokenLatencyProfile`, `TokenChunk`, `StallEvent`, `InterTokenLatencyStats`.

