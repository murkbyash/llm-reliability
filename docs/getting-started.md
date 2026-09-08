# Getting Started

This tutorial walks you through installing `llm-reliability`, diagnosing your first execution trace, using the command-line interface (CLI), and integrating the Python SDK into your existing application.

---

## 1. Installation

`llm-reliability` requires Python **3.10** or newer.

```bash
# Install core package
pip install llm-reliability

# Or install from source in editable mode
git clone https://github.com/murkbyash/llm-reliability.git
cd llm-reliability
pip install -e ".[dev]"
```

---

## 2. Using the Command Line Interface (CLI)

The package provides a built-in CLI executable named `llm-reliability`.

### Diagnosing a Trace

```bash
# Terminal text output
llm-reliability diagnose sample_trace.json

# Markdown output
llm-reliability diagnose sample_trace.json --format markdown

# JSON output (for CI/CD automation)
llm-reliability diagnose sample_trace.json --format json

# Interactive standalone HTML dashboard
llm-reliability diagnose sample_trace.json --format html --output report.html
```

### Verifying a Fix

Compare before vs after traces to verify whether a code or prompt change resolved the failure without introducing regressions:

```bash
llm-reliability verify baseline_trace.json candidate_trace.json
```

### Batch Comparative Regression Testing

Compare two batches of traces (e.g. staging vs production) for statistical regressions, latency drift, and newly introduced failure categories:

```bash
llm-reliability compare baseline_batch.json candidate_batch.json
```

---

## 3. Python SDK Quickstart

### Basic Diagnosis

```python
from llm_reliability import diagnose, load_trace

# Load trace from disk or raw dictionary
trace = load_trace("sample_trace.json")

# Execute multi-layer root-cause analysis
diagnosis = diagnose(trace)

print(f"Root Cause: {diagnosis.primary_category.value}")
print(f"Confidence: {diagnosis.confidence * 100:.1f}%")
print(f"Severity: {diagnosis.severity.value}")
print(f"Summary: {diagnosis.summary}")

# Print ranked hypotheses
for hyp in diagnosis.hypotheses:
    print(f"- [{hyp.confidence * 100:.0f}%] {hyp.title}: {hyp.description}")
```

### Live Tracing Decorators

Instrument your existing functions with zero-overhead tracing hooks:

```python
from llm_reliability import get_tracer, trace_llm, trace_tool, trace_retrieval, diagnose

tracer = get_tracer()


@trace_retrieval(query_param="query")
def search_docs(query: str) -> list[dict]:
    return [{"id": "doc-1", "content": "Python 3.13 free-threading", "score": 0.95}]


@trace_tool(tool_name="calculator")
def compute_math(expression: str) -> float:
    return 42.0


@trace_llm(model="gpt-4o")
def call_model(prompt: str) -> str:
    return "Python 3.13 introduces free-threaded execution."


# Capture execution in a trace context
with tracer.start_trace("user-query-123") as trace_ctx:
    docs = search_docs("Python 3.13 features")
    result = compute_math("21 * 2")
    answer = call_model(f"Context: {docs}. Answer query.")

# Automatically diagnose live trace
diagnosis = diagnose(trace_ctx.trace)
print(f"Status: {diagnosis.primary_category.value}")
```

---

## 4. Streaming Token Latency Profiling

Profile time-to-first-token (TTFT), tokens per second (TPS), inter-token percentiles, and generation stalls:

```python
from llm_reliability import StreamingProfiler, wrap_stream

profiler = StreamingProfiler(stall_threshold_ms=400.0)


def streaming_generator():
    yield "Hello"
    yield " world"
    yield " from"
    yield " streaming!"


def on_done(profile, full_text):
    print(f"TTFT: {profile.time_to_first_token_ms:.1f}ms")
    print(f"TPS: {profile.tokens_per_second:.1f} tokens/s")
    print(f"Stall Count: {profile.stall_count}")


wrapped_stream = wrap_stream(streaming_generator(), on_complete=on_done)
for chunk in wrapped_stream:
    print(chunk, end="", flush=True)
```

