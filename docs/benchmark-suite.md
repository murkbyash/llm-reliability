# Synthetic Trace Generator & Benchmark Suite

To guarantee diagnostic precision and prevent regressions in root cause classification, LLM Reliability Analyzer includes a programmatic synthetic trace generator and an automated calibration benchmark runner.

---

## 1. Synthetic Failure Trace Generator

`llm_reliability.benchmark.SyntheticTraceGenerator` programmatically generates synthetic traces with controlled failure injection across all failure categories:

- `generate_clean_trace()`: Healthy RAG execution with grounded answer.
- `generate_empty_retrieval_trace()`: 0 document retrieval result (`RETRIEVAL_FAILURE`).
- `generate_low_relevance_trace()`: Irrelevant candidates ($< 0.40$ score) (`RETRIEVAL_FAILURE`).
- `generate_duplicate_chunks_trace()`: High Jaccard redundancy (`CONTEXT_CONSTRUCTION_FAILURE`).
- `generate_hallucination_trace()`: Factually unsupported statements (`GROUNDING_FAILURE`).
- `generate_agent_loop_trace()`: Infinite repetitive tool calling (`AGENT_LOOP`).
- `generate_tool_error_trace()`: Internal tool exception (`TOOL_ERROR`).
- `generate_tool_argument_error_trace()`: Missing required parameter keys (`SCHEMA_VIOLATION`).
- `generate_llm_api_error_trace()`: Provider 429 rate limit error (`LLM_CALL_FAILURE`).

---

## 2. Benchmark Runner

`llm_reliability.benchmark.BenchmarkRunner` evaluates diagnostic accuracy across 45 ground-truth labeled benchmark samples:

```python
from llm_reliability import run_benchmark

report = run_benchmark(samples_per_category=5)

print(f"Overall Accuracy: {report.overall_accuracy * 100:.1f}%")
print(f"Macro Precision:  {report.macro_precision:.2f}")
print(f"Macro Recall:     {report.macro_recall:.2f}")
print(f"Macro F1 Score:   {report.macro_f1:.2f}")
print(f"Benchmark Passed: {report.passed}")

for cat_name, metrics in report.category_metrics.items():
    print(
        f"  • {cat_name:28s}: Precision: {metrics.precision:.2f}, Recall: {metrics.recall:.2f}, F1: {metrics.f1_score:.2f}"
    )
```

