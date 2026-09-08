# LLM Reliability Analyzer Documentation

Welcome to the official documentation for **LLM Reliability Analyzer** (`llm-reliability`).

LLM Reliability Analyzer is a **production-grade, local-first, evidence-based reliability analyzer and root-cause diagnosis engine** for LLM, RAG, and AI Agent systems.

---

## 🎯 What is LLM Reliability Analyzer?

Traditional software systems fail deterministically with explicit stack traces pointing to line numbers. LLM, RAG, and Agent systems fail probabilistically across multi-step retrieval, prompt injection, reasoning loops, tool invocation, and token streaming.

When an AI system returns an incorrect, slow, or hallucinated response, developers need answers to four critical questions:
1. **What failed?** (Failure classification)
2. **Why did it fail?** (Ranked root-cause hypotheses backed by mathematical evidence)
3. **How do I fix it?** (Prioritized, actionable remediation advice and code snippets)
4. **Did my fix work?** (Counterfactual simulation and batch comparative regression testing)

LLM Reliability Analyzer provides complete local-first answers with **₹0 cloud API dependencies**.

---

## 🧭 Documentation Navigation

| Guide | Description |
| :--- | :--- |
| [**Getting Started**](getting-started.md) | Installation, quickstart tutorial, CLI commands, and Python SDK. |
| [**Architecture & Philosophy**](architecture.md) | Layered architecture, local-first execution, and data model schemas. |
| [**RAG Failure Analysis**](rag-analysis.md) | Retrieval score distributions, duplicate chunk detection, and dropoff analysis. |
| [**Grounding & Faithfulness**](grounding-analysis.md) | Sentence-level claim support, lexical overlap, and entity hallucination tracking. |
| [**Agent & Tool Analysis**](agent-analysis.md) | Repetitive loops, alternating cycles, schema validation, and step budget limits. |
| [**Root Cause Diagnosis**](root-cause-diagnosis.md) | Multi-hypothesis ranking, confidence scoring, and evidence chains. |
| [**Recommendation Engine**](recommendation-engine.md) | Actionable developer advice, parameter suggestions, and code snippets. |
| [**Verification & Simulation**](verification-rerun.md) | Counterfactual simulation and before/after fix verification. |
| [**Regression Testing**](regression-testing.md) | Multi-run batch comparative regression and latency percentile tracking. |
| [**Failure Signatures & Patterns**](patterns-and-signatures.md) | Normalized failure fingerprinting and recurring failure clustering. |
| [**Custom Policies & Rules**](custom-policies.md) | Defining threshold policies, custom rules, and strict grounding gates. |
| [**Benchmark Suite**](benchmark-suite.md) | Programmatic synthetic generator and 45-sample ground-truth benchmark. |
| [**OpenTelemetry Ingestion**](opentelemetry.md) | Importing OTLP JSON, OpenInference, OpenLLMetry, and LangSmith traces. |
| [**Live Tracing SDK**](tracing-sdk.md) | Zero-overhead function decorators, context managers, and provider hooks. |
| [**Streaming & Latency Profiler**](streaming-profiler.md) | TTFT, TPS throughput, inter-token percentiles, and stall detection. |
| [**Interactive HTML Reports**](html-reports.md) | Generating self-contained, offline interactive HTML waterfall dashboards. |
| [**Cookbook & Examples**](cookbook.md) | Practical recipes for LangChain, LlamaIndex, OpenAI, and FastAPI. |
| [**API Reference**](api-reference.md) | Comprehensive reference for all public classes, functions, and models. |
| [**Troubleshooting & FAQ**](troubleshooting.md) | Common errors, diagnostic interpretations, and FAQ. |

---

## ⚡ Quick Example

```python
from llm_reliability import diagnose, load_trace

# 1. Load an execution trace (supports JSON, JSONL, OTel, LangSmith)
trace = load_trace("trace.json")

# 2. Run deterministic root cause diagnosis
diagnosis = diagnose(trace)

# 3. Inspect findings
print(f"Primary Root Cause: {diagnosis.primary_category.value}")
print(f"Confidence: {diagnosis.confidence * 100:.1f}%")
print(f"Severity: {diagnosis.severity.value}")

# 4. Review prioritized remediation recommendations
for rec in diagnosis.recommendations:
    print(f"[P{rec.priority}] {rec.title}: {rec.description}")
```

