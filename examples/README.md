# LLM Reliability Analyzer — Example Scripts

This directory contains standalone, production-grade example scripts demonstrating all core capabilities of `llm-reliability`.

Every script is 100% self-contained and executes locally with **₹0 cloud API dependencies**.

---

## 📚 Example Catalog

| Script | Title | Description |
| :--- | :--- | :--- |
| [`01_diagnose_rag_failure.py`](01_diagnose_rag_failure.py) | **RAG Failure Diagnosis** | Ingests a failing RAG execution trace, diagnoses empty/low relevance retrieval, extracts evidence, and generates prioritized remediation advice. |
| [`02_diagnose_agent_loop.py`](02_diagnose_agent_loop.py) | **Agent Loop & Tool Trajectory** | Detects repetitive tool calling, alternating ping-pong cycle traps, and step limit exhaustion in agent workflows. |
| [`03_streaming_latency_profiling.py`](03_streaming_latency_profiling.py) | **Streaming Latency & Stalls** | Profiles live token streams, measuring Time-To-First-Token (TTFT), tokens per second (TPS), inter-token percentiles, and detects generation pauses $>250\text{ ms}$. |
| [`04_counterfactual_verification.py`](04_counterfactual_verification.py) | **Counterfactual Fix Verification** | Simulates parameter interventions (top-k tuning, deduplication) and verifies before vs after traces for resolved failures and regressions. |
| [`05_batch_regression_testing.py`](05_batch_regression_testing.py) | **Multi-Run Batch Regression** | Compares baseline vs candidate trace batches for statistical failure rate shifts, latency percentiles (p50, p95), and automated regression verdicts. |
| [`06_custom_policy_evaluation.py`](06_custom_policy_evaluation.py) | **Custom SLA Policies & Rules** | Evaluates production traces against strict enterprise SLA threshold configurations and custom rule conditions. |
| [`07_generate_interactive_html_report.py`](07_generate_interactive_html_report.py) | **Offline Interactive HTML Report** | Generates a standalone, 100% offline single-file HTML diagnostic dashboard with waterfall timeline and span modal inspector. |
| [`08_opentelemetry_import.py`](08_opentelemetry_import.py) | **OpenTelemetry Ingestion** | Ingests and normalizes distributed microservice OTLP JSON / OpenInference traces directly into the diagnostic engine. |

---

## 🚀 Running the Examples

You can run any example directly with Python:

```bash
# Run Example 1 (RAG Diagnosis)
python examples/01_diagnose_rag_failure.py

# Run Example 3 (Streaming Profiling)
python examples/03_streaming_latency_profiling.py

# Run Example 7 (Generate HTML Dashboard)
python examples/07_generate_interactive_html_report.py
```

