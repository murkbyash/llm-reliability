# Architecture & Design Philosophy

LLM Reliability Analyzer is engineered with a strict **local-first, layered, and zero-cost design philosophy**.

---

## 1. Core Architectural Pipeline

The system is structured as an acyclic feed-forward pipeline where each layer is strictly isolated from subsequent presentation layers:

```text
                  LLM APPLICATION
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Live Tracing &              External Traces
 Streaming Profiler          (OTel / LangSmith / etc.)
 (Decorators / Wrappers)              │
          │                   ┌───────┴────────┐
          │                   │ OTel Importer  │
          │                   └───────┬────────┘
          └─────────────┬─────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Normalization Layer  │ (Multi-Format Ingestion)
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   Analysis Engine    │ (RAG, Grounding, Agent, LLM)
             └──────────┬───────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       RAG Analyzer  Agent Analyzer  Grounding Analyzer
          │             │             │
          └─────────────┼─────────────┘
                        ▼
             ┌──────────────────────┐
             │  Root Cause Engine   │ (DiagnosticEngine)
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │    Recommendation    │ (RecommendationEngine)
             │        Engine        │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │     Verification     │ (VerificationEngine & RegressionEngine)
             │     Rerun Engine     │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Policy & Benchmark   │ (PolicyEngine & BenchmarkRunner)
             └──────────┬───────────┘
                        │
                 ┌──────┴───────┐
                 ▼              ▼
           Terminal CLI    HTML Reports
```

---

## 2. Layer Independence Principle

1. **Ingestion Layer (`llm_reliability.normalization`, `llm_reliability.otel`):**
   Converts heterogeneous inputs (dictionaries, JSONL, OTLP OpenTelemetry, LangSmith run trees, raw spans) into the canonical strongly-typed `Trace` model.
2. **Analysis Layer (`llm_reliability.rag`, `llm_reliability.grounding`, `llm_reliability.agent`):**
   Computes statistical metrics (relevance distributions, token budgets, lexical overlap ratios, cycle traps).
3. **Diagnostic Layer (`llm_reliability.diagnosis`):**
   Synthesizes domain metrics into ranked `Hypothesis` objects, assigns confidence values ($0.0 \dots 1.0$), compiles evidence items, and assigns overall `Severity`.
4. **Action Layer (`llm_reliability.recommendations`, `llm_reliability.verification`):**
   Generates actionable remediation guidance (`Recommendation`) and performs counterfactual simulations.
5. **Presentation Layer (`llm_reliability.cli`, `llm_reliability.report`):**
   Renders diagnostic outputs across Text, Markdown, JSON, and standalone offline HTML dashboards.

---

## 3. Data Model Hierarchy

All data structures are implemented using Pydantic v2 schemas:

```text
Trace
└── runs: list[Run]
    ├── run_id: str
    ├── trace_id: str
    ├── spans: list[Span]
    │   ├── span_id: str
    │   ├── parent_span_id: str | None
    │   ├── name: str
    │   ├── kind: SpanKind (LLM, RETRIEVAL, TOOL, AGENT, CHAIN, CUSTOM, ROOT)
    │   ├── status: SpanStatus (SUCCESS, ERROR, UNSET)
    │   ├── duration_ms: float
    │   ├── llm_call: LLMCall (model, prompt, response, token_usage)
    │   ├── retrieval: RetrievalStep (query, documents)
    │   ├── tool_call: ToolCall (tool_name, arguments)
    │   └── tool_result: ToolResult (tool_name, output, error)
    └── final_response: FinalResponse (text)
```

Diagnostic Output Hierarchy:

```text
Diagnosis
├── primary_category: FailureCategory
├── root_cause: FailureCategory
├── confidence: float (0.0 to 1.0)
├── severity: Severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
├── summary: str
├── uncertainty_note: str | None
├── failures: list[Failure]
├── hypotheses: list[Hypothesis]
│   ├── category: FailureCategory
│   ├── confidence: float
│   ├── title: str
│   ├── description: str
│   ├── contributing_factors: list[str]
│   └── evidence: list[Evidence]
├── recommendations: list[Recommendation]
│   ├── title: str
│   ├── description: str
│   ├── action_type: str
│   ├── priority: int (1 = Highest)
│   └── rationale: str | None
└── metrics: list[Metric]
```

---

## 4. Local-First & Zero Paid API Philosophy

- **Zero Cloud Costs (₹0 / $0):** Core ingestion, metric extraction, failure detection, hypothesis generation, recommendations, counterfactual simulations, regression evaluations, CLI rendering, pattern clustering, custom policy evaluations, benchmark suites, OpenTelemetry imports, live tracing SDK, streaming latency profiling, and interactive HTML report generation require no API keys or third-party cloud connections.
- **Air-Gapped & Privacy-Preserving:** Traces and proprietary documents never leave your machine or local infrastructure.

