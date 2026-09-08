# Root Cause Diagnosis Engine

The Root Cause Diagnostic Engine (`DiagnosticEngine`) orchestrates domain metrics across RAG, Grounding, Agent, and LLM calls into a single, unified diagnostic report.

---

## 1. Diagnostic Taxonomy

The engine classifies traces into canonical `FailureCategory` types:

| Failure Category | Description |
| :--- | :--- |
| `RETRIEVAL_FAILURE` | Empty retrieval candidate sets, low semantic similarity scores, or steep score dropoffs. |
| `CONTEXT_CONSTRUCTION_FAILURE` | Excessive duplicate chunks, context character deficits, or unranked context bloat. |
| `GROUNDING_FAILURE` | Factual contradictions, unsupported generated sentences, or fabricated named entities. |
| `AGENT_LOOP` | Repetitive tool calling, alternating ping-pong traps, or step limit exhaustion. |
| `TOOL_ERROR` | Tool runtime exceptions, database timeouts, or HTTP 500 responses. |
| `SCHEMA_VIOLATION` | Malformed tool JSON arguments, missing required parameters, or type schema errors. |
| `LLM_CALL_FAILURE` | Upstream provider 429 rate limits, 500 server errors, context window overruns, or timeouts. |
| `LATENCY_SPIKE` | High Time-To-First-Token (TTFT) or streaming pauses exceeding stall thresholds. |
| `NONE` | Healthy, successful execution meeting all reliability thresholds. |

---

## 2. Ranked Hypotheses & Confidence Scoring

Instead of opaque classification labels, the engine produces ranked `Hypothesis` candidates with mathematical confidence scores:

```text
Hypothesis #1 [Confidence: 95%]
  Statement: Retriever Failed to Return Documents
  Rationale: Vector search returned an empty document candidate set for query.
  Contributing Factors:
    - Embedding mismatch
    - Overly restrictive metadata query filter
    - Empty collection index
  Evidence:
    - [retrieval_score] Retriever returned 0 documents (Threshold: >= 1).
```

---

## 3. Severity Classification

Every diagnosis assigns an overall `Severity` level:
- `CRITICAL`: System-breaking tool error, continuous infinite loop, or total LLM call failure.
- `HIGH`: Empty retrieval result, major factual contradiction, or high-confidence hallucination.
- `MEDIUM`: Low relevance retrieval candidates or moderate duplicate context chunks.
- `LOW`: Minor grounding deficit or slight token bloat.
- `INFO`: Clean execution meeting all reliability thresholds.

---

## 4. Python Example

```python
from llm_reliability import DiagnosticEngine, load_trace

trace = load_trace("failing_trace.json")
engine = DiagnosticEngine()
diagnosis = engine.diagnose(trace)

print(f"Primary Root Cause: {diagnosis.primary_category.value}")
print(f"Confidence: {diagnosis.confidence * 100:.1f}%")
print(f"Severity: {diagnosis.severity.value}")

print("\nHypotheses:")
for hyp in diagnosis.hypotheses:
    print(f"  • [{hyp.confidence * 100:.0f}%] {hyp.title}: {hyp.description}")

print("\nEvidence Items:")
for ev in diagnosis.evidence:
    print(f"  • [{ev.evidence_type.value}] {ev.description}")
```

