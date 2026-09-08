# Developer Recommendation Engine

The Developer Recommendation Engine (`RecommendationEngine`) bridges the gap between diagnostic evidence and developer action, converting root cause findings into prioritized, actionable remediation steps.

---

## 1. Action Types & Priority Levels

Recommendations are prioritized on an ascending scale:
- `Priority 1 (P1 - Urgent)`: Immediate fix required to restore basic functionality (e.g. relaxing query filters, fixing malformed JSON arguments, handling 429 rate limits).
- `Priority 2 (P2 - Recommended)`: High-value improvements to boost accuracy and latency (e.g. adding a cross-encoder reranker, prompt tuning for grounding constraints, implementing response caching).
- `Priority 3 (P3 - Optimization)`: Architectural optimizations (e.g. optimizing chunk size and top-k retrieval parameters, context deduplication).

### Standard Action Types

- `RETRIEVER_CONFIG`: Knowledge base indexing, query filters, HyDE expansion.
- `RERANKING`: Cross-encoder rerankers (e.g., BGE, Cohere), similarity threshold cutoffs.
- `PROMPT_TUNING`: Grounding constraints, negative answering instructions, few-shot refusal examples.
- `AGENT_GUARDRAILS`: Loop detection middleware, maximum retry budgets, state recovery hooks.
- `TOOL_SCHEMA`: JSON Schema validation, strict parameter typing, structured outputs.
- `PROVIDER_FALLBACK`: Exponential backoff retry policies, multi-provider model routing.
- `TOP_K_ADJUSTMENT`: Balancing chunk size against LLM context budget.

---

## 2. Python Example

```python
from llm_reliability import DiagnosticEngine, RecommendationEngine, load_trace

trace = load_trace("rag_trace.json")
diagnosis = DiagnosticEngine().diagnose(trace)

rec_engine = RecommendationEngine()
recommendations = rec_engine.recommend(diagnosis)

for rec in recommendations:
    print(f"\n[P{rec.priority}] {rec.title} ({rec.action_type})")
    print(f"Description: {rec.description}")
    if rec.rationale:
        print(f"Rationale: {rec.rationale}")
```

