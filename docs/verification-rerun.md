# Counterfactual Simulation & Fix Verification

When modifying prompt templates, retrieval filters, or agent guardrails, developers need guarantees that a fix actually resolves the failure mode without introducing new regressions.

`llm_reliability.verification.VerificationEngine` provides:
1. **Fix Verification (`verify_fix`):** Comparing before vs after execution traces.
2. **Counterfactual Simulation (`simulate_intervention`):** Simulating configuration fixes directly against raw traces.

---

## 1. Counterfactual Interventions

You can simulate how a trace would behave under adjusted parameters:

- `TOP_K`: Simulate reducing top-k candidates from $k=10$ to $k=3$.
- `SCORE_CUTOFF`: Simulate applying a relevance threshold cutoff ($\text{score} \ge 0.70$).
- `DEDUPLICATION`: Simulate pre-context Jaccard deduplication.
- `AGENT_GUARDRAIL`: Simulate middleware terminating loops after $N$ steps.

---

## 2. Before / After Comparison Report

The verification report (`VerificationReport`) produces:
- `is_verified`: Boolean flag indicating whether the fix improved reliability.
- `severity_before` $\to$ `severity_after`: Change in severity assessment (e.g. `HIGH` $\to$ `INFO`).
- `resolved_failures`: List of failure categories successfully eliminated.
- `new_regressions`: New failure categories inadvertently introduced by the fix.
- `metric_comparisons`: Before vs after values and percentage changes for all diagnostic metrics.

---

## 3. Python Example

```python
from llm_reliability import VerificationEngine, load_trace

baseline_trace = load_trace("before_fix.json")
candidate_trace = load_trace("after_fix.json")

verifier = VerificationEngine()
report = verifier.verify_fix(baseline_trace, candidate_trace)

print(f"Fix Verified: {report.is_verified}")
print(f"Severity Change: {report.severity_before.value} -> {report.severity_after.value}")
print(f"Resolved Failures: {[f.value for f in report.resolved_failures]}")
print(f"Regressions: {[f.value for f in report.new_regressions]}")
print(f"\nSummary:\n{report.summary}")
```

