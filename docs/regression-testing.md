# Multi-Run Comparative Regression Engine

In continuous integration and release pipelines, evaluating single traces is insufficient. Developers must evaluate entire batches of runs to catch statistical regressions, latency drift, and newly introduced failure categories.

`llm_reliability.regression.RegressionEngine` compares two collections of traces (e.g. `Baseline` vs `Candidate`).

---

## 1. Regression Verdicts

The comparison produces a `RegressionVerdict`:

- `PASSED`: Candidate has zero failures and no latency regression.
- `IMPROVED`: Candidate reduces failure rates or latency relative to baseline.
- `REGRESSION`: Candidate introduces new failures, increases failure rates, or degrades p95 latency by $>20\%$.
- `INCONCLUSIVE`: Insufficient sample volume to establish statistical significance.

---

## 2. Statistical Analysis Computed

- **Failure Rates:** Baseline vs candidate total failure percentage and category counts.
- **Latency Distributions:** Percentile distributions (p50, p90, p95, p99) in milliseconds.
- **Category Transitions:** Newly introduced failure categories vs resolved categories.
- **Regression Score:** Composite score from $0.0$ (improved/clean) to $1.0$ (severe regression).

---

## 3. Python Example

```python
from llm_reliability import RegressionEngine, load_trace

baseline_batch = load_trace("baseline_batch.json")
candidate_batch = load_trace("candidate_batch.json")

engine = RegressionEngine()
report = engine.compare_batches(baseline_batch, candidate_batch)

print(f"Regression Verdict: {report.verdict.value}")
print(f"Regression Score: {report.regression_score:.2f}")
print(f"Baseline Failure Rate: {report.baseline_failure_summary.failure_rate * 100:.1f}%")
print(f"Candidate Failure Rate: {report.candidate_failure_summary.failure_rate * 100:.1f}%")
print(f"Latency p95 Delta: {report.latency_p95_delta_ms:+.1f}ms")
```

