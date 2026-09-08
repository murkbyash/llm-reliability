# CI/CD Evaluation Harness & Gatekeeper Action

The `llm-reliability` evaluation harness enables development teams to automate reliability regression testing and enforce strict SLA gatekeeping directly inside CI/CD pull requests before code merges.

## Overview

The Gatekeeper engine evaluates candidate execution traces against baseline traces or fixed SLA criteria:
- **Failure Rate Ceilings**: Block pull requests if candidate error rates exceed a defined threshold.
- **Regression Score Limits**: Prevent regressions in failure rate or newly introduced failure categories.
- **Latency SLAs**: Enforce p95/p99 execution latency constraints.
- **Grounding Standards**: Require minimum average grounding / faithfulness scores.
- **Disallowed Categories**: Disallow critical failure categories such as `HALLUCINATION` or `AGENT_LOOP`.
- **Custom Declarative Policies**: Evaluate custom rules written in YAML or JSON.

---

## CLI Usage

Run gatekeeping directly from your CI runner:

```bash
# Basic failure rate gate
llm-reliability gate candidate_trace.json --max-failure-rate 0.0

# Comparative regression gate with baseline
llm-reliability gate candidate_trace.json \
  --baseline baseline_trace.json \
  --max-regression 0.0 \
  --max-latency 1200 \
  --min-grounding 0.85 \
  --disallowed-categories HALLUCINATION,AGENT_LOOP \
  --comment-file pr_comment.md
```

### Exit Codes
- `0`: All gates and SLAs passed successfully.
- `1`: One or more gate criteria were violated (fails the CI build).

---

## Python API Usage

```python
from llm_reliability import Gatekeeper, GatekeeperConfig, evaluate_gate, load_trace

# 1. Convenience function
report = evaluate_gate(
    candidate_file="candidate.json",
    baseline_file="baseline.json",
    config=GatekeeperConfig(
        max_failure_rate=0.05,
        max_regression_score=0.0,
        max_latency_p95_ms=2000.0,
    ),
)

if not report.passed:
    print(f"Gatekeeper rejected build: {report.violations}")
```
