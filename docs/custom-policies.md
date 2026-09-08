# Custom Diagnostic Rules & Policy Engine

Organizations frequently have strict domain-specific SLA requirements (e.g. maximum latency budgets, disallowed tool lists, minimum grounding thresholds, mandatory metadata attributes).

`llm_reliability.policies.PolicyEngine` allows defining custom policies via Python, JSON, or YAML.

---

## 1. Policy Configuration Structure

A `DiagnosticPolicy` contains:
- `thresholds`: `ThresholdConfig` defining numerical limits:
  - `min_grounding_score`: Minimum answer faithfulness ratio (default: $0.70$).
  - `min_retrieval_relevance`: Minimum candidate relevance score (default: $0.50$).
  - `max_duplicate_ratio`: Maximum duplicate chunk ratio (default: $0.30$).
  - `max_agent_steps`: Maximum allowed agent iterations (default: $10$).
  - `max_latency_ms`: Maximum acceptable trace execution duration in ms.
  - `disallowed_tools`: Prohibited tool names.
  - `required_metadata_keys`: Mandatory metadata keys (e.g. `user_id`, `environment`).
- `custom_rules`: List of `CustomRule` objects evaluating metric conditions with operators: `>`, `<`, `>=`, `<=`, `==`, `!=`.

---

## 2. YAML Policy Example

```yaml
policy_name: production_rag_sla
strict_mode: true
thresholds:
  min_grounding_score: 0.85
  min_retrieval_relevance: 0.70
  max_duplicate_ratio: 0.20
  max_latency_ms: 1500.0
  disallowed_tools:
    - shell_exec
    - dangerous_eval
  required_metadata_keys:
    - customer_tier
    - environment
custom_rules:
  - rule_id: rule_no_hallucinations
    description: Grounding score must exceed 0.85
    metric_name: answer_grounding_score
    operator: ">="
    threshold_value: 0.85
    severity: HIGH
```

---

## 3. Python Example

```python
from llm_reliability import DiagnosticPolicy, PolicyEngine, ThresholdConfig, load_trace

policy = DiagnosticPolicy(
    policy_name="strict_sla",
    thresholds=ThresholdConfig(
        min_grounding_score=0.90, max_latency_ms=2000.0, disallowed_tools=["raw_exec"]
    ),
)

trace = load_trace("trace.json")
engine = PolicyEngine(policy=policy)
result = engine.evaluate(trace)

print(f"Policy Passed: {result.passed}")
print(f"Violations: {len(result.violations)}")
for violation in result.violations:
    print(f"- [{violation.severity.value}] {violation.rule_name}: {violation.message}")
```

