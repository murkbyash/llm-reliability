# Failure Signatures & Pattern Detection

When processing thousands of production traces, developers need to cluster recurring failure modes and identify the root structural patterns across disparate executions.

`llm_reliability.patterns.PatternDetectionEngine` normalizes and clusters failures into structural signatures.

---

## 1. Failure Signatures

A `FailureSignature` represents a normalized structural fingerprint:
- `signature_hash`: Deterministic SHA-256 hash derived from the failure category, root cause, and normalized error message.
- `category`: Primary failure category.
- `error_pattern`: Normalized template with dynamic tokens (UUIDs, timestamps, file paths) masked.
- `contributing_factors`: Environmental or algorithmic factors.

---

## 2. Failure Clustering

The engine groups traces into `FailureCluster` objects:
- `frequency`: Total number of occurrences.
- `percentage`: Proportion of all analyzed traces.
- `representative_trace_ids`: Sample trace IDs for debugging.
- `common_contributing_factors`: Aggregated root cause factors.

---

## 3. Python Example

```python
from llm_reliability import PatternDetectionEngine, SyntheticTraceGenerator

# Generate sample traces
generator = SyntheticTraceGenerator()
traces = [s.trace for s in generator.generate_benchmark_suite(samples_per_category=3)]

engine = PatternDetectionEngine()
report = engine.analyze_traces(traces)

print(f"Total Traces Analyzed: {report.total_traces}")
print(f"Unique Failure Clusters: {report.unique_clusters_count}")

for cluster in report.clusters:
    print(
        f"\n[Cluster {cluster.signature.category.value}] ({cluster.frequency} occurrences - {cluster.percentage:.1f}%)"
    )
    print(f"Pattern: {cluster.signature.error_pattern}")
```

