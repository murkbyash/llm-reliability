# Plugins & Extensibility SDK

`llm-reliability` provides an extensibility framework allowing teams to register custom analyzers, telemetry exporters, and middleware hooks without modifying core package internals.

## Plugin Types

1. **`BaseAnalyzerPlugin`**: Implement domain-specific root cause analyzers.
2. **`BaseExporterPlugin`**: Implement custom telemetry export destinations (e.g. S3, Kafka, Datadog).
3. **`BaseMiddlewarePlugin`**: Implement pre/post hooks surrounding trace processing.

---

## Creating a Custom Analyzer Plugin

```python
from llm_reliability import (
    BaseAnalyzerPlugin,
    Evidence,
    Metric,
    PluginMetadata,
    Trace,
    register_plugin,
)


class HallucinatedQuoteAnalyzer(BaseAnalyzerPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="quote_analyzer",
            version="1.0.0",
            description="Checks for fabricated direct quotations.",
        )

    def analyze(self, trace: Trace):
        failures = []
        evidence = []
        metrics = []

        for run in trace.runs:
            for span in run.spans:
                if span.llm_call and span.llm_call.response:
                    if '"' in span.llm_call.response:
                        metrics.append(Metric(name="has_quotes", value=1.0, unit="bool"))

        return failures, evidence, metrics


# Register plugin globally
register_plugin(HallucinatedQuoteAnalyzer())
```
