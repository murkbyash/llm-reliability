# OpenTelemetry & Distributed Tracing Ingestion

LLM Reliability Analyzer seamlessly ingests industry-standard distributed tracing formats from OpenTelemetry (OTel), OpenInference, OpenLLMetry, Arize Phoenix, and LangSmith.

---

## 1. Supported Formats

- **Standard OTel JSON (OTLP):** Ingests `resourceSpans` $\to$ `scopeSpans` $\to$ `spans` hierarchy.
- **OpenInference / OpenLLMetry Semantic Conventions:** Automatically extracts attributes like `openinference.span.kind`, `llm.model_name`, `llm.prompt_template.template`, `retrieval.documents`, `tool.parameters`.
- **LangSmith Run Trees:** Flattens recursive nested `child_runs` trees into canonical span hierarchies.
- **Arize Phoenix Traces:** Automatically maps spans into standard SpanKind representations.

---

## 2. Transparent Ingestion via `load_trace`

`load_trace` automatically detects OpenTelemetry and LangSmith formats:

```python
from llm_reliability import load_trace, diagnose

# Automatically detects and imports OTel JSON format
trace = load_trace("otel_trace_export.json")

# Directly diagnose OpenTelemetry trace
diagnosis = diagnose(trace)
print(f"Diagnosed OTel Trace: {diagnosis.primary_category.value}")
```

---

## 3. Explicit Ingestion Adapter

You can also parse raw OTel dictionaries or JSON strings directly:

```python
from llm_reliability.otel import OTelImporter

# Raw OTLP dictionary payload
otel_payload = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "traceId": "trace-101",
                            "spanId": "span-1",
                            "name": "llm_completion",
                            "attributes": {
                                "openinference.span.kind": "LLM",
                                "llm.model_name": "gpt-4o",
                                "llm.input_messages": "What is Python?",
                                "llm.output_messages": "Python is a language.",
                            },
                        }
                    ]
                }
            ]
        }
    ]
}

trace = OTelImporter.import_trace(otel_payload)
print(f"Imported Trace ID: {trace.trace_id}, Spans: {len(trace.runs[0].spans)}")
```

