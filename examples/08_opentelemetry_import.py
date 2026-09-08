"""Example 08: Ingesting and Diagnosing Distributed OpenTelemetry Traces."""

from llm_reliability import DiagnosticEngine, OTelImporter


def main() -> None:
    print("=" * 70)
    print("Example 08: OpenTelemetry OTLP JSON Import & Diagnosis")
    print("=" * 70)

    # 1. Standard OpenTelemetry / OpenInference JSON payload
    otel_payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace-otel-distributed-01",
                                "spanId": "span-root-01",
                                "name": "rag_pipeline_entrypoint",
                                "attributes": {
                                    "openinference.span.kind": "CHAIN",
                                },
                            },
                            {
                                "traceId": "trace-otel-distributed-01",
                                "spanId": "span-ret-02",
                                "parentSpanId": "span-root-01",
                                "name": "qdrant_vector_retriever",
                                "attributes": {
                                    "openinference.span.kind": "RETRIEVER",
                                    "input.value": "What is quantum teleportation?",
                                    "retrieval.documents": [],  # Empty retrieval failure
                                },
                            },
                        ]
                    }
                ]
            }
        ]
    }

    # 2. Import OTel payload into canonical Trace model
    trace = OTelImporter.import_trace(otel_payload)
    print(f"\n[1] Successfully Imported OTel Trace: {trace.trace_id}")
    print(f"    * Spans Captured: {len(trace.runs[0].spans)}")

    # 3. Diagnose imported trace
    engine = DiagnosticEngine()
    diagnosis = engine.diagnose(trace)

    print("\n[2] Diagnosis of Distributed OTel Trace:")
    print(f"    * Primary Root Cause : {diagnosis.primary_category.value}")
    print(f"    * Confidence         : {diagnosis.confidence * 100:.1f}%")
    print(f"    * Severity           : {diagnosis.severity.value.upper()}")
    print(f"    * Summary            : {diagnosis.summary}")
    print("=" * 70)


if __name__ == "__main__":
    main()
