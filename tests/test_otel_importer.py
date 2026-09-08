"""Tests for OpenTelemetry and Distributed Tracing Importer."""

import json
from pathlib import Path

from llm_reliability import (
    OTelAttributeParser,
    OTelImporter,
    SpanKind,
    SpanStatus,
    Trace,
    diagnose,
    import_otel_trace,
    load_trace,
)


class TestOTelImporter:
    """Deterministic validation of OpenTelemetry, OpenInference, OpenLLMetry, and LangSmith ingestion."""

    def test_otel_attribute_parser_typed_values(self) -> None:
        raw_otel_attributes = [
            {"key": "llm.model_name", "value": {"stringValue": "gpt-4o"}},
            {"key": "llm.usage.prompt_tokens", "value": {"intValue": 150}},
            {"key": "temperature", "value": {"doubleValue": 0.7}},
            {"key": "is_cached", "value": {"boolValue": True}},
            {
                "key": "tags",
                "value": {
                    "arrayValue": {"values": [{"stringValue": "prod"}, {"stringValue": "v1"}]}
                },
            },
        ]
        parsed = OTelAttributeParser.parse_attributes(raw_otel_attributes)

        assert parsed["llm.model_name"] == "gpt-4o"
        assert parsed["llm.usage.prompt_tokens"] == 150
        assert parsed["temperature"] == 0.7
        assert parsed["is_cached"] is True
        assert parsed["tags"] == ["prod", "v1"]

    def test_import_otel_resource_spans(self) -> None:
        otlp_payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
                                    "spanId": "00f067aa0ba902b7",
                                    "name": "chat_generation",
                                    "startTimeUnixNano": 1724900000000000000,
                                    "endTimeUnixNano": 1724900001500000000,
                                    "attributes": [
                                        {
                                            "key": "openinference.span.kind",
                                            "value": {"stringValue": "LLM"},
                                        },
                                        {
                                            "key": "llm.model_name",
                                            "value": {"stringValue": "claude-3-5-sonnet"},
                                        },
                                        {
                                            "key": "llm.prompts",
                                            "value": {"stringValue": "Explain quantum computing."},
                                        },
                                        {
                                            "key": "llm.completion",
                                            "value": {
                                                "stringValue": "Quantum computing uses qubits."
                                            },
                                        },
                                        {
                                            "key": "llm.usage.prompt_tokens",
                                            "value": {"intValue": 10},
                                        },
                                        {
                                            "key": "llm.usage.completion_tokens",
                                            "value": {"intValue": 25},
                                        },
                                    ],
                                    "status": {"code": 1},
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        trace = OTelImporter.import_trace(otlp_payload)
        assert isinstance(trace, Trace)
        assert trace.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert len(trace.runs) == 1
        assert len(trace.runs[0].spans) == 1

        span = trace.runs[0].spans[0]
        assert span.kind == SpanKind.LLM
        assert span.status == SpanStatus.SUCCESS
        assert span.llm_call is not None
        assert span.llm_call.model == "claude-3-5-sonnet"
        assert span.llm_call.token_usage is not None
        assert span.llm_call.token_usage.total_tokens == 35
        assert span.duration_ms == 1500.0

    def test_import_openinference_retrieval_and_tool(self) -> None:
        spans_payload = [
            {
                "traceId": "trace-oi-123",
                "spanId": "span-ret-1",
                "name": "vector_retrieval",
                "attributes": {
                    "openinference.span.kind": "RETRIEVER",
                    "input.value": "postgresql vector extensions",
                    "retrieval.documents": [
                        {
                            "id": "doc-pgvector",
                            "content": "pgvector adds vector similarity search capabilities to Postgres.",
                            "score": 0.94,
                        }
                    ],
                },
            },
            {
                "traceId": "trace-oi-123",
                "spanId": "span-tool-1",
                "parentSpanId": "span-ret-1",
                "name": "sql_executor",
                "attributes": {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "sql_executor",
                    "tool.parameters": {"query": "SELECT * FROM vectors LIMIT 5;"},
                    "tool.output": {"rows": 5},
                },
            },
        ]

        trace = import_otel_trace(spans_payload)
        assert trace.trace_id == "trace-oi-123"
        assert len(trace.runs[0].spans) == 2

        ret_span = trace.runs[0].spans[0]
        assert ret_span.kind == SpanKind.RETRIEVAL
        assert ret_span.retrieval is not None
        assert ret_span.retrieval.query == "postgresql vector extensions"
        assert len(ret_span.retrieval.documents) == 1
        assert ret_span.retrieval.documents[0].doc_id == "doc-pgvector"

        tool_span = trace.runs[0].spans[1]
        assert tool_span.kind == SpanKind.TOOL
        assert tool_span.parent_span_id == "span-ret-1"
        assert tool_span.tool_call is not None
        assert tool_span.tool_call.tool_name == "sql_executor"

    def test_import_langsmith_nested_run_tree(self) -> None:
        langsmith_payload = {
            "id": "ls-root-run-123",
            "name": "RagAgent",
            "run_type": "chain",
            "inputs": {"input": "What is Python?"},
            "outputs": {"output": "Python is a high-level programming language."},
            "child_runs": [
                {
                    "id": "ls-child-retriever",
                    "name": "VectorRetriever",
                    "run_type": "retriever",
                    "inputs": {"query": "What is Python?"},
                    "outputs": {
                        "documents": [
                            {
                                "id": "doc-py",
                                "page_content": "Python is a high-level language.",
                                "score": 0.98,
                            }
                        ]
                    },
                },
                {
                    "id": "ls-child-llm",
                    "name": "ChatOpenAI",
                    "run_type": "llm",
                    "inputs": {"prompts": "What is Python?"},
                    "outputs": {"generations": "Python is a high-level programming language."},
                },
            ],
        }

        trace = OTelImporter.import_langsmith(langsmith_payload)
        assert trace.trace_id == "ls-root-run-123"
        assert len(trace.runs[0].spans) == 3

        kinds = [s.kind for s in trace.runs[0].spans]
        assert SpanKind.RETRIEVAL in kinds
        assert SpanKind.LLM in kinds

    def test_end_to_end_diagnosis_on_otel_trace(self, tmp_path: Path) -> None:
        otel_data = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "trace-diagnose-otel",
                                    "spanId": "s1",
                                    "name": "retrieval",
                                    "attributes": {
                                        "openinference.span.kind": "RETRIEVER",
                                        "input.value": "mars gravity",
                                        "retrieval.documents": [
                                            {
                                                "id": "d1",
                                                "content": "Mars gravity is 3.72 m/s².",
                                                "score": 0.95,
                                            }
                                        ],
                                    },
                                },
                                {
                                    "traceId": "trace-diagnose-otel",
                                    "spanId": "s2",
                                    "name": "llm",
                                    "attributes": {
                                        "openinference.span.kind": "LLM",
                                        "llm.model_name": "gpt-4o",
                                        "llm.completion": "Mars gravity is 3.72 m/s².",
                                    },
                                },
                            ]
                        }
                    ]
                }
            ]
        }

        file_path = tmp_path / "otel_trace.json"
        file_path.write_text(json.dumps(otel_data), encoding="utf-8")

        # Verify load_trace automatically parses OTel JSON file
        loaded_trace = load_trace(file_path)
        assert loaded_trace.trace_id == "trace-diagnose-otel"

        # Verify diagnose accepts OTel JSON file directly
        diag = diagnose(file_path)
        assert diag.trace_id == "trace-diagnose-otel"
        assert diag.primary_category.value == "NONE"
