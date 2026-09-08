"""Tests for trace loading, ingestion, and normalization layer."""

import io
import json
from pathlib import Path

import pytest

from llm_reliability.exceptions import TraceParseError, TraceValidationError
from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.trace import Trace
from llm_reliability.normalization import (
    load_trace,
    normalize_document,
    normalize_llm_call,
    normalize_retrieval_step,
    normalize_span_kind,
    normalize_span_status,
    normalize_token_usage,
    normalize_tool_call,
    normalize_tool_result,
    normalize_trace,
    parse_timestamp,
)


class TestTimestampAndEnumParsing:
    """Tests for primitive parsing helpers."""

    def test_parse_timestamp_formats(self) -> None:
        # ISO string with UTC
        dt1 = parse_timestamp("2026-08-29T12:00:00Z")
        assert dt1 is not None
        assert dt1.year == 2026

        # ISO string with offset
        dt2 = parse_timestamp("2026-08-29T12:00:00+00:00")
        assert dt2 is not None

        # Seconds epoch
        dt3 = parse_timestamp(1700000000)
        assert dt3 is not None
        assert dt3.tzinfo is not None

        # Milliseconds epoch
        dt4 = parse_timestamp(1700000000000)
        assert dt4 is not None

        # Invalid/empty values
        assert parse_timestamp(None) is None
        assert parse_timestamp("") is None
        assert parse_timestamp("not-a-timestamp") is None

    def test_normalize_span_kind(self) -> None:
        assert normalize_span_kind("llm") == SpanKind.LLM
        assert normalize_span_kind("LLM") == SpanKind.LLM
        assert normalize_span_kind("generation") == SpanKind.LLM
        assert normalize_span_kind("retriever") == SpanKind.RETRIEVAL
        assert normalize_span_kind("vector_search") == SpanKind.RETRIEVAL
        assert normalize_span_kind("tool") == SpanKind.TOOL
        assert normalize_span_kind("function") == SpanKind.TOOL
        assert normalize_span_kind("agent") == SpanKind.AGENT
        assert normalize_span_kind("unrecognized_kind") == SpanKind.CUSTOM

    def test_normalize_span_status(self) -> None:
        assert normalize_span_status("success") == SpanStatus.SUCCESS
        assert normalize_span_status("OK") == SpanStatus.SUCCESS
        assert normalize_span_status("error") == SpanStatus.ERROR
        assert normalize_span_status("failed") == SpanStatus.ERROR
        assert normalize_span_status(True) == SpanStatus.SUCCESS
        assert normalize_span_status(False) == SpanStatus.ERROR
        assert normalize_span_status("success", has_error=True) == SpanStatus.ERROR


class TestComponentNormalizers:
    """Tests for granular payload normalization functions."""

    def test_normalize_document(self) -> None:
        # String document
        doc1 = normalize_document("Plain text chunk", default_rank=1)
        assert doc1.content == "Plain text chunk"
        assert doc1.rank == 1

        # Dict document with aliases
        doc2 = normalize_document(
            {"id": "doc-99", "text": "Aliased text", "similarity": 0.92, "rank": 2}
        )
        assert doc2.doc_id == "doc-99"
        assert doc2.content == "Aliased text"
        assert doc2.score == 0.92
        assert doc2.rank == 2

    def test_normalize_retrieval_step(self) -> None:
        raw = {
            "search_query": "Explain LLM observability",
            "docs": [
                {"id": "d1", "page_content": "Observability provides telemetry."},
                {"id": "d2", "page_content": "Traces capture step pipelines."},
            ],
            "k": 2,
            "duration": 34.5,
        }
        step = normalize_retrieval_step(raw)
        assert step.query == "Explain LLM observability"
        assert len(step.documents) == 2
        assert step.top_k == 2
        assert step.latency_ms == 34.5

    def test_normalize_llm_call(self) -> None:
        raw = {
            "model_name": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hi"}],
            "completion": "Hello! How can I help you?",
            "usage": {"input_tokens": 10, "output_tokens": 15, "cost": 0.0005},
            "duration_ms": 120.0,
        }
        call = normalize_llm_call(raw)
        assert call.model == "claude-3-5-sonnet"
        assert call.response == "Hello! How can I help you?"
        assert call.token_usage is not None
        assert call.token_usage.prompt_tokens == 10
        assert call.token_usage.completion_tokens == 15
        assert call.token_usage.cost_usd == 0.0005

    def test_normalize_token_usage_edge_cases(self) -> None:
        assert normalize_token_usage(None) is None
        assert normalize_token_usage("invalid") is None
        usage = normalize_token_usage({"prompt": 5, "completion": 10})
        assert usage is not None
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 15

    def test_normalize_tool_call_and_result(self) -> None:
        tool_call = normalize_tool_call({"name": "calculator", "args": {"expr": "2 + 2"}})
        assert tool_call.tool_name == "calculator"
        assert tool_call.arguments == {"expr": "2 + 2"}

        tool_res = normalize_tool_result({"name": "calculator", "result": 4, "latency_ms": 5.0})
        assert tool_res.tool_name == "calculator"
        assert tool_res.output == 4
        assert tool_res.status == SpanStatus.SUCCESS


class TestTraceNormalization:
    """Tests for normalize_trace covering various input shapes."""

    def test_normalize_already_valid_trace(self) -> None:
        trace = Trace(trace_id="t-100")
        assert normalize_trace(trace) is trace

    def test_normalize_canonical_dict(self) -> None:
        raw = {
            "trace_id": "trace-canonical-1",
            "name": "qa_flow",
            "runs": [
                {
                    "run_id": "run-1",
                    "input_query": "What is RAG?",
                    "spans": [
                        {
                            "span_id": "s1",
                            "name": "search",
                            "kind": "retrieval",
                            "retrieval": {
                                "query": "What is RAG?",
                                "documents": [{"content": "Retrieval Augmented Generation"}],
                            },
                        }
                    ],
                    "final_response": {"text": "RAG is retrieval augmented generation."},
                }
            ],
        }
        trace = normalize_trace(raw)
        assert trace.trace_id == "trace-canonical-1"
        assert len(trace.runs) == 1
        assert trace.runs[0].spans[0].kind == SpanKind.RETRIEVAL
        assert trace.runs[0].final_response is not None
        assert trace.runs[0].final_response.text == "RAG is retrieval augmented generation."

    def test_normalize_flat_spans_list(self) -> None:
        spans_list = [
            {"id": "span-1", "operation": "retrieve", "type": "retrieval"},
            {"id": "span-2", "operation": "generate", "type": "llm", "parent_id": "span-1"},
        ]
        trace = normalize_trace(spans_list)
        assert len(trace.runs) == 1
        assert len(trace.runs[0].spans) == 2
        assert trace.runs[0].spans[0].span_id == "span-1"
        assert trace.runs[0].spans[1].parent_span_id == "span-1"

    def test_normalize_nested_span_tree(self) -> None:
        tree = {
            "id": "root-span",
            "name": "agent_execution",
            "type": "agent",
            "children": [
                {
                    "id": "child-retrieval",
                    "name": "search_docs",
                    "type": "retrieval",
                    "retrieval": {"query": "Find facts", "docs": [{"content": "Fact 1"}]},
                },
                {
                    "id": "child-llm",
                    "name": "generate_answer",
                    "type": "llm",
                    "llm_call": {"model": "gpt-4o", "prompt": "Hi", "response": "Hello"},
                    "children": [
                        {
                            "id": "grandchild-tool",
                            "name": "lookup_user",
                            "type": "tool",
                            "tool_call": {"tool_name": "lookup", "arguments": {"user": "Alice"}},
                        }
                    ],
                },
            ],
        }
        trace = normalize_trace(tree)
        assert len(trace.runs) == 1
        spans = trace.runs[0].spans
        assert len(spans) == 4
        # Verify parent linkage
        root = spans[0]
        assert root.span_id == "root-span"
        child1 = spans[1]
        assert child1.parent_span_id == "root-span"
        child2 = spans[2]
        assert child2.parent_span_id == "root-span"
        grandchild = spans[3]
        assert grandchild.parent_span_id == "child-llm"

    def test_normalize_simple_rag_dictionary(self) -> None:
        raw_rag = {
            "query": "How do embeddings work?",
            "retrieval": {
                "query": "How do embeddings work?",
                "documents": [
                    {"content": "Embeddings represent words as dense vectors.", "score": 0.95}
                ],
            },
            "llm_call": {
                "model": "mistral-7b",
                "prompt": "Explain embeddings.",
                "response": "Embeddings map text to vector space.",
            },
            "response": "Embeddings map text to vector space.",
        }
        trace = normalize_trace(raw_rag)
        assert len(trace.runs) == 1
        run = trace.runs[0]
        assert run.input_query == "How do embeddings work?"
        assert run.final_response is not None
        assert run.final_response.text == "Embeddings map text to vector space."
        assert len(run.spans) == 3  # Root + Retrieval + LLM

    def test_normalize_missing_fields_and_defaults(self) -> None:
        raw = {"random_key": "some_value"}
        trace = normalize_trace(raw)
        assert trace.trace_id.startswith("trace-")
        assert len(trace.runs) == 1
        assert len(trace.runs[0].spans) == 1

    def test_normalize_invalid_types_raise_validation_error(self) -> None:
        with pytest.raises(TraceValidationError):
            normalize_trace(None)

        with pytest.raises(TraceValidationError):
            normalize_trace(12345)

        with pytest.raises(TraceValidationError):
            normalize_trace([])


class TestTraceLoader:
    """Tests for load_trace supporting files, strings, streams, and dicts."""

    def test_load_trace_from_dict(self) -> None:
        data = {"trace_id": "t-dict-1", "spans": [{"name": "step1"}]}
        trace = load_trace(data)
        assert trace.trace_id == "t-dict-1"

    def test_load_trace_from_json_string(self) -> None:
        json_str = json.dumps({"trace_id": "t-json-str", "spans": [{"name": "step1"}]})
        trace = load_trace(json_str)
        assert trace.trace_id == "t-json-str"

    def test_load_trace_from_file_path_str(self, tmp_path: Path) -> None:
        trace_file = tmp_path / "sample_trace.json"
        trace_file.write_text(
            json.dumps({"trace_id": "t-file-1", "query": "hello", "response": "hi"}),
            encoding="utf-8",
        )
        trace = load_trace(str(trace_file))
        assert trace.trace_id == "t-file-1"

    def test_load_trace_from_path_object(self, tmp_path: Path) -> None:
        trace_file = tmp_path / "sample_trace.json"
        trace_file.write_text(
            json.dumps({"trace_id": "t-file-2", "spans": []}),
            encoding="utf-8",
        )
        trace = load_trace(trace_file)
        assert trace.trace_id == "t-file-2"

    def test_load_trace_from_stream(self) -> None:
        stream = io.StringIO(json.dumps({"trace_id": "t-stream-1", "spans": []}))
        trace = load_trace(stream)
        assert trace.trace_id == "t-stream-1"

    def test_load_trace_missing_file_raises_parse_error(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(TraceParseError, match="Trace file not found"):
            load_trace(non_existent)

    def test_load_trace_malformed_json_raises_parse_error(self, tmp_path: Path) -> None:
        bad_json_file = tmp_path / "bad.json"
        bad_json_file.write_text("{ unquoted_key: 123 ", encoding="utf-8")
        with pytest.raises(TraceParseError, match="Invalid JSON syntax"):
            load_trace(bad_json_file)

    def test_load_trace_empty_raises_validation_error(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("   \n", encoding="utf-8")
        with pytest.raises(TraceValidationError, match="empty"):
            load_trace(empty_file)

        with pytest.raises(TraceValidationError):
            load_trace("")

        with pytest.raises(TraceValidationError):
            load_trace(None)
