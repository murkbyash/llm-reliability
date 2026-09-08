"""OpenTelemetry and distributed trace ingestion adapters with semantic convention mappings."""

import json
import uuid
from pathlib import Path
from typing import Any

from llm_reliability.exceptions import TraceParseError
from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    LLMCall,
    RetrievalStep,
    RetrievedDocument,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span, Trace
from llm_reliability.normalization.normalizer import parse_timestamp
from llm_reliability.otel.models import OTelAttributeParser


class OTelImporter:
    """Imports OpenTelemetry, OpenInference, OpenLLMetry, Arize Phoenix, and LangSmith traces into canonical format."""

    @classmethod
    def import_trace(cls, source: dict[str, Any] | list[Any] | str | Path) -> Trace:
        """Automatically detect payload format and convert into a canonical Trace hierarchy."""
        data = cls._load_raw(source)

        if isinstance(data, dict):
            # Check for standard OTLP format
            if "resourceSpans" in data or "resource_spans" in data:
                return cls.import_otel_json(data)
            # Check for LangSmith format
            if "run_type" in data or (
                "inputs" in data and "outputs" in data and "child_runs" in data
            ):
                return cls.import_langsmith(data)
            # Check for OpenInference / Arize Phoenix span dictionary
            if "attributes" in data or "openinference" in str(data):
                return cls.import_openinference(data)

        if isinstance(data, list):
            return cls.import_span_list(data)

        if isinstance(data, dict) and "spans" in data:
            return cls.import_span_list(
                data["spans"], trace_id=data.get("trace_id", data.get("traceId"))
            )

        # Fallback to general span list or single span dict
        if isinstance(data, dict):
            return cls.import_span_list([data])

        raise TraceParseError(
            f"Unrecognized OpenTelemetry trace data format: {type(data).__name__}"
        )

    @classmethod
    def import_otel_json(cls, data: dict[str, Any]) -> Trace:
        """Parse standard OTLP JSON export structure (resourceSpans -> scopeSpans -> spans)."""
        resource_spans = data.get("resourceSpans", data.get("resource_spans", []))
        extracted_spans: list[dict[str, Any]] = []
        global_trace_id: str | None = None

        for r_span in resource_spans:
            scope_spans = r_span.get("scopeSpans", r_span.get("scope_spans", []))
            for s_span in scope_spans:
                spans = s_span.get("spans", [])
                for sp in spans:
                    extracted_spans.append(sp)
                    if not global_trace_id:
                        global_trace_id = sp.get("traceId", sp.get("trace_id"))

        return cls.import_span_list(extracted_spans, trace_id=global_trace_id)

    @classmethod
    def import_openinference(cls, data: dict[str, Any] | list[Any]) -> Trace:
        """Parse Arize Phoenix / OpenInference format with semantic conventions."""
        if isinstance(data, dict) and "spans" in data:
            return cls.import_span_list(
                data["spans"], trace_id=data.get("trace_id", data.get("traceId"))
            )
        if isinstance(data, list):
            return cls.import_span_list(data)
        return cls.import_span_list([data])

    @classmethod
    def import_langsmith(cls, data: dict[str, Any]) -> Trace:
        """Parse LangSmith run tree export JSON."""
        trace_id = str(data.get("id", data.get("trace_id", f"trace-{uuid.uuid4().hex[:8]}")))
        flat_runs: list[dict[str, Any]] = []

        def _flatten_langsmith_run(node: dict[str, Any], parent_id: str | None = None) -> None:
            node_copy = dict(node)
            node_copy["parent_run_id"] = parent_id
            flat_runs.append(node_copy)
            for child in node.get("child_runs", []):
                _flatten_langsmith_run(child, parent_id=str(node.get("id", "")))

        _flatten_langsmith_run(data)

        canonical_spans: list[Span] = []
        final_answer: str | None = None

        for r in flat_runs:
            span_id = str(r.get("id", f"span-{uuid.uuid4().hex[:6]}"))
            parent_id = r.get("parent_run_id") or None
            run_type = str(r.get("run_type", "")).lower()
            name = str(r.get("name", "langsmith_span"))

            inputs = r.get("inputs", {})
            outputs = r.get("outputs", {})
            error = r.get("error")

            kind = SpanKind.CUSTOM
            llm_call: LLMCall | None = None
            retrieval_step: RetrievalStep | None = None
            tool_call: ToolCall | None = None
            tool_result: ToolResult | None = None

            if run_type in ("llm", "chat_model"):
                kind = SpanKind.LLM
                prompt_text = str(
                    inputs.get("prompts", inputs.get("input", inputs.get("messages", "")))
                )
                resp_text = str(
                    outputs.get("generations", outputs.get("output", outputs.get("text", "")))
                )
                llm_call = LLMCall(
                    model=str(r.get("extra", {}).get("metadata", {}).get("model", "unknown")),
                    prompt=prompt_text,
                    response=resp_text,
                )
                if not final_answer and resp_text:
                    final_answer = resp_text
            elif run_type in ("retriever", "retrieval"):
                kind = SpanKind.RETRIEVAL
                query = str(inputs.get("query", inputs.get("input", "")))
                docs = cls._extract_documents_from_payload(outputs)
                retrieval_step = RetrievalStep(query=query, documents=docs)
            elif run_type in ("tool", "tool_call"):
                kind = SpanKind.TOOL
                tool_call = ToolCall(tool_name=name, arguments=inputs)
                tool_result = ToolResult(
                    tool_name=name, output=outputs, error=str(error) if error else None
                )

            status = SpanStatus.ERROR if error else SpanStatus.SUCCESS

            canonical_spans.append(
                Span(
                    span_id=span_id,
                    parent_span_id=parent_id,
                    name=name,
                    kind=kind,
                    status=status,
                    error_message=str(error) if error else None,
                    llm_call=llm_call,
                    retrieval=retrieval_step,
                    tool_call=tool_call,
                    tool_result=tool_result,
                )
            )

        run = Run(
            run_id=f"run-{trace_id}",
            trace_id=trace_id,
            spans=canonical_spans,
            final_response=FinalResponse(text=final_answer) if final_answer else None,
        )
        return Trace(trace_id=trace_id, runs=[run])

    @classmethod
    def import_span_list(
        cls, raw_spans: list[dict[str, Any]], trace_id: str | None = None
    ) -> Trace:
        """Convert a list of OpenTelemetry / OpenInference span dictionaries into a canonical Trace."""
        if not raw_spans:
            t_id = trace_id or f"trace-empty-{uuid.uuid4().hex[:8]}"
            return Trace(trace_id=t_id, runs=[Run(run_id=f"run-{t_id}", trace_id=t_id)])

        t_id = (
            trace_id
            or str(raw_spans[0].get("traceId", raw_spans[0].get("trace_id", "")))
            or f"trace-{uuid.uuid4().hex[:8]}"
        )

        canonical_spans: list[Span] = []
        final_answer: str | None = None

        for sp in raw_spans:
            span_id = str(sp.get("spanId", sp.get("span_id", f"span-{uuid.uuid4().hex[:6]}")))
            parent_id = sp.get("parentSpanId", sp.get("parent_span_id"))
            if parent_id:
                parent_id = str(parent_id)

            name = str(sp.get("name", "otel_span"))
            attrs = OTelAttributeParser.parse_attributes(sp.get("attributes", {}))

            # Duration and Timestamps
            start_time = cls._parse_otel_time(sp.get("startTimeUnixNano", sp.get("start_time")))
            end_time = cls._parse_otel_time(sp.get("endTimeUnixNano", sp.get("end_time")))
            duration_ms: float | None = None
            if start_time and end_time:
                duration_ms = max(0.0, (end_time - start_time).total_seconds() * 1000.0)

            # Determine Span Status
            status_obj = sp.get("status", {})
            status_code = (
                status_obj.get("code") if isinstance(status_obj, dict) else sp.get("status")
            )
            error_msg: str | None = None
            if isinstance(status_obj, dict) and status_obj.get("message"):
                error_msg = str(status_obj["message"])

            status = SpanStatus.SUCCESS
            if status_code in (2, "ERROR", "error", SpanStatus.ERROR) or error_msg:
                status = SpanStatus.ERROR

            # Determine Span Kind via OpenInference / OpenLLMetry semantic conventions
            oi_kind = str(
                attrs.get("openinference.span.kind", attrs.get("traceloop.span.kind", ""))
            ).upper()
            kind = SpanKind.CUSTOM
            if oi_kind == "CHAIN" or "chain" in name.lower():
                kind = SpanKind.CHAIN
            elif oi_kind == "AGENT" or "agent" in name.lower():
                kind = SpanKind.AGENT
            elif oi_kind == "ROOT" or "root" in name.lower():
                kind = SpanKind.ROOT

            llm_call: LLMCall | None = None
            retrieval_step: RetrievalStep | None = None
            tool_call: ToolCall | None = None
            tool_result: ToolResult | None = None

            if oi_kind == "LLM" or "llm" in name.lower() or "gen_ai" in str(attrs):
                kind = SpanKind.LLM
                model_name = str(
                    attrs.get(
                        "llm.model_name",
                        attrs.get("gen_ai.request.model", attrs.get("model", "unknown")),
                    )
                )
                prompt_text = str(
                    attrs.get(
                        "llm.prompts",
                        attrs.get(
                            "gen_ai.prompt",
                            attrs.get("input.value", attrs.get("llm.input_messages", "")),
                        ),
                    )
                )
                resp_text = str(
                    attrs.get(
                        "llm.completion",
                        attrs.get(
                            "gen_ai.completion",
                            attrs.get("output.value", attrs.get("llm.output_messages", "")),
                        ),
                    )
                )

                prompt_tokens = int(
                    attrs.get("llm.usage.prompt_tokens", attrs.get("gen_ai.usage.input_tokens", 0))
                    or 0
                )
                completion_tokens = int(
                    attrs.get(
                        "llm.usage.completion_tokens",
                        attrs.get("gen_ai.usage.output_tokens", 0),
                    )
                    or 0
                )
                token_usage = (
                    TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    if (prompt_tokens or completion_tokens)
                    else None
                )

                llm_call = LLMCall(
                    model=model_name,
                    prompt=prompt_text,
                    response=resp_text,
                    token_usage=token_usage,
                )
                if not final_answer and resp_text:
                    final_answer = resp_text

            elif oi_kind in ("RETRIEVER", "RETRIEVAL") or "retriev" in name.lower():
                kind = SpanKind.RETRIEVAL
                query = str(
                    attrs.get("input.value", attrs.get("retrieval.query", attrs.get("query", "")))
                )
                docs = cls._extract_documents_from_attributes(attrs)
                retrieval_step = RetrievalStep(query=query, documents=docs)

            elif oi_kind == "TOOL" or "tool" in name.lower():
                kind = SpanKind.TOOL
                tool_name = str(attrs.get("tool.name", attrs.get("name", name)))
                tool_args = attrs.get("tool.parameters", attrs.get("input.value", {}))
                tool_out = attrs.get("tool.output", attrs.get("output.value", {}))
                tool_call = ToolCall(tool_name=tool_name, arguments=tool_args)
                tool_result = ToolResult(
                    tool_name=tool_name,
                    output=tool_out,
                    status=status,
                    error=error_msg,
                    latency_ms=duration_ms,
                )

            canonical_spans.append(
                Span(
                    span_id=span_id,
                    parent_span_id=parent_id,
                    name=name,
                    kind=kind,
                    status=status,
                    duration_ms=duration_ms,
                    start_time=start_time,
                    end_time=end_time,
                    error_message=error_msg,
                    llm_call=llm_call,
                    retrieval=retrieval_step,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    attributes=attrs,
                )
            )

        run = Run(
            run_id=f"run-{t_id}",
            trace_id=t_id,
            spans=canonical_spans,
            final_response=FinalResponse(text=final_answer) if final_answer else None,
        )
        return Trace(trace_id=t_id, runs=[run])

    @classmethod
    def _extract_documents_from_attributes(cls, attrs: dict[str, Any]) -> list[RetrievedDocument]:
        """Extract retrieved documents from OpenInference / OpenLLMetry document attribute lists."""
        docs: list[RetrievedDocument] = []
        raw_docs = attrs.get("retrieval.documents", attrs.get("documents", []))
        if isinstance(raw_docs, list):
            for i, d in enumerate(raw_docs):
                if isinstance(d, dict):
                    doc_id = str(d.get("id", d.get("document.id", f"doc-{i + 1}")))
                    content = str(d.get("content", d.get("document.content", d.get("text", ""))))
                    score = d.get("score", d.get("document.score", 1.0))
                    score_val = float(score) if score is not None else 1.0
                    docs.append(RetrievedDocument(doc_id=doc_id, content=content, score=score_val))
                elif isinstance(d, str):
                    docs.append(RetrievedDocument(doc_id=f"doc-{i + 1}", content=d, score=1.0))
        return docs

    @classmethod
    def _extract_documents_from_payload(cls, payload: Any) -> list[RetrievedDocument]:
        """Extract retrieved documents from LangSmith output dictionary."""
        docs: list[RetrievedDocument] = []
        if isinstance(payload, dict):
            raw_docs = payload.get("documents", payload.get("output", []))
            if isinstance(raw_docs, list):
                for i, d in enumerate(raw_docs):
                    if isinstance(d, dict):
                        content = str(d.get("page_content", d.get("content", d.get("text", ""))))
                        score = d.get("score", 1.0)
                        docs.append(
                            RetrievedDocument(
                                doc_id=str(d.get("id", f"doc-{i + 1}")),
                                content=content,
                                score=float(score) if score is not None else 1.0,
                            )
                        )
        return docs

    @classmethod
    def _parse_otel_time(cls, raw_val: Any) -> Any:
        """Parse nanosecond timestamps or ISO strings."""
        if not raw_val:
            return None
        if isinstance(raw_val, (int, float)):
            # If timestamp is in nanoseconds (> 1e16), convert to seconds
            if raw_val > 1e16:
                return parse_timestamp(raw_val / 1e9)
            if raw_val > 1e11:
                return parse_timestamp(raw_val / 1e3)
            return parse_timestamp(raw_val)
        return parse_timestamp(raw_val)

    @classmethod
    def _load_raw(cls, source: Any) -> Any:
        """Safely parse input string, file path, or dictionary."""
        if isinstance(source, (dict, list)):
            return source
        if isinstance(source, Path):
            return json.loads(source.read_text(encoding="utf-8"))
        if isinstance(source, str):
            p = Path(source)
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
            return json.loads(source)
        raise TraceParseError(f"Unsupported source type for OTel importer: {type(source).__name__}")


def import_otel_trace(source: dict[str, Any] | list[Any] | str | Path) -> Trace:
    """Convenience helper function to import OpenTelemetry / OpenInference / LangSmith traces."""
    return OTelImporter.import_trace(source)
