"""Trace normalization engine converting raw dictionaries and lists into canonical Trace models."""

import math
import uuid
from datetime import datetime, timezone
from typing import Any

from llm_reliability.exceptions import TraceValidationError
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


def parse_timestamp(value: Any) -> datetime | None:
    """Parse various timestamp representations into timezone-aware datetime objects."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        if value > 1e11:  # Milliseconds epoch
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            try:
                numeric_val = float(cleaned)
                return parse_timestamp(numeric_val)
            except ValueError:
                return None
    return None


def normalize_span_kind(kind_raw: Any) -> SpanKind:
    """Map string or enum values to standard SpanKind enum."""
    if isinstance(kind_raw, SpanKind):
        return kind_raw
    if not isinstance(kind_raw, str):
        return SpanKind.CUSTOM

    cleaned = kind_raw.strip().lower()
    mapping = {
        "root": SpanKind.ROOT,
        "llm": SpanKind.LLM,
        "generation": SpanKind.LLM,
        "chat": SpanKind.LLM,
        "retrieval": SpanKind.RETRIEVAL,
        "retriever": SpanKind.RETRIEVAL,
        "search": SpanKind.RETRIEVAL,
        "vector_search": SpanKind.RETRIEVAL,
        "embedding": SpanKind.EMBEDDING,
        "embed": SpanKind.EMBEDDING,
        "reranking": SpanKind.RERANKING,
        "rerank": SpanKind.RERANKING,
        "tool": SpanKind.TOOL,
        "function": SpanKind.TOOL,
        "chain": SpanKind.CHAIN,
        "pipeline": SpanKind.CHAIN,
        "agent": SpanKind.AGENT,
        "evaluation": SpanKind.EVALUATION,
        "eval": SpanKind.EVALUATION,
        "custom": SpanKind.CUSTOM,
    }
    return mapping.get(cleaned, SpanKind.CUSTOM)


def normalize_span_status(status_raw: Any, has_error: bool = False) -> SpanStatus:
    """Map string or boolean status indicators to SpanStatus enum."""
    if has_error:
        return SpanStatus.ERROR
    if isinstance(status_raw, SpanStatus):
        return status_raw
    if isinstance(status_raw, bool):
        return SpanStatus.SUCCESS if status_raw else SpanStatus.ERROR
    if not isinstance(status_raw, str):
        return SpanStatus.SUCCESS

    cleaned = status_raw.strip().lower()
    if cleaned in ("success", "ok", "passed", "200"):
        return SpanStatus.SUCCESS
    if cleaned in ("error", "failed", "failure", "err", "500"):
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def normalize_document(doc_raw: Any, default_rank: int | None = None) -> RetrievedDocument:
    """Normalize a single document chunk representation."""
    if isinstance(doc_raw, RetrievedDocument):
        return doc_raw

    if isinstance(doc_raw, str):
        return RetrievedDocument(
            doc_id=f"doc-{uuid.uuid4().hex[:8]}",
            content=doc_raw,
            rank=default_rank,
        )

    if isinstance(doc_raw, dict):
        doc_id = str(
            doc_raw.get("doc_id")
            or doc_raw.get("id")
            or doc_raw.get("_id")
            or f"doc-{uuid.uuid4().hex[:8]}"
        )
        content = str(
            doc_raw.get("content")
            or doc_raw.get("text")
            or doc_raw.get("page_content")
            or doc_raw.get("body")
            or ""
        )
        score_raw = doc_raw.get("score") if "score" in doc_raw else doc_raw.get("similarity")
        score = float(score_raw) if score_raw is not None else None
        rank_raw = doc_raw.get("rank") or default_rank
        rank = int(rank_raw) if rank_raw is not None else None

        raw_meta = doc_raw.get("metadata")
        metadata: dict[str, Any] = (
            raw_meta
            if isinstance(raw_meta, dict)
            else ({"raw_metadata": raw_meta} if raw_meta else {})
        )

        return RetrievedDocument(
            doc_id=doc_id,
            content=content,
            score=score,
            rank=rank,
            metadata=metadata,
        )

    return RetrievedDocument(
        doc_id=f"doc-{uuid.uuid4().hex[:8]}",
        content=str(doc_raw),
        rank=default_rank,
    )


def normalize_retrieval_step(data: dict[str, Any]) -> RetrievalStep:
    """Normalize retrieval step payload from raw dictionary."""
    query = str(
        data.get("query")
        or data.get("search_query")
        or data.get("input")
        or data.get("prompt")
        or ""
    )
    raw_docs = (
        data.get("documents")
        or data.get("docs")
        or data.get("chunks")
        or data.get("retrieved_documents")
        or []
    )
    if not isinstance(raw_docs, list):
        raw_docs = [raw_docs]

    documents = [normalize_document(doc, default_rank=i + 1) for i, doc in enumerate(raw_docs)]
    top_k_raw = data.get("top_k") or data.get("k") or len(documents)
    top_k = int(top_k_raw) if top_k_raw else None
    retriever_name = data.get("retriever_name") or data.get("retriever") or data.get("name")
    latency_raw = (
        data.get("latency_ms")
        or data.get("duration_ms")
        or data.get("latency")
        or data.get("duration")
    )
    latency_ms = float(latency_raw) if latency_raw is not None else None
    raw_meta = data.get("metadata")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}

    return RetrievalStep(
        query=query,
        documents=documents,
        top_k=top_k,
        retriever_name=str(retriever_name) if retriever_name else None,
        latency_ms=latency_ms,
        metadata=metadata,
    )


def normalize_token_usage(data: Any) -> TokenUsage | None:
    """Normalize token usage payload."""
    if isinstance(data, TokenUsage):
        return data
    if not isinstance(data, dict):
        return None

    prompt_tokens = int(
        data.get("prompt_tokens")
        or data.get("prompt")
        or data.get("input_tokens")
        or data.get("input")
        or 0
    )
    completion_tokens = int(
        data.get("completion_tokens")
        or data.get("completion")
        or data.get("output_tokens")
        or data.get("output")
        or 0
    )
    total_tokens = int(
        data.get("total_tokens") or data.get("total") or (prompt_tokens + completion_tokens)
    )
    cost_usd = data.get("cost_usd") or data.get("cost")
    return TokenUsage(
        prompt_tokens=max(0, prompt_tokens),
        completion_tokens=max(0, completion_tokens),
        total_tokens=max(0, total_tokens),
        cost_usd=float(cost_usd) if cost_usd is not None else None,
    )


def normalize_llm_call(data: dict[str, Any]) -> LLMCall:
    """Normalize LLM call payload from raw dictionary."""
    model = str(
        data.get("model") or data.get("model_name") or data.get("engine") or "unknown-model"
    )
    prompt = data.get("prompt") or data.get("messages") or data.get("input") or ""
    response = (
        data.get("response") or data.get("completion") or data.get("output") or data.get("text")
    )
    temperature_raw = data.get("temperature") or data.get("temp")
    temperature = float(temperature_raw) if temperature_raw is not None else None
    token_usage = normalize_token_usage(
        data.get("token_usage") or data.get("tokens") or data.get("usage")
    )
    latency_raw = (
        data.get("latency_ms")
        or data.get("duration_ms")
        or data.get("latency")
        or data.get("duration")
    )
    latency_ms = float(latency_raw) if latency_raw is not None else None
    raw_params = data.get("raw_parameters") or data.get("parameters") or {}
    params_dict: dict[str, Any] = (
        raw_params if isinstance(raw_params, dict) else {"params": raw_params}
    )

    return LLMCall(
        model=model,
        prompt=prompt if isinstance(prompt, (str, list)) else str(prompt),
        response=str(response) if response is not None else None,
        temperature=temperature,
        token_usage=token_usage,
        latency_ms=latency_ms,
        raw_parameters=params_dict,
    )


def normalize_tool_call(data: dict[str, Any]) -> ToolCall:
    """Normalize tool call invocation."""
    tool_name = str(data.get("tool_name") or data.get("name") or data.get("tool") or "unknown_tool")
    arguments = data.get("arguments") or data.get("args") or data.get("parameters") or {}
    call_id = data.get("call_id") or data.get("id")
    return ToolCall(
        tool_name=tool_name,
        arguments=arguments if isinstance(arguments, (dict, str)) else str(arguments),
        call_id=str(call_id) if call_id else None,
    )


def normalize_tool_result(data: dict[str, Any]) -> ToolResult:
    """Normalize tool execution result."""
    tool_name = str(data.get("tool_name") or data.get("name") or data.get("tool") or "unknown_tool")
    output = data.get("output") if "output" in data else data.get("result")
    error = data.get("error") or data.get("error_message")
    call_id = data.get("call_id") or data.get("id")
    status = normalize_span_status(data.get("status"), has_error=bool(error))
    latency_raw = data.get("latency_ms") or data.get("duration_ms")
    latency_ms = float(latency_raw) if latency_raw is not None else None

    return ToolResult(
        tool_name=tool_name,
        output=output,
        error=str(error) if error else None,
        call_id=str(call_id) if call_id else None,
        status=status,
        latency_ms=latency_ms,
    )


def _flatten_span_tree(raw_span: dict[str, Any], parent_id: str | None = None) -> list[Span]:
    """Recursively parse a span and its nested children into a flat list of linked Spans."""
    span_id = str(
        raw_span.get("span_id")
        or raw_span.get("id")
        or raw_span.get("spanId")
        or f"span-{uuid.uuid4().hex[:8]}"
    )
    assigned_parent_id = (
        str(
            raw_span.get("parent_span_id")
            or raw_span.get("parent_id")
            or raw_span.get("parentId")
            or parent_id
        )
        if (
            raw_span.get("parent_span_id")
            or raw_span.get("parent_id")
            or raw_span.get("parentId")
            or parent_id
        )
        else None
    )

    name = str(raw_span.get("name") or raw_span.get("operation") or "unnamed_operation")
    kind = normalize_span_kind(
        raw_span.get("kind") or raw_span.get("type") or raw_span.get("span_type")
    )
    error_msg = raw_span.get("error_message") or raw_span.get("error")
    status = normalize_span_status(raw_span.get("status"), has_error=bool(error_msg))

    start_time = parse_timestamp(raw_span.get("start_time") or raw_span.get("startTime"))
    end_time = parse_timestamp(raw_span.get("end_time") or raw_span.get("endTime"))
    duration_raw = (
        raw_span.get("duration_ms") or raw_span.get("duration") or raw_span.get("latency_ms")
    )
    duration_ms = float(duration_raw) if duration_raw is not None else None

    if duration_ms is None and start_time and end_time:
        duration_ms = max(0.0, (end_time - start_time).total_seconds() * 1000.0)

    raw_attr = raw_span.get("attributes")
    attributes: dict[str, Any] = (
        raw_attr
        if isinstance(raw_attr, dict)
        else ({"raw_attributes": raw_attr} if raw_attr else {})
    )

    events = raw_span.get("events") or []
    if not isinstance(events, list):
        events = [events]

    # Typed payloads
    llm_payload: LLMCall | None = None
    retrieval_payload: RetrievalStep | None = None
    tool_call_payload: ToolCall | None = None
    tool_result_payload: ToolResult | None = None

    if "llm_call" in raw_span and isinstance(raw_span["llm_call"], dict):
        llm_payload = normalize_llm_call(raw_span["llm_call"])
    elif kind == SpanKind.LLM:
        llm_payload = normalize_llm_call(raw_span)

    if "retrieval" in raw_span and isinstance(raw_span["retrieval"], dict):
        retrieval_payload = normalize_retrieval_step(raw_span["retrieval"])
    elif kind == SpanKind.RETRIEVAL:
        retrieval_payload = normalize_retrieval_step(raw_span)

    if "tool_call" in raw_span and isinstance(raw_span["tool_call"], dict):
        tool_call_payload = normalize_tool_call(raw_span["tool_call"])
    elif kind == SpanKind.TOOL and ("arguments" in raw_span or "args" in raw_span):
        tool_call_payload = normalize_tool_call(raw_span)

    if "tool_result" in raw_span and isinstance(raw_span["tool_result"], dict):
        tool_result_payload = normalize_tool_result(raw_span["tool_result"])
    elif kind == SpanKind.TOOL and ("output" in raw_span or "result" in raw_span):
        tool_result_payload = normalize_tool_result(raw_span)

    current_span = Span(
        span_id=span_id,
        parent_span_id=assigned_parent_id,
        name=name,
        kind=kind,
        status=status,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        attributes=attributes,
        events=events,
        error_message=str(error_msg) if error_msg else None,
        llm_call=llm_payload,
        retrieval=retrieval_payload,
        tool_call=tool_call_payload,
        tool_result=tool_result_payload,
    )

    flattened: list[Span] = [current_span]

    # Process nested children
    children = raw_span.get("children") or raw_span.get("subspans") or []
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                flattened.extend(_flatten_span_tree(child, parent_id=span_id))

    return flattened


def normalize_trace(raw_data: Any) -> Trace:
    """Convert raw dictionary, list, or existing object into a canonical Trace model.

    Raises:
        TraceValidationError: If the input data structure cannot be interpreted or normalized.
    """
    if isinstance(raw_data, Trace):
        return raw_data

    if raw_data is None:
        raise TraceValidationError("Cannot normalize empty or None trace data.")

    if not isinstance(raw_data, (dict, list)):
        raise TraceValidationError(
            f"Expected trace data as dictionary or list of spans, got {type(raw_data).__name__}"
        )

    # Case 0: OpenTelemetry / LangSmith / OpenInference dictionaries
    if isinstance(raw_data, dict) and (
        "resourceSpans" in raw_data
        or "resource_spans" in raw_data
        or "child_runs" in raw_data
        or "run_type" in raw_data
    ):
        from llm_reliability.otel.importer import OTelImporter

        return OTelImporter.import_trace(raw_data)

    # Case 1: Raw list of spans
    if isinstance(raw_data, list):
        if len(raw_data) == 0:
            raise TraceValidationError("Cannot normalize an empty list of spans.")
        spans: list[Span] = []
        for item in raw_data:
            if isinstance(item, dict):
                spans.extend(_flatten_span_tree(item))
            elif isinstance(item, Span):
                spans.append(item)

        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=f"trace-{uuid.uuid4().hex[:8]}",
            spans=spans,
        )
        return Trace(
            trace_id=run.trace_id,
            runs=[run],
            created_at=datetime.now(timezone.utc),
        )

    # Case 2: Dictionary format
    # Check if this is already formatted as a Trace structure with runs
    if "runs" in raw_data and isinstance(raw_data["runs"], list):
        trace_id = str(raw_data.get("trace_id") or f"trace-{uuid.uuid4().hex[:8]}")
        runs: list[Run] = []
        for run_dict in raw_data["runs"]:
            if isinstance(run_dict, dict):
                run_id = str(run_dict.get("run_id") or f"run-{uuid.uuid4().hex[:8]}")
                raw_spans = run_dict.get("spans") or []
                spans = []
                for s in raw_spans:
                    if isinstance(s, dict):
                        spans.extend(_flatten_span_tree(s))
                    elif isinstance(s, Span):
                        spans.append(s)

                final_resp_raw = run_dict.get("final_response")
                final_resp: FinalResponse | None = None
                if isinstance(final_resp_raw, dict):
                    final_resp = FinalResponse(
                        text=str(
                            final_resp_raw.get("text") or final_resp_raw.get("response") or ""
                        ),
                        grounded=final_resp_raw.get("grounded"),
                        confidence=final_resp_raw.get("confidence"),
                        metadata=final_resp_raw.get("metadata", {}),
                    )
                elif isinstance(final_resp_raw, str):
                    final_resp = FinalResponse(text=final_resp_raw)

                runs.append(
                    Run(
                        run_id=run_id,
                        trace_id=trace_id,
                        name=run_dict.get("name"),
                        spans=spans,
                        input_query=run_dict.get("input_query") or run_dict.get("query"),
                        final_response=final_resp,
                        metadata=run_dict.get("metadata") or {},
                        start_time=parse_timestamp(run_dict.get("start_time")),
                        end_time=parse_timestamp(run_dict.get("end_time")),
                        duration_ms=float(run_dict["duration_ms"])
                        if run_dict.get("duration_ms")
                        else None,
                    )
                )

        return Trace(
            trace_id=trace_id,
            name=raw_data.get("name"),
            runs=runs,
            metadata=raw_data.get("metadata") or {},
            created_at=parse_timestamp(raw_data.get("created_at")) or datetime.now(timezone.utc),
        )

    # Case 3: Dictionary that represents a root span with children
    if "children" in raw_data:
        spans = _flatten_span_tree(raw_data)
        trace_id = str(
            raw_data.get("trace_id") or raw_data.get("id") or f"trace-{uuid.uuid4().hex[:8]}"
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=trace_id,
            name=raw_data.get("name"),
            spans=spans,
            input_query=raw_data.get("input_query")
            or raw_data.get("query")
            or raw_data.get("prompt"),
            metadata=raw_data.get("metadata") or {},
        )
        return Trace(
            trace_id=trace_id,
            name=raw_data.get("name"),
            runs=[run],
            created_at=parse_timestamp(raw_data.get("created_at")) or datetime.now(timezone.utc),
        )

    # Case 4: Flat dictionary containing 'spans'
    trace_id = str(
        raw_data.get("trace_id") or raw_data.get("id") or f"trace-{uuid.uuid4().hex[:8]}"
    )
    if "spans" in raw_data:
        raw_spans = raw_data.get("spans") or []
        spans = []
        if isinstance(raw_spans, list):
            for s in raw_spans:
                if isinstance(s, dict):
                    spans.extend(_flatten_span_tree(s))
                elif isinstance(s, Span):
                    spans.append(s)

        final_resp_raw = (
            raw_data.get("final_response") or raw_data.get("response") or raw_data.get("answer")
        )
        final_resp = None
        if isinstance(final_resp_raw, dict):
            final_resp = FinalResponse(
                text=str(final_resp_raw.get("text") or final_resp_raw.get("response") or ""),
                grounded=final_resp_raw.get("grounded"),
                confidence=final_resp_raw.get("confidence"),
                metadata=final_resp_raw.get("metadata", {}),
            )
        elif isinstance(final_resp_raw, str):
            final_resp = FinalResponse(text=final_resp_raw)

        run = Run(
            run_id=str(raw_data.get("run_id") or f"run-{uuid.uuid4().hex[:8]}"),
            trace_id=trace_id,
            name=raw_data.get("name"),
            spans=spans,
            input_query=raw_data.get("input_query")
            or raw_data.get("query")
            or raw_data.get("prompt"),
            final_response=final_resp,
            metadata=raw_data.get("metadata") or {},
        )
        return Trace(
            trace_id=trace_id,
            name=raw_data.get("name"),
            runs=[run],
            metadata=raw_data.get("metadata") or {},
            created_at=parse_timestamp(raw_data.get("created_at")) or datetime.now(timezone.utc),
        )

    # Case 5: Simple RAG / LLM execution payload
    if (
        "retrieval" in raw_data
        or "llm_call" in raw_data
        or "query" in raw_data
        or "response" in raw_data
    ):
        query = raw_data.get("query") or raw_data.get("input_query") or raw_data.get("prompt") or ""
        spans = []

        root_span_id = f"span-root-{uuid.uuid4().hex[:6]}"
        root_span = Span(
            span_id=root_span_id,
            name="pipeline_execution",
            kind=SpanKind.ROOT,
            status=SpanStatus.SUCCESS,
            start_time=parse_timestamp(raw_data.get("start_time")),
            end_time=parse_timestamp(raw_data.get("end_time")),
        )
        spans.append(root_span)

        if "retrieval" in raw_data and isinstance(raw_data["retrieval"], dict):
            ret_step = normalize_retrieval_step(raw_data["retrieval"])
            ret_span = Span(
                span_id=f"span-retrieval-{uuid.uuid4().hex[:6]}",
                parent_span_id=root_span_id,
                name="retrieval_step",
                kind=SpanKind.RETRIEVAL,
                retrieval=ret_step,
                duration_ms=ret_step.latency_ms,
            )
            spans.append(ret_span)

        if "llm_call" in raw_data and isinstance(raw_data["llm_call"], dict):
            llm_step = normalize_llm_call(raw_data["llm_call"])
            llm_span = Span(
                span_id=f"span-llm-{uuid.uuid4().hex[:6]}",
                parent_span_id=root_span_id,
                name="llm_generation",
                kind=SpanKind.LLM,
                llm_call=llm_step,
                duration_ms=llm_step.latency_ms,
            )
            spans.append(llm_span)

        final_resp_raw = (
            raw_data.get("final_response") or raw_data.get("response") or raw_data.get("answer")
        )
        final_resp = None
        if isinstance(final_resp_raw, dict):
            final_resp = FinalResponse(
                text=str(final_resp_raw.get("text") or final_resp_raw.get("response") or ""),
                grounded=final_resp_raw.get("grounded"),
                confidence=final_resp_raw.get("confidence"),
                metadata=final_resp_raw.get("metadata", {}),
            )
        elif isinstance(final_resp_raw, str):
            final_resp = FinalResponse(text=final_resp_raw)

        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=trace_id,
            name=raw_data.get("name") or "rag_execution",
            spans=spans,
            input_query=str(query) if query else None,
            final_response=final_resp,
            metadata=raw_data.get("metadata") or {},
        )
        return Trace(
            trace_id=trace_id,
            name=raw_data.get("name") or "rag_trace",
            runs=[run],
            created_at=parse_timestamp(raw_data.get("created_at")) or datetime.now(timezone.utc),
        )

    # Fallback: treat arbitrary dictionary as a single span within a run
    single_span = _flatten_span_tree(raw_data)
    run = Run(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        trace_id=trace_id,
        spans=single_span,
    )
    return Trace(
        trace_id=trace_id,
        runs=[run],
        created_at=datetime.now(timezone.utc),
    )
