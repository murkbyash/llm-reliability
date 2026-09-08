"""Lightweight, zero-overhead live tracing SDK, context managers, and function decorators."""

import functools
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar, cast

from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    LLMCall,
    RetrievalStep,
    RetrievedDocument,
    ToolCall,
    ToolResult,
)
from llm_reliability.models.trace import Run, Span, Trace
from llm_reliability.tracing.context import (
    get_current_context,
    get_current_parent_span_id,
    reset_current_context,
    reset_current_parent_span_id,
    set_current_context,
    set_current_parent_span_id,
)

F = TypeVar("F", bound=Callable[..., Any])


class ActiveSpan:
    """Represents a span currently being recorded."""

    def __init__(
        self,
        span_id: str,
        name: str,
        kind: SpanKind,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.span_id = span_id
        self.name = name
        self.kind = kind
        self.parent_span_id = parent_span_id
        self.attributes = attributes or {}
        self.start_time = datetime.now(timezone.utc)
        self.start_ticks = time.perf_counter()
        self.end_time: datetime | None = None
        self.duration_ms: float | None = None
        self.status = SpanStatus.SUCCESS
        self.error_message: str | None = None
        self.llm_call: LLMCall | None = None
        self.retrieval: RetrievalStep | None = None
        self.tool_call: ToolCall | None = None
        self.tool_result: ToolResult | None = None

    def finish(
        self, status: SpanStatus = SpanStatus.SUCCESS, error: Exception | str | None = None
    ) -> Span:
        """Complete the span recording and produce a canonical Span model."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = max(0.0, (time.perf_counter() - self.start_ticks) * 1000.0)
        self.status = status
        if error:
            self.error_message = str(error)
            self.status = SpanStatus.ERROR

        return Span(
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_ms=self.duration_ms,
            error_message=self.error_message,
            llm_call=self.llm_call,
            retrieval=self.retrieval,
            tool_call=self.tool_call,
            tool_result=self.tool_result,
            attributes=self.attributes,
        )


class SpanContextManager:
    """Context manager for managing the lifecycle of an active span."""

    def __init__(self, span: ActiveSpan, collector: list[Span]) -> None:
        self.span = span
        self.collector = collector
        self._token: Any = None

    def __enter__(self) -> ActiveSpan:
        self._token = set_current_parent_span_id(self.span.span_id)
        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        if self._token:
            reset_current_parent_span_id(self._token)

        if exc_val is not None:
            span_model = self.span.finish(status=SpanStatus.ERROR, error=exc_val)
        else:
            span_model = self.span.finish(status=self.span.status, error=self.span.error_message)

        self.collector.append(span_model)
        return False  # Propagate exceptions


class TraceContextManager:
    """Context manager for creating and capturing a complete trace."""

    def __init__(self, trace_id: str, run_id: str) -> None:
        self.trace_id = trace_id
        self.run_id = run_id
        self.spans: list[Span] = []
        self.final_response: str | None = None
        self._ctx_token: Any = None
        self._parent_token: Any = None
        self.trace: Trace | None = None

    def __enter__(self) -> "TraceContextManager":
        ctx = {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "collector": self.spans,
            "manager": self,
        }
        self._ctx_token = set_current_context(ctx)
        self._parent_token = set_current_parent_span_id(None)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        if self._parent_token:
            reset_current_parent_span_id(self._parent_token)
        if self._ctx_token:
            reset_current_context(self._ctx_token)

        run = Run(
            run_id=self.run_id,
            trace_id=self.trace_id,
            spans=list(self.spans),
            final_response=FinalResponse(text=self.final_response) if self.final_response else None,
        )
        self.trace = Trace(trace_id=self.trace_id, runs=[run])
        return False

    def set_final_response(self, text: str) -> None:
        """Attach final response text to the trace."""
        self.final_response = text


class Tracer:
    """Main live tracer for starting traces and emitting spans."""

    def __init__(self) -> None:
        self._completed_traces: list[Trace] = []

    def start_trace(
        self, trace_id: str | None = None, run_id: str | None = None
    ) -> TraceContextManager:
        """Start a new trace context."""
        t_id = trace_id or f"trace-{uuid.uuid4().hex[:8]}"
        r_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        mgr = TraceContextManager(trace_id=t_id, run_id=r_id)
        return mgr

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanContextManager:
        """Start an active span within the current trace context."""
        ctx = get_current_context()
        collector: list[Span] = ctx["collector"] if ctx and "collector" in ctx else []
        s_id = span_id or f"span-{uuid.uuid4().hex[:6]}"
        p_id = parent_span_id or get_current_parent_span_id()
        active = ActiveSpan(
            span_id=s_id, name=name, kind=kind, parent_span_id=p_id, attributes=attributes
        )
        return SpanContextManager(active, collector)

    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        attributes: dict[str, Any] | None = None,
    ) -> SpanContextManager:
        """Convenience alias for start_span."""
        return self.start_span(name=name, kind=kind, attributes=attributes)

    def trace_llm(
        self,
        model: str = "unknown",
        prompt: str = "",
        name: str = "llm_call",
    ) -> SpanContextManager:
        """Start an LLM generation span."""
        mgr = self.start_span(name=name, kind=SpanKind.LLM)
        mgr.span.llm_call = LLMCall(model=model, prompt=prompt)
        return mgr

    def trace_tool(
        self,
        tool_name: str,
        arguments: Any = None,
        name: str | None = None,
    ) -> SpanContextManager:
        """Start a tool execution span."""
        s_name = name or f"tool:{tool_name}"
        mgr = self.start_span(name=s_name, kind=SpanKind.TOOL)
        mgr.span.tool_call = ToolCall(tool_name=tool_name, arguments=arguments or {})
        return mgr

    def trace_retrieval(
        self,
        query: str,
        documents: list[RetrievedDocument] | None = None,
        name: str = "retrieval",
    ) -> SpanContextManager:
        """Start a document retrieval span."""
        mgr = self.start_span(name=name, kind=SpanKind.RETRIEVAL)
        mgr.span.retrieval = RetrievalStep(query=query, documents=documents or [])
        return mgr


# Global Default Tracer Instance
_default_tracer = Tracer()


def get_tracer() -> Tracer:
    """Retrieve the global default Tracer instance."""
    return _default_tracer


def trace(name: str | None = None, kind: SpanKind = SpanKind.CUSTOM) -> Callable[[F], F]:
    """Function decorator for capturing execution as a trace span."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = str(name or getattr(func, "__name__", "traced_func"))
            with _default_tracer.start_span(name=span_name, kind=kind):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def trace_llm(model: str = "unknown", name: str | None = None) -> Callable[[F], F]:
    """Function decorator for wrapping an LLM generation call."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = str(name or getattr(func, "__name__", "llm_call"))
            with _default_tracer.start_span(name=span_name, kind=SpanKind.LLM) as active:
                prompt_arg = (
                    str(args[0]) if args else str(kwargs.get("prompt", kwargs.get("input", "")))
                )
                active.llm_call = LLMCall(model=model, prompt=prompt_arg)
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    active.llm_call.response = result
                elif isinstance(result, dict) and "text" in result:
                    active.llm_call.response = str(result["text"])
                return result

        return cast(F, wrapper)

    return decorator


def trace_tool(tool_name: str | None = None) -> Callable[[F], F]:
    """Function decorator for wrapping tool executions."""

    def decorator(func: F) -> F:
        t_name = str(tool_name or getattr(func, "__name__", "tool_func"))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            args_payload = kwargs if kwargs else ({"args": list(args)} if args else {})
            with _default_tracer.start_span(name=f"tool:{t_name}", kind=SpanKind.TOOL) as active:
                active.tool_call = ToolCall(tool_name=t_name, arguments=args_payload)
                try:
                    result = func(*args, **kwargs)
                    active.tool_result = ToolResult(
                        tool_name=t_name,
                        output=result
                        if isinstance(result, (dict, list, str, int, float, bool))
                        else str(result),
                        status=SpanStatus.SUCCESS,
                    )
                    return result
                except Exception as err:
                    active.tool_result = ToolResult(
                        tool_name=t_name,
                        status=SpanStatus.ERROR,
                        error=str(err),
                    )
                    raise

        return cast(F, wrapper)

    return decorator


def trace_retrieval(
    query_param: str = "query",
) -> Callable[[F], F]:
    """Function decorator for wrapping retrieval and vector search functions."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            query = str(kwargs.get(query_param, args[0] if args else ""))
            with _default_tracer.start_span(name="retrieval", kind=SpanKind.RETRIEVAL) as active:
                active.retrieval = RetrievalStep(query=query, documents=[])
                result = func(*args, **kwargs)
                docs: list[RetrievedDocument] = []
                if isinstance(result, list):
                    for i, item in enumerate(result):
                        if isinstance(item, RetrievedDocument):
                            docs.append(item)
                        elif isinstance(item, dict):
                            docs.append(
                                RetrievedDocument(
                                    doc_id=str(item.get("id", f"doc-{i + 1}")),
                                    content=str(item.get("content", item.get("text", ""))),
                                    score=float(item.get("score", 1.0)),
                                )
                            )
                        elif isinstance(item, str):
                            docs.append(
                                RetrievedDocument(doc_id=f"doc-{i + 1}", content=item, score=1.0)
                            )
                active.retrieval.documents = docs
                return result

        return cast(F, wrapper)

    return decorator
