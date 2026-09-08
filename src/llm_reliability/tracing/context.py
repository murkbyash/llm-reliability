"""Async-safe and thread-safe execution context management for live tracing."""

import contextvars
from typing import Any

_current_trace_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "current_trace_context", default=None
)

_current_parent_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_parent_span_id", default=None
)


def get_current_context() -> dict[str, Any] | None:
    """Retrieve active tracing context for the current async task or thread."""
    return _current_trace_context.get()


def set_current_context(ctx: dict[str, Any] | None) -> contextvars.Token[Any]:
    """Set active tracing context for the current async task or thread."""
    return _current_trace_context.set(ctx)


def reset_current_context(token: contextvars.Token[Any]) -> None:
    """Reset tracing context using previous token."""
    _current_trace_context.reset(token)


def get_current_parent_span_id() -> str | None:
    """Retrieve active parent span ID."""
    return _current_parent_span_id.get()


def set_current_parent_span_id(span_id: str | None) -> contextvars.Token[Any]:
    """Set active parent span ID."""
    return _current_parent_span_id.set(span_id)


def reset_current_parent_span_id(token: contextvars.Token[Any]) -> None:
    """Reset active parent span ID using previous token."""
    _current_parent_span_id.reset(token)
