"""Synchronous and asynchronous streaming generator wrappers with automatic token latency profiling."""

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeVar

from llm_reliability.streaming.models import TokenLatencyProfile
from llm_reliability.streaming.profiler import StreamingProfiler

T = TypeVar("T")


def _extract_chunk_text(chunk: Any) -> str:
    """Extract plain text string from arbitrary provider chunk objects."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")

    # OpenAI ChatCompletionChunk
    if hasattr(chunk, "choices") and chunk.choices:
        first_choice = chunk.choices[0]
        if hasattr(first_choice, "delta") and hasattr(first_choice.delta, "content"):
            content = first_choice.delta.content
            return str(content) if content is not None else ""
        if isinstance(first_choice, dict) and "delta" in first_choice:
            return str(first_choice["delta"].get("content", ""))

    # Anthropic Stream Event
    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
        return str(chunk.delta.text)

    # General dict
    if isinstance(chunk, dict):
        if "text" in chunk:
            return str(chunk["text"])
        if "content" in chunk:
            return str(chunk["content"])
        if "delta" in chunk and isinstance(chunk["delta"], dict):
            return str(chunk["delta"].get("content", chunk["delta"].get("text", "")))

    return ""


def wrap_stream(
    stream: Iterator[T] | Any,
    profiler: StreamingProfiler | None = None,
    on_complete: Callable[[TokenLatencyProfile, str], None] | None = None,
    stall_threshold_ms: float = 500.0,
) -> Iterator[T]:
    """Wrap a synchronous streaming iterator to capture token latency and throughput metrics.

    Yields each original chunk unmodified while profiling arrival times in the background.
    """
    active_profiler = profiler or StreamingProfiler(stall_threshold_ms=stall_threshold_ms)
    active_profiler.start()
    accumulated_text: list[str] = []

    try:
        for chunk in stream:
            chunk_text = _extract_chunk_text(chunk)
            if chunk_text:
                accumulated_text.append(chunk_text)
            active_profiler.record_chunk(text=chunk_text)
            yield chunk
    finally:
        profile = active_profiler.finish()
        full_text = "".join(accumulated_text)
        if on_complete:
            on_complete(profile, full_text)


async def wrap_async_stream(
    stream: AsyncIterator[T] | Any,
    profiler: StreamingProfiler | None = None,
    on_complete: Callable[[TokenLatencyProfile, str], None] | None = None,
    stall_threshold_ms: float = 500.0,
) -> AsyncIterator[T]:
    """Wrap an asynchronous streaming iterator to capture token latency and throughput metrics.

    Yields each original chunk asynchronously while profiling arrival times in the background.
    """
    active_profiler = profiler or StreamingProfiler(stall_threshold_ms=stall_threshold_ms)
    active_profiler.start()
    accumulated_text: list[str] = []

    try:
        async for chunk in stream:
            chunk_text = _extract_chunk_text(chunk)
            if chunk_text:
                accumulated_text.append(chunk_text)
            active_profiler.record_chunk(text=chunk_text)
            yield chunk
    finally:
        profile = active_profiler.finish()
        full_text = "".join(accumulated_text)
        if on_complete:
            on_complete(profile, full_text)
