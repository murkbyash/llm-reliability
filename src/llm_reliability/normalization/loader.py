"""Trace loader supporting JSON files, raw strings, file streams, and dictionaries."""

import json
from pathlib import Path
from typing import Any, TextIO

from llm_reliability.exceptions import TraceParseError, TraceValidationError
from llm_reliability.models.trace import Trace
from llm_reliability.normalization.normalizer import normalize_trace


def load_trace(source: str | Path | dict[str, Any] | list[Any] | TextIO | None) -> Trace:
    """Load and normalize a trace from a file path, raw JSON string, file stream, or dict.

    Args:
        source: File path (str or Path), raw JSON string, open text stream, or python dict/list.

    Returns:
        Canonical Trace model instance.

    Raises:
        TraceParseError: If file cannot be read or contains invalid JSON syntax.
        TraceValidationError: If trace payload is empty or invalid.
    """
    if source is None:
        raise TraceValidationError("Trace source cannot be None.")

    # Direct dictionary or list
    if isinstance(source, (dict, list)):
        return normalize_trace(source)

    # Path object
    if isinstance(source, Path):
        if not source.is_file():
            raise TraceParseError(f"Trace file not found: {source}")
        try:
            content = source.read_text(encoding="utf-8")
        except Exception as e:
            raise TraceParseError(f"Failed to read trace file {source}: {e}") from e
        return _parse_and_normalize_json_string(content, source_hint=str(source))

    # TextIO stream (e.g. open file, StringIO)
    if hasattr(source, "read") and callable(getattr(source, "read", None)):
        try:
            content = source.read()
        except Exception as e:
            raise TraceParseError(f"Failed to read from trace stream: {e}") from e
        return _parse_and_normalize_json_string(content, source_hint="stream")

    # String (could be a filepath OR a raw JSON string)
    if isinstance(source, str):
        cleaned = source.strip()
        if not cleaned:
            raise TraceValidationError("Trace source string is empty.")

        # Check if it looks like a filepath on filesystem
        path_candidate = Path(cleaned)
        if cleaned.endswith(".json") or cleaned.endswith(".jsonl") or path_candidate.is_file():
            if not path_candidate.is_file():
                raise TraceParseError(f"Trace file not found: {cleaned}")
            try:
                content = path_candidate.read_text(encoding="utf-8")
            except Exception as e:
                raise TraceParseError(f"Failed to read trace file {cleaned}: {e}") from e
            return _parse_and_normalize_json_string(content, source_hint=cleaned)

        # Otherwise treat as raw JSON string
        return _parse_and_normalize_json_string(cleaned, source_hint="string")

    raise TraceValidationError(
        f"Unsupported trace source type: {type(source).__name__}. "
        "Expected str, Path, dict, list, or TextIO."
    )


def _parse_and_normalize_json_string(content: str, source_hint: str) -> Trace:
    """Parse JSON string and pass the parsed object to normalize_trace."""
    cleaned = content.strip()
    if not cleaned:
        raise TraceValidationError(f"Trace content from {source_hint} is empty.")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as err:
        raise TraceParseError(f"Invalid JSON syntax in trace from {source_hint}: {err}") from err

    return normalize_trace(data)
