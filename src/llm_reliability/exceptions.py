"""Custom exception hierarchy for LLM Reliability Analyzer."""


class LLMReliabilityError(Exception):
    """Base exception for all errors raised by LLM Reliability Analyzer."""


class TraceError(LLMReliabilityError):
    """Base exception for trace loading, parsing, and normalization errors."""


class TraceParseError(TraceError):
    """Raised when trace data cannot be parsed from JSON or file input."""


class TraceValidationError(TraceError):
    """Raised when trace data structure cannot be validated or normalized."""
