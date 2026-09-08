"""Core enumeration types for LLM Reliability Analyzer."""

from enum import Enum


class SpanKind(str, Enum):
    """Categorization of individual execution step or span."""

    ROOT = "root"
    LLM = "llm"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    TOOL = "tool"
    CHAIN = "chain"
    AGENT = "agent"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    """Execution status of a span."""

    UNSET = "unset"
    SUCCESS = "success"
    ERROR = "error"


class FailureCategory(str, Enum):
    """Root-cause and failure category classifications."""

    NONE = "NONE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    CONTEXT_CONSTRUCTION_FAILURE = "CONTEXT_CONSTRUCTION_FAILURE"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    HALLUCINATION = "HALLUCINATION"
    PROMPT_FAILURE = "PROMPT_FAILURE"
    OUTPUT_FORMAT_FAILURE = "OUTPUT_FORMAT_FAILURE"
    AGENT_LOOP = "AGENT_LOOP"
    AGENT_LOOP_FAILURE = "AGENT_LOOP_FAILURE"
    TOOL_ERROR = "TOOL_ERROR"
    TOOL_EXECUTION_FAILURE = "TOOL_EXECUTION_FAILURE"
    TOOL_SELECTION_FAILURE = "TOOL_SELECTION_FAILURE"
    LLM_CALL_FAILURE = "LLM_CALL_FAILURE"
    LATENCY_FAILURE = "LATENCY_FAILURE"
    TOKEN_ANOMALY = "TOKEN_ANOMALY"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    """Severity levels for diagnostic failures and alerts."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceType(str, Enum):
    """Classification of diagnostic evidence."""

    METRIC_THRESHOLD = "metric_threshold"
    TOKEN_DISCREPANCY = "token_discrepancy"
    RELEVANCE_SCORE = "relevance_score"
    RETRIEVAL_SCORE = "retrieval_score"
    HALLUCINATION_OVERLAP = "hallucination_overlap"
    GROUNDING_DEFICIT = "grounding_deficit"
    CONTRADICTION = "contradiction"
    TOOL_REPETITION = "tool_repetition"
    SCHEMA_VIOLATION = "schema_violation"
    LATENCY_SPIKE = "latency_spike"
    ERROR_LOG = "error_log"
    CUSTOM = "custom"
