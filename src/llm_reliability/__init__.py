"""LLM Reliability Analyzer.

A local-first, evidence-based reliability analyzer and root-cause diagnosis engine
for LLM, RAG, and Agent execution traces.
"""

from pathlib import Path
from typing import Any, TextIO

from llm_reliability.agent import (
    AgentAnalyzer,
    AgentLoopPattern,
    AgentMetrics,
    ToolCallEvaluation,
    ToolMetrics,
)
from llm_reliability.anonymization import (
    PIIScrubber,
    PIIType,
    RedactedItem,
    RedactionConfig,
    RedactionMaskType,
    RedactionResult,
    RedactionRule,
    TelemetryAnonymizer,
    anonymize_trace,
    scrub_pii,
)
from llm_reliability.benchmark import (
    BenchmarkReport,
    BenchmarkRunner,
    BenchmarkSample,
    CategoryAccuracyMetrics,
    SyntheticTraceGenerator,
    run_benchmark,
)
from llm_reliability.diagnosis import DiagnosticEngine
from llm_reliability.eval_harness import (
    Gatekeeper,
    GatekeeperConfig,
    GatekeeperReport,
    evaluate_gate,
)
from llm_reliability.exceptions import (
    LLMReliabilityError,
    TraceError,
    TraceParseError,
    TraceValidationError,
)
from llm_reliability.grounding import (
    ClaimSupport,
    GroundingAnalyzer,
    GroundingMetrics,
    SupportStatus,
)
from llm_reliability.models import (
    Diagnosis,
    Evidence,
    EvidenceType,
    Failure,
    FailureCategory,
    FinalResponse,
    Hypothesis,
    LLMCall,
    Metric,
    Recommendation,
    RetrievalStep,
    RetrievedDocument,
    Run,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    TokenUsage,
    ToolCall,
    ToolResult,
    Trace,
)
from llm_reliability.multimodal import (
    MultiModalAnalysisResult,
    MultiModalAnalyzer,
    MultiModalMetrics,
    MultiModalReliabilityAnalyzer,
    StructuredOutputAnalyzer,
    StructuredOutputMetrics,
)
from llm_reliability.normalization import (
    load_trace,
    normalize_trace,
)
from llm_reliability.otel import (
    OTelAttributeParser,
    OTelImporter,
    import_otel_trace,
)
from llm_reliability.patterns import (
    FailureCluster,
    FailureSignature,
    PatternAnalysisReport,
    PatternDetectionEngine,
)
from llm_reliability.plugins import (
    BaseAnalyzerPlugin,
    BaseExporterPlugin,
    BaseMiddlewarePlugin,
    BasePlugin,
    PluginManager,
    PluginMetadata,
    get_plugin_manager,
    register_plugin,
)
from llm_reliability.policies import (
    CustomRule,
    DiagnosticPolicy,
    PolicyEngine,
    PolicyEvaluationResult,
    PolicyViolation,
    ThresholdConfig,
)
from llm_reliability.rag import (
    RAGAnalyzer,
    RetrievalMetrics,
)
from llm_reliability.recommendations import RecommendationEngine
from llm_reliability.regression import (
    BatchComparisonReport,
    DistributionSummary,
    FailureRateSummary,
    RegressionEngine,
    RegressionVerdict,
)
from llm_reliability.report import (
    DiagnosticReportGenerator,
    render_html_report,
    save_html_report,
)
from llm_reliability.server import (
    DiagnosticServer,
    ServerConfig,
    start_server,
)
from llm_reliability.streaming import (
    InterTokenLatencyStats,
    StallEvent,
    StreamingProfiler,
    TokenChunk,
    TokenLatencyProfile,
    wrap_async_stream,
    wrap_stream,
)
from llm_reliability.tracing import (
    ActiveSpan,
    SpanContextManager,
    TraceContextManager,
    Tracer,
    get_tracer,
    trace,
    trace_llm,
    trace_retrieval,
    trace_tool,
    wrap_anthropic,
    wrap_openai,
    wrap_tool,
)
from llm_reliability.verification import (
    CounterfactualResult,
    MetricComparison,
    VerificationEngine,
    VerificationReport,
)

__version__ = "0.1.0.dev0"


def diagnose(
    source: Trace | Run | str | Path | dict[str, Any] | list[Any] | TextIO,
) -> Diagnosis:
    """Diagnose an LLM/RAG/Agent trace or run and return root-cause analysis with actionable recommendations."""
    engine = DiagnosticEngine()
    return engine.diagnose(source)


__all__ = [
    "__version__",
    # Main Diagnostic Entrypoints
    "diagnose",
    "DiagnosticEngine",
    "RecommendationEngine",
    "VerificationEngine",
    "VerificationReport",
    "CounterfactualResult",
    "MetricComparison",
    # Multi-Run Regression Engine
    "RegressionEngine",
    "BatchComparisonReport",
    "DistributionSummary",
    "FailureRateSummary",
    "RegressionVerdict",
    # Failure Pattern Detection Engine
    "PatternDetectionEngine",
    "PatternAnalysisReport",
    "FailureCluster",
    "FailureSignature",
    # Custom Diagnostic Policies Engine
    "PolicyEngine",
    "DiagnosticPolicy",
    "ThresholdConfig",
    "CustomRule",
    "PolicyViolation",
    "PolicyEvaluationResult",
    # Synthetic Generator & Benchmark Suite
    "SyntheticTraceGenerator",
    "BenchmarkRunner",
    "BenchmarkSample",
    "BenchmarkReport",
    "CategoryAccuracyMetrics",
    "run_benchmark",
    # OpenTelemetry & Distributed Tracing
    "OTelImporter",
    "OTelAttributeParser",
    "import_otel_trace",
    # Live Tracing SDK & Middleware
    "Tracer",
    "get_tracer",
    "ActiveSpan",
    "SpanContextManager",
    "TraceContextManager",
    "trace",
    "trace_llm",
    "trace_tool",
    "trace_retrieval",
    "wrap_openai",
    "wrap_anthropic",
    "wrap_tool",
    # Streaming & Token Latency Profiler
    "StreamingProfiler",
    "TokenLatencyProfile",
    "TokenChunk",
    "StallEvent",
    "InterTokenLatencyStats",
    "wrap_stream",
    "wrap_async_stream",
    # Interactive HTML Report Generator
    "DiagnosticReportGenerator",
    "render_html_report",
    "save_html_report",
    # Local Diagnostic Visualizer Server
    "DiagnosticServer",
    "ServerConfig",
    "start_server",
    # Telemetry Anonymization & PII Scrubbing
    "PIIScrubber",
    "TelemetryAnonymizer",
    "PIIType",
    "RedactionMaskType",
    "RedactionRule",
    "RedactionConfig",
    "RedactedItem",
    "RedactionResult",
    "scrub_pii",
    "anonymize_trace",
    # Multi-Modal & Structured Output Reliability
    "StructuredOutputMetrics",
    "MultiModalMetrics",
    "MultiModalAnalysisResult",
    "StructuredOutputAnalyzer",
    "MultiModalAnalyzer",
    "MultiModalReliabilityAnalyzer",
    # Evaluation Harness & CI Gatekeeper
    "Gatekeeper",
    "GatekeeperConfig",
    "GatekeeperReport",
    "evaluate_gate",
    # Plugins & Extensibility SDK
    "BasePlugin",
    "BaseAnalyzerPlugin",
    "BaseExporterPlugin",
    "BaseMiddlewarePlugin",
    "PluginMetadata",
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    # Ingestion & Normalization
    "load_trace",
    "normalize_trace",
    # RAG Analysis Engine
    "RAGAnalyzer",
    "RetrievalMetrics",
    # Grounding Analysis Engine
    "GroundingAnalyzer",
    "GroundingMetrics",
    "ClaimSupport",
    "SupportStatus",
    # Agent & Tool Analysis Engine
    "AgentAnalyzer",
    "AgentMetrics",
    "ToolMetrics",
    "AgentLoopPattern",
    "ToolCallEvaluation",
    # Exceptions
    "LLMReliabilityError",
    "TraceError",
    "TraceParseError",
    "TraceValidationError",
    # Enums
    "SpanKind",
    "SpanStatus",
    "FailureCategory",
    "Severity",
    "EvidenceType",
    # Execution
    "TokenUsage",
    "RetrievedDocument",
    "RetrievalStep",
    "LLMCall",
    "ToolCall",
    "ToolResult",
    "FinalResponse",
    # Trace hierarchy
    "Span",
    "Run",
    "Trace",
    # Diagnosis
    "Metric",
    "Evidence",
    "Failure",
    "Hypothesis",
    "Recommendation",
    "Diagnosis",
]
