"""Tests for agent and tool failure analysis engine."""

import pytest

from llm_reliability.agent import (
    AgentAnalyzer,
    AgentMetrics,
)
from llm_reliability.models.enums import EvidenceType, SpanKind, SpanStatus
from llm_reliability.models.execution import ToolCall, ToolResult
from llm_reliability.models.trace import Run, Span, Trace


class TestAgentAnalyzer:
    """Deterministic tests for agent trajectory and tool failure analysis."""

    @pytest.fixture
    def analyzer(self) -> AgentAnalyzer:
        return AgentAnalyzer(max_step_limit=10, max_repeated_calls_threshold=2)

    def test_clean_agent_run(self, analyzer: AgentAnalyzer) -> None:
        span1 = Span(
            span_id="s1",
            name="search_users",
            kind=SpanKind.TOOL,
            status=SpanStatus.SUCCESS,
            tool_call=ToolCall(tool_name="search_users", arguments={"query": "Alice"}),
            tool_result=ToolResult(tool_name="search_users", output={"id": 1, "name": "Alice"}),
            duration_ms=45.0,
        )
        span2 = Span(
            span_id="s2",
            name="get_orders",
            kind=SpanKind.TOOL,
            status=SpanStatus.SUCCESS,
            tool_call=ToolCall(tool_name="get_orders", arguments={"user_id": 1}),
            tool_result=ToolResult(tool_name="get_orders", output=[{"order_id": 101}]),
            duration_ms=60.0,
        )

        run = Run(run_id="r1", trace_id="t1", spans=[span1, span2])
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.total_steps == 2
        assert metrics.tool_call_count == 2
        assert metrics.has_loop is False
        assert metrics.is_exceeded_step_limit is False
        assert metrics.has_unresolved_error is False
        assert metrics.state_drift_detected is False
        assert metrics.tool_metrics.failed_calls == 0
        assert metrics.tool_metrics.failure_rate == 0.0
        assert metrics.tool_metrics.argument_error_count == 0
        assert metrics.tool_metrics.repeated_call_count == 0
        assert len(metrics.tool_metrics.unique_tools_used) == 2

    def test_tool_execution_failure_detection(self, analyzer: AgentAnalyzer) -> None:
        span1 = Span(
            span_id="s1",
            name="fetch_api",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="HTTP 503 Service Unavailable",
            tool_call=ToolCall(tool_name="fetch_api", arguments={"endpoint": "/data"}),
            tool_result=ToolResult(tool_name="fetch_api", error="HTTP 503 Service Unavailable"),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span1])
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.tool_metrics.total_tool_calls == 1
        assert metrics.tool_metrics.failed_calls == 1
        assert metrics.tool_metrics.failure_rate == 1.0
        assert metrics.has_unresolved_error is True

    def test_malformed_argument_error(self, analyzer: AgentAnalyzer) -> None:
        span1 = Span(
            span_id="s1",
            name="run_sql",
            kind=SpanKind.TOOL,
            tool_call=ToolCall(tool_name="run_sql", arguments="{ unquoted_sql: SELECT * "),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span1])
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.tool_metrics.argument_error_count == 1
        assert metrics.tool_metrics.evaluations[0].has_argument_error is True
        assert "Invalid JSON" in (metrics.tool_metrics.evaluations[0].argument_error_detail or "")

    def test_exact_repetition_loop_detection(self, analyzer: AgentAnalyzer) -> None:
        spans = [
            Span(
                span_id=f"s{i}",
                name="search_web",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search_web", arguments={"q": "current weather"}),
            )
            for i in range(3)
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.has_loop is True
        assert metrics.tool_metrics.repeated_call_count == 2
        assert any(loop.loop_type == "exact_repetition" for loop in metrics.detected_loops)

    def test_alternating_ping_pong_loop_detection(self, analyzer: AgentAnalyzer) -> None:
        spans = [
            Span(
                span_id="s1",
                name="tool_a",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="tool_a", arguments={"x": 1}),
            ),
            Span(
                span_id="s2",
                name="tool_b",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="tool_b", arguments={"y": 2}),
            ),
            Span(
                span_id="s3",
                name="tool_a",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="tool_a", arguments={"x": 1}),
            ),
            Span(
                span_id="s4",
                name="tool_b",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="tool_b", arguments={"y": 2}),
            ),
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.has_loop is True
        assert any(loop.loop_type == "alternating_cycle" for loop in metrics.detected_loops)

    def test_failed_retry_loop_detection(self, analyzer: AgentAnalyzer) -> None:
        span1 = Span(
            span_id="s1",
            name="db_query",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="Table not found",
            tool_call=ToolCall(tool_name="db_query", arguments={"table": "customers"}),
        )
        span2 = Span(
            span_id="s2",
            name="db_query",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="Table not found",
            tool_call=ToolCall(tool_name="db_query", arguments={"table": "customers"}),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span1, span2])
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.has_loop is True
        assert any(loop.loop_type == "failed_retry_loop" for loop in metrics.detected_loops)

    def test_max_step_limit_exceeded(self, analyzer: AgentAnalyzer) -> None:
        spans = [
            Span(
                span_id=f"s{i}",
                name=f"step_{i}",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name=f"t_{i}"),
            )
            for i in range(12)
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.total_steps == 12
        assert metrics.is_exceeded_step_limit is True

    def test_agent_state_drift_detection(self, analyzer: AgentAnalyzer) -> None:
        spans = [
            Span(
                span_id="s1",
                name="search",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search", arguments={"q": "same"}),
            ),
            Span(
                span_id="s2",
                name="search",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search", arguments={"q": "same"}),
            ),
            Span(
                span_id="s3",
                name="search",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search", arguments={"q": "same"}),
            ),
            Span(
                span_id="s4",
                name="search",
                kind=SpanKind.TOOL,
                tool_call=ToolCall(tool_name="search", arguments={"q": "same"}),
            ),
        ]
        run = Run(run_id="r1", trace_id="t1", spans=spans)
        metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        assert metrics.state_drift_detected is True

    def test_agent_metrics_to_diagnostic_metrics_and_evidence(
        self, analyzer: AgentAnalyzer
    ) -> None:
        span1 = Span(
            span_id="s1",
            name="calculator",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="Division by zero",
            tool_call=ToolCall(tool_name="calculator", arguments="{ invalid_json: "),
            tool_result=ToolResult(tool_name="calculator", error="Division by zero"),
        )
        span2 = Span(
            span_id="s2",
            name="calculator",
            kind=SpanKind.TOOL,
            tool_call=ToolCall(tool_name="calculator", arguments="{ invalid_json: "),
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span1, span2])
        agent_metrics: AgentMetrics = analyzer.analyze_agent_run(run)

        # Convert to metrics
        metrics = agent_metrics.to_diagnostic_metrics()
        metric_names = [m.name for m in metrics]
        assert "agent_loop_detected" in metric_names
        assert "tool_call_failure_rate" in metric_names
        assert "tool_argument_errors" in metric_names
        assert "tool_repeated_calls" in metric_names

        # Convert to evidence
        evidence = agent_metrics.to_evidence(span_id="run-span-1")
        assert len(evidence) >= 2
        evidence_types = [e.evidence_type for e in evidence]
        assert EvidenceType.TOOL_REPETITION in evidence_types
        assert EvidenceType.ERROR_LOG in evidence_types
        assert EvidenceType.SCHEMA_VIOLATION in evidence_types

    def test_trace_level_agent_analysis(self, analyzer: AgentAnalyzer) -> None:
        span1 = Span(
            span_id="s1", name="tool_1", kind=SpanKind.TOOL, tool_call=ToolCall(tool_name="tool_1")
        )
        trace = Trace(trace_id="t1", runs=[Run(run_id="r1", trace_id="t1", spans=[span1])])
        results = analyzer.analyze_trace(trace)

        assert len(results) == 1
        assert results[0].tool_call_count == 1
