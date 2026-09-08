"""Programmatic synthetic failure trace generator for benchmarking and regression testing."""

import uuid

from llm_reliability.benchmark.models import BenchmarkSample
from llm_reliability.models.enums import FailureCategory, Severity, SpanKind, SpanStatus
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


class SyntheticTraceGenerator:
    """Generates synthetic execution traces with controlled failure injection across all failure categories."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize synthetic generator."""
        self.seed = seed

    def generate_clean_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a healthy trace with relevant retrieval, grounded response, and successful tool execution."""
        t_id = trace_id or f"trace-clean-{uuid.uuid4().hex[:8]}"
        doc = RetrievedDocument(
            doc_id="doc-1",
            content="Python 3.13 introduces experimental free-threaded execution mode.",
            score=0.96,
        )
        retrieval_span = Span(
            span_id=f"span-ret-{uuid.uuid4().hex[:6]}",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            retrieval=RetrievalStep(query="python 3.13 features", documents=[doc]),
        )
        llm_span = Span(
            span_id=f"span-llm-{uuid.uuid4().hex[:6]}",
            name="llm_generate",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            llm_call=LLMCall(
                model="gpt-4o",
                prompt="Summarize Python 3.13",
                response="Python 3.13 introduces experimental free-threaded execution mode.",
                token_usage=TokenUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            ),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[retrieval_span, llm_span],
            final_response=FinalResponse(
                text="Python 3.13 introduces experimental free-threaded execution mode."
            ),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_empty_retrieval_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace with an empty retrieval step (0 documents)."""
        t_id = trace_id or f"trace-empty-ret-{uuid.uuid4().hex[:8]}"
        retrieval_span = Span(
            span_id=f"span-ret-{uuid.uuid4().hex[:6]}",
            name="vector_search",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            retrieval=RetrievalStep(query="rare scientific constant zeta", documents=[]),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[retrieval_span],
            final_response=FinalResponse(text="I could not find any information about that."),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_low_relevance_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace with low relevance retrieval candidates."""
        t_id = trace_id or f"trace-low-rel-{uuid.uuid4().hex[:8]}"
        doc = RetrievedDocument(
            doc_id="doc-irrelevant",
            content="Ancient pottery techniques used clay from riverbanks in Mesopotamia.",
            score=0.25,
        )
        retrieval_span = Span(
            span_id=f"span-ret-{uuid.uuid4().hex[:6]}",
            name="vector_search",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            retrieval=RetrievalStep(query="how to deploy kubernetes cluster", documents=[doc]),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[retrieval_span],
            final_response=FinalResponse(
                text="Pottery in Mesopotamia was built with riverbank clay."
            ),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_duplicate_chunks_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace with severe document duplication in retrieved context."""
        t_id = trace_id or f"trace-dup-{uuid.uuid4().hex[:8]}"
        content = "OAuth 2.0 is an authorization framework that enables applications to obtain limited access."
        docs = [RetrievedDocument(doc_id=f"doc-{i}", content=content, score=0.92) for i in range(4)]
        retrieval_span = Span(
            span_id=f"span-ret-{uuid.uuid4().hex[:6]}",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            retrieval=RetrievalStep(query="what is oauth2", documents=docs),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[retrieval_span],
            final_response=FinalResponse(
                text="OAuth 2.0 is an authorization framework for limited access."
            ),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_hallucination_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace where the generated response explicitly contradicts retrieved context."""
        t_id = trace_id or f"trace-hallucination-{uuid.uuid4().hex[:8]}"
        doc = RetrievedDocument(
            doc_id="doc-mars",
            content="Mars has two small moons, Phobos and Deimos.",
            score=0.95,
        )
        retrieval_span = Span(
            span_id=f"span-ret-{uuid.uuid4().hex[:6]}",
            name="retrieval",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            retrieval=RetrievalStep(query="how many moons does mars have", documents=[doc]),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[retrieval_span],
            final_response=FinalResponse(
                text="Mars has zero moons and cannot support natural satellites."
            ),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_agent_loop_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace where an agent enters an infinite repetitive tool loop."""
        t_id = trace_id or f"trace-loop-{uuid.uuid4().hex[:8]}"
        spans: list[Span] = []
        for i in range(4):
            spans.append(
                Span(
                    span_id=f"span-tool-{i}",
                    name="web_search",
                    kind=SpanKind.TOOL,
                    status=SpanStatus.SUCCESS,
                    tool_call=ToolCall(
                        tool_name="web_search", arguments={"query": "weather today"}
                    ),
                    tool_result=ToolResult(
                        tool_name="web_search", output={"status": "retry required"}
                    ),
                )
            )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=spans,
            final_response=FinalResponse(text="Agent exceeded execution limit."),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_tool_error_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace where a tool execution fails with an error status."""
        t_id = trace_id or f"trace-tool-err-{uuid.uuid4().hex[:8]}"
        span = Span(
            span_id=f"span-tool-{uuid.uuid4().hex[:6]}",
            name="database_query",
            kind=SpanKind.TOOL,
            status=SpanStatus.ERROR,
            error_message="DatabaseConnectionError: Connection timeout to primary replica",
            tool_call=ToolCall(
                tool_name="database_query", arguments={"sql": "SELECT * FROM users;"}
            ),
            tool_result=ToolResult(
                tool_name="database_query",
                status=SpanStatus.ERROR,
                error="DatabaseConnectionError: Connection timeout to primary replica",
            ),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[span],
            final_response=FinalResponse(
                text="An error occurred while querying the user database."
            ),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_tool_argument_error_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace where tool arguments fail validation or are malformed."""
        t_id = trace_id or f"trace-tool-arg-{uuid.uuid4().hex[:8]}"
        span = Span(
            span_id=f"span-tool-{uuid.uuid4().hex[:6]}",
            name="send_payment",
            kind=SpanKind.TOOL,
            status=SpanStatus.SUCCESS,
            tool_call=ToolCall(
                tool_name="send_payment", arguments='{"amount": null, "broken_json": '
            ),
        )
        run = Run(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            trace_id=t_id,
            spans=[span],
            final_response=FinalResponse(text="Payment failed due to invalid arguments."),
        )
        return Trace(trace_id=t_id, runs=[run])

    def generate_llm_api_error_trace(self, trace_id: str | None = None) -> Trace:
        """Generate a trace where an LLM provider call fails with a server/rate-limit error."""
        t_id = trace_id or f"trace-llm-err-{uuid.uuid4().hex[:8]}"
        span = Span(
            span_id=f"span-llm-{uuid.uuid4().hex[:6]}",
            name="llm_call",
            kind=SpanKind.LLM,
            status=SpanStatus.ERROR,
            error_message="RateLimitError: 429 Too Many Requests from OpenAI API",
            llm_call=LLMCall(model="gpt-4o", prompt="Generate comprehensive financial plan"),
        )
        run = Run(run_id=f"run-{uuid.uuid4().hex[:8]}", trace_id=t_id, spans=[span])
        return Trace(trace_id=t_id, runs=[run])

    def generate_benchmark_suite(self, samples_per_category: int = 5) -> list[BenchmarkSample]:
        """Generate a comprehensive ground-truth labeled benchmark suite covering all failure categories."""
        samples: list[BenchmarkSample] = []
        generators = [
            (
                "clean",
                FailureCategory.NONE,
                Severity.INFO,
                self.generate_clean_trace,
                "Clean healthy execution",
            ),
            (
                "empty_retrieval",
                FailureCategory.RETRIEVAL_FAILURE,
                Severity.HIGH,
                self.generate_empty_retrieval_trace,
                "Zero document retrieval failure",
            ),
            (
                "low_relevance",
                FailureCategory.RETRIEVAL_FAILURE,
                Severity.HIGH,
                self.generate_low_relevance_trace,
                "Low relevance retrieval candidates",
            ),
            (
                "duplicate_chunks",
                FailureCategory.CONTEXT_CONSTRUCTION_FAILURE,
                Severity.MEDIUM,
                self.generate_duplicate_chunks_trace,
                "Excessive duplicate context chunks",
            ),
            (
                "hallucination",
                FailureCategory.GROUNDING_FAILURE,
                Severity.HIGH,
                self.generate_hallucination_trace,
                "Factual contradiction against context",
            ),
            (
                "agent_loop",
                FailureCategory.AGENT_LOOP,
                Severity.HIGH,
                self.generate_agent_loop_trace,
                "Infinite repetitive tool loop",
            ),
            (
                "tool_error",
                FailureCategory.TOOL_ERROR,
                Severity.HIGH,
                self.generate_tool_error_trace,
                "Tool execution server failure",
            ),
            (
                "tool_arg_error",
                FailureCategory.SCHEMA_VIOLATION,
                Severity.HIGH,
                self.generate_tool_argument_error_trace,
                "Tool argument validation failure",
            ),
            (
                "llm_api_error",
                FailureCategory.LLM_CALL_FAILURE,
                Severity.HIGH,
                self.generate_llm_api_error_trace,
                "LLM provider rate limit / 429 error",
            ),
        ]

        for prefix, cat, sev, gen_fn, desc in generators:
            for i in range(samples_per_category):
                trace = gen_fn(trace_id=f"{prefix}-{i + 1}")
                samples.append(
                    BenchmarkSample(
                        sample_id=f"sample-{prefix}-{i + 1}",
                        trace=trace,
                        ground_truth_category=cat,
                        expected_severity=sev,
                        description=desc,
                    )
                )

        return samples
