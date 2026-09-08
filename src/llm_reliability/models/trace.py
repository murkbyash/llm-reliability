"""Hierarchical trace, run, and span models for capturing execution pipelines."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import (
    FinalResponse,
    LLMCall,
    RetrievalStep,
    TokenUsage,
    ToolCall,
    ToolResult,
)


class Span(BaseModel):
    """A single logical unit of execution in a pipeline (e.g. LLM call, retrieval, tool)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    span_id: str = Field(description="Unique identifier for this span")
    parent_span_id: str | None = Field(
        default=None, description="Identifier of parent span if nested"
    )
    name: str = Field(
        description="Descriptive name of the operation (e.g. 'vector_search', 'llm_generate')"
    )
    kind: SpanKind = Field(default=SpanKind.CUSTOM, description="Classification of this span")
    status: SpanStatus = Field(default=SpanStatus.SUCCESS, description="Execution status")
    start_time: datetime | None = Field(default=None, description="Start timestamp")
    end_time: datetime | None = Field(default=None, description="End timestamp")
    duration_ms: float | None = Field(default=None, ge=0.0, description="Duration in milliseconds")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary key-value attributes"
    )
    events: list[dict[str, Any]] = Field(
        default_factory=list, description="Timestamped events or logs within span"
    )
    error_message: str | None = Field(default=None, description="Error message if status is ERROR")

    # Granular typed payloads
    llm_call: LLMCall | None = Field(default=None, description="Typed LLM invocation payload")
    retrieval: RetrievalStep | None = Field(
        default=None, description="Typed retrieval step payload"
    )
    tool_call: ToolCall | None = Field(default=None, description="Typed tool call payload")
    tool_result: ToolResult | None = Field(default=None, description="Typed tool result payload")

    @property
    def is_root(self) -> bool:
        """Return True if this is a top-level root span without a parent."""
        return self.parent_span_id is None

    @property
    def is_error(self) -> bool:
        """Return True if the span resulted in an error."""
        return self.status == SpanStatus.ERROR or self.error_message is not None


class Run(BaseModel):
    """An execution run encompassing a sequence or hierarchy of spans."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    run_id: str = Field(description="Unique identifier for this run")
    trace_id: str = Field(description="Identifier of parent trace")
    name: str | None = Field(default=None, description="Name of the run or pipeline")
    spans: list[Span] = Field(default_factory=list, description="List of spans in this run")
    input_query: str | None = Field(default=None, description="Original user prompt or input query")
    final_response: FinalResponse | None = Field(
        default=None, description="Final response produced by run"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Run-level metadata")
    start_time: datetime | None = Field(default=None, description="Run start timestamp")
    end_time: datetime | None = Field(default=None, description="Run completion timestamp")
    duration_ms: float | None = Field(default=None, ge=0.0, description="Total run duration in ms")

    def get_span(self, span_id: str) -> Span | None:
        """Find a span by its ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_spans_by_kind(self, kind: SpanKind) -> list[Span]:
        """Filter spans by kind."""
        return [span for span in self.spans if span.kind == kind]

    def get_root_spans(self) -> list[Span]:
        """Return all root spans in this run."""
        return [span for span in self.spans if span.is_root]

    def get_child_spans(self, parent_span_id: str) -> list[Span]:
        """Return direct children of a given span ID."""
        return [span for span in self.spans if span.parent_span_id == parent_span_id]

    def get_llm_calls(self) -> list[LLMCall]:
        """Return all typed LLM calls in this run."""
        return [span.llm_call for span in self.spans if span.llm_call is not None]

    def get_retrievals(self) -> list[RetrievalStep]:
        """Return all typed retrieval steps in this run."""
        return [span.retrieval for span in self.spans if span.retrieval is not None]

    def get_tool_calls(self) -> list[ToolCall]:
        """Return all typed tool calls in this run."""
        return [span.tool_call for span in self.spans if span.tool_call is not None]


class Trace(BaseModel):
    """Top-level container representing an end-to-end execution trace."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    trace_id: str = Field(description="Unique identifier for the trace")
    name: str | None = Field(default=None, description="Trace or workflow name")
    runs: list[Run] = Field(default_factory=list, description="Runs associated with this trace")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Trace-level metadata")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")

    def get_all_spans(self) -> list[Span]:
        """Collect all spans across all runs in this trace."""
        all_spans: list[Span] = []
        for run in self.runs:
            all_spans.extend(run.spans)
        return all_spans

    def get_llm_calls(self) -> list[LLMCall]:
        """Collect all typed LLM calls in this trace."""
        calls: list[LLMCall] = []
        for span in self.get_all_spans():
            if span.llm_call is not None:
                calls.append(span.llm_call)
        return calls

    def get_retrievals(self) -> list[RetrievalStep]:
        """Collect all typed retrieval steps in this trace."""
        retrievals: list[RetrievalStep] = []
        for span in self.get_all_spans():
            if span.retrieval is not None:
                retrievals.append(span.retrieval)
        return retrievals

    def get_tool_calls(self) -> list[ToolCall]:
        """Collect all typed tool calls in this trace."""
        tools: list[ToolCall] = []
        for span in self.get_all_spans():
            if span.tool_call is not None:
                tools.append(span.tool_call)
        return tools

    def aggregate_token_usage(self) -> TokenUsage:
        """Compute total token consumption across all LLM spans."""
        prompt_total = 0
        completion_total = 0
        total_total = 0
        cost_total = 0.0
        has_cost = False

        for call in self.get_llm_calls():
            if call.token_usage is not None:
                prompt_total += call.token_usage.prompt_tokens
                completion_total += call.token_usage.completion_tokens
                total_total += call.token_usage.total_tokens
                if call.token_usage.cost_usd is not None:
                    cost_total += call.token_usage.cost_usd
                    has_cost = True

        return TokenUsage(
            prompt_tokens=prompt_total,
            completion_tokens=completion_total,
            total_tokens=total_total if total_total > 0 else (prompt_total + completion_total),
            cost_usd=cost_total if has_cost else None,
        )
