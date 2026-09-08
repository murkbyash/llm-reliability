"""Models for agent and tool execution failure analysis."""

from pydantic import BaseModel, ConfigDict, Field

from llm_reliability.models.diagnosis import Evidence, Metric
from llm_reliability.models.enums import EvidenceType, SpanStatus


class ToolCallEvaluation(BaseModel):
    """Detailed evaluation of an individual tool call within an agent run."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tool_name: str = Field(..., description="Name of the invoked tool")
    call_id: str | None = Field(default=None, description="Identifier of the tool call")
    status: SpanStatus = Field(default=SpanStatus.SUCCESS, description="Execution status")
    has_argument_error: bool = Field(
        default=False, description="True if arguments were malformed or invalid"
    )
    argument_error_detail: str | None = Field(
        default=None, description="Description of argument parsing error"
    )
    is_duplicate_call: bool = Field(
        default=False, description="True if identical tool+arguments was called before"
    )
    is_failed_retry: bool = Field(
        default=False, description="True if repeating a failed call with identical parameters"
    )
    latency_ms: float | None = Field(default=None, ge=0.0, description="Tool execution duration")
    error_message: str | None = Field(default=None, description="Error message if execution failed")


class ToolMetrics(BaseModel):
    """Summary metrics evaluating tool usage across an execution."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    total_tool_calls: int = Field(default=0, ge=0, description="Total number of tool calls")
    successful_calls: int = Field(default=0, ge=0, description="Number of successful tool calls")
    failed_calls: int = Field(default=0, ge=0, description="Number of failed tool calls")
    failure_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Proportion of failed tool calls"
    )
    unique_tools_used: list[str] = Field(
        default_factory=list, description="List of unique tool names invoked"
    )
    argument_error_count: int = Field(
        default=0, ge=0, description="Count of argument or schema errors"
    )
    repeated_call_count: int = Field(
        default=0, ge=0, description="Count of identical repeated tool invocations"
    )
    average_tool_latency_ms: float | None = Field(
        default=None, description="Average tool latency in ms"
    )
    evaluations: list[ToolCallEvaluation] = Field(
        default_factory=list, description="Per-call evaluations"
    )

    def to_diagnostic_metrics(self) -> list[Metric]:
        """Convert tool metrics into standard diagnostic Metric instances."""
        return [
            Metric(
                name="tool_call_failure_rate",
                value=self.failure_rate,
                threshold=0.0,
                unit="ratio",
                passed=self.failure_rate == 0.0,
                details={
                    "total_tool_calls": self.total_tool_calls,
                    "failed_calls": self.failed_calls,
                },
            ),
            Metric(
                name="tool_argument_errors",
                value=float(self.argument_error_count),
                threshold=0.0,
                unit="count",
                passed=self.argument_error_count == 0,
                details={"argument_errors": self.argument_error_count},
            ),
            Metric(
                name="tool_repeated_calls",
                value=float(self.repeated_call_count),
                threshold=0.0,
                unit="count",
                passed=self.repeated_call_count == 0,
                details={"repeated_calls": self.repeated_call_count},
            ),
        ]


class AgentLoopPattern(BaseModel):
    """Identified repetitive loop or cycle pattern in an agent execution."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    loop_type: str = Field(
        ...,
        description="Type of loop: exact_repetition, alternating_cycle, failed_retry_loop",
    )
    cycle_length: int = Field(default=1, ge=1, description="Number of steps in repeating cycle")
    repetitions: int = Field(default=2, ge=2, description="Number of times the cycle repeated")
    tools_involved: list[str] = Field(
        default_factory=list, description="Tools participating in the loop"
    )
    description: str = Field(..., description="Human-readable description of the loop")


class AgentMetrics(BaseModel):
    """Aggregate metrics evaluating multi-step agent behavior, loops, and stability."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    total_steps: int = Field(default=0, ge=0, description="Total execution steps in agent run")
    llm_call_count: int = Field(default=0, ge=0, description="Total LLM generation calls")
    tool_call_count: int = Field(default=0, ge=0, description="Total tool call steps")
    has_loop: bool = Field(
        default=False, description="True if a repetitive execution loop was detected"
    )
    detected_loops: list[AgentLoopPattern] = Field(
        default_factory=list, description="Detected loop patterns"
    )
    is_exceeded_step_limit: bool = Field(
        default=False, description="True if step count exceeded configured limit"
    )
    max_step_limit: int = Field(default=10, ge=1, description="Configured maximum step threshold")
    tool_metrics: ToolMetrics = Field(
        default_factory=ToolMetrics, description="Aggregated tool metrics"
    )
    has_unresolved_error: bool = Field(
        default=False, description="True if final state or last step had error"
    )
    state_drift_detected: bool = Field(
        default=False, description="True if prompt grew excessively without progress"
    )

    def to_diagnostic_metrics(self) -> list[Metric]:
        """Convert agent analysis into standard diagnostic Metric instances."""
        metrics: list[Metric] = [
            Metric(
                name="agent_loop_detected",
                value=1.0 if self.has_loop else 0.0,
                threshold=0.0,
                unit="flag",
                passed=not self.has_loop,
                details={"loops_count": len(self.detected_loops)},
            ),
            Metric(
                name="agent_step_limit_exceeded",
                value=float(self.total_steps),
                threshold=float(self.max_step_limit),
                unit="steps",
                passed=not self.is_exceeded_step_limit,
                details={
                    "total_steps": self.total_steps,
                    "max_limit": self.max_step_limit,
                },
            ),
            Metric(
                name="agent_state_drift",
                value=1.0 if self.state_drift_detected else 0.0,
                threshold=0.0,
                unit="flag",
                passed=not self.state_drift_detected,
                details={"drift_detected": self.state_drift_detected},
            ),
        ]
        metrics.extend(self.tool_metrics.to_diagnostic_metrics())
        return metrics

    def to_evidence(self, span_id: str | None = None) -> list[Evidence]:
        """Convert agent failures into structured Evidence objects."""
        evidence_list: list[Evidence] = []

        if self.has_loop:
            for loop in self.detected_loops:
                evidence_list.append(
                    Evidence(
                        evidence_type=EvidenceType.TOOL_REPETITION,
                        description=f"Agent loop detected: {loop.description}",
                        supporting_data={
                            "loop_type": loop.loop_type,
                            "repetitions": loop.repetitions,
                            "tools_involved": loop.tools_involved,
                            "cycle_length": loop.cycle_length,
                        },
                        span_id=span_id,
                    )
                )

        if self.tool_metrics.failed_calls > 0:
            for ev in self.tool_metrics.evaluations:
                if ev.status == SpanStatus.ERROR or ev.error_message:
                    evidence_list.append(
                        Evidence(
                            evidence_type=EvidenceType.ERROR_LOG,
                            description=f"Tool execution failed for '{ev.tool_name}': {ev.error_message or 'Unknown error'}",
                            supporting_data={
                                "tool_name": ev.tool_name,
                                "call_id": ev.call_id,
                                "error": ev.error_message,
                            },
                            span_id=span_id,
                        )
                    )

        if self.tool_metrics.argument_error_count > 0:
            for ev in self.tool_metrics.evaluations:
                if ev.has_argument_error:
                    evidence_list.append(
                        Evidence(
                            evidence_type=EvidenceType.SCHEMA_VIOLATION,
                            description=f"Malformed tool call arguments for '{ev.tool_name}': {ev.argument_error_detail}",
                            supporting_data={
                                "tool_name": ev.tool_name,
                                "call_id": ev.call_id,
                                "detail": ev.argument_error_detail,
                            },
                            span_id=span_id,
                        )
                    )

        if self.state_drift_detected:
            evidence_list.append(
                Evidence(
                    evidence_type=EvidenceType.TOKEN_DISCREPANCY,
                    description="Agent state drift detected: repetitive steps without productive state advancement",
                    supporting_data={"total_steps": self.total_steps},
                    span_id=span_id,
                )
            )

        return evidence_list
