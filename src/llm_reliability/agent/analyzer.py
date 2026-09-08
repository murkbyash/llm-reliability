"""Agent and tool failure analysis engine detecting loops, errors, and drift."""

import json
import statistics
from typing import Any

from llm_reliability.agent.models import (
    AgentLoopPattern,
    AgentMetrics,
    ToolCallEvaluation,
    ToolMetrics,
)
from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.trace import Run, Span, Trace


class AgentAnalyzer:
    """Deterministic analyzer evaluating multi-step agent trajectories and tool execution failures."""

    def __init__(
        self,
        max_step_limit: int = 10,
        max_repeated_calls_threshold: int = 2,
    ) -> None:
        """Initialize agent analyzer.

        Args:
            max_step_limit: Maximum expected steps before flagging excessive iteration.
            max_repeated_calls_threshold: Threshold for repeated tool invocations before flagging a loop.
        """
        self.max_step_limit = max_step_limit
        self.max_repeated_calls_threshold = max_repeated_calls_threshold

    def analyze_tool_calls(self, spans: list[Span]) -> ToolMetrics:
        """Analyze all tool calls within a span list for errors, repetitions, and argument integrity."""
        tool_spans = [
            s
            for s in spans
            if s.kind == SpanKind.TOOL or s.tool_call is not None or s.tool_result is not None
        ]

        if not tool_spans:
            return ToolMetrics()

        evaluations: list[ToolCallEvaluation] = []
        seen_calls: dict[str, list[dict[str, Any]]] = {}  # tool_name -> list of call history
        successful_count = 0
        failed_count = 0
        argument_error_count = 0
        repeated_call_count = 0
        latencies: list[float] = []

        for span in tool_spans:
            tool_name = "unknown_tool"
            call_id = None
            raw_args: Any = {}

            if span.tool_call is not None:
                tool_name = span.tool_call.tool_name
                call_id = span.tool_call.call_id
                raw_args = span.tool_call.arguments
            elif span.tool_result is not None:
                tool_name = span.tool_result.tool_name
                call_id = span.tool_result.call_id

            if tool_name == "unknown_tool" and span.name:
                tool_name = span.name

            # Validate arguments
            parsed_args, has_arg_error, arg_error_detail = self._parse_arguments(raw_args)
            if has_arg_error:
                argument_error_count += 1

            # Determine execution status and error message
            error_message = span.error_message
            status = span.status

            if span.tool_result is not None:
                if span.tool_result.error:
                    error_message = span.tool_result.error
                    status = SpanStatus.ERROR
                elif span.tool_result.status == SpanStatus.ERROR:
                    status = SpanStatus.ERROR

            if error_message or status == SpanStatus.ERROR:
                failed_count += 1
                status = SpanStatus.ERROR
            else:
                successful_count += 1
                status = SpanStatus.SUCCESS

            # Latency tracking
            duration = span.duration_ms
            if duration is None and span.tool_result and span.tool_result.latency_ms:
                duration = span.tool_result.latency_ms
            if duration is not None:
                latencies.append(duration)

            # Repetition and retry tracking
            canonical_args_str = self._canonical_args_repr(
                parsed_args if parsed_args is not None else raw_args
            )
            history = seen_calls.setdefault(tool_name, [])

            is_duplicate = False
            is_failed_retry = False

            if history:
                prev_call = history[-1]
                if prev_call["args_str"] == canonical_args_str:
                    is_duplicate = True
                    repeated_call_count += 1
                    if prev_call["failed"]:
                        is_failed_retry = True

            history.append(
                {
                    "args_str": canonical_args_str,
                    "failed": status == SpanStatus.ERROR,
                }
            )

            evaluations.append(
                ToolCallEvaluation(
                    tool_name=tool_name,
                    call_id=call_id,
                    status=status,
                    has_argument_error=has_arg_error,
                    argument_error_detail=arg_error_detail,
                    is_duplicate_call=is_duplicate,
                    is_failed_retry=is_failed_retry,
                    latency_ms=duration,
                    error_message=error_message,
                )
            )

        total_tool_calls = len(evaluations)
        failure_rate = round(failed_count / total_tool_calls, 4) if total_tool_calls > 0 else 0.0
        avg_latency = round(statistics.mean(latencies), 2) if latencies else None
        unique_tools = list(seen_calls.keys())

        return ToolMetrics(
            total_tool_calls=total_tool_calls,
            successful_calls=successful_count,
            failed_calls=failed_count,
            failure_rate=failure_rate,
            unique_tools_used=unique_tools,
            argument_error_count=argument_error_count,
            repeated_call_count=repeated_call_count,
            average_tool_latency_ms=avg_latency,
            evaluations=evaluations,
        )

    def detect_loops(self, spans: list[Span]) -> list[AgentLoopPattern]:
        """Detect repetitive loops, ping-pong alternating cycles, and failed retry loops."""
        tool_spans = [
            s
            for s in spans
            if s.kind == SpanKind.TOOL or s.tool_call is not None or s.tool_result is not None
        ]

        if len(tool_spans) < 2:
            return []

        patterns: list[AgentLoopPattern] = []

        # Represent sequence as list of (tool_name, args_str, is_error)
        sequence: list[tuple[str, str, bool]] = []
        for s in tool_spans:
            tool_name = (
                (s.tool_call.tool_name if s.tool_call else None)
                or (s.tool_result.tool_name if s.tool_result else None)
                or s.name
            )
            raw_args = s.tool_call.arguments if s.tool_call else {}
            args_str = self._canonical_args_repr(raw_args)
            is_error = (
                s.status == SpanStatus.ERROR
                or bool(s.error_message)
                or bool(s.tool_result and s.tool_result.error)
            )
            sequence.append((tool_name, args_str, is_error))

        # Check 1: Exact consecutive repetitions (Cycle length 1)
        consecutive_same_count = 1
        last_sig = sequence[0]

        for i in range(1, len(sequence)):
            if sequence[i][0] == last_sig[0] and sequence[i][1] == last_sig[1]:
                consecutive_same_count += 1
            else:
                if consecutive_same_count >= self.max_repeated_calls_threshold:
                    patterns.append(
                        AgentLoopPattern(
                            loop_type="exact_repetition",
                            cycle_length=1,
                            repetitions=consecutive_same_count,
                            tools_involved=[last_sig[0]],
                            description=(
                                f"Tool '{last_sig[0]}' called {consecutive_same_count} times consecutively "
                                f"with identical arguments"
                            ),
                        )
                    )
                last_sig = sequence[i]
                consecutive_same_count = 1

        if consecutive_same_count >= self.max_repeated_calls_threshold:
            patterns.append(
                AgentLoopPattern(
                    loop_type="exact_repetition",
                    cycle_length=1,
                    repetitions=consecutive_same_count,
                    tools_involved=[last_sig[0]],
                    description=(
                        f"Tool '{last_sig[0]}' called {consecutive_same_count} times consecutively "
                        f"with identical arguments"
                    ),
                )
            )

        # Check 2: Failed retry loops (calling same tool with identical args after failure)
        for i in range(1, len(sequence)):
            prev_tool, prev_args, prev_error = sequence[i - 1]
            curr_tool, curr_args, _ = sequence[i]
            if prev_error and prev_tool == curr_tool and prev_args == curr_args:
                patterns.append(
                    AgentLoopPattern(
                        loop_type="failed_retry_loop",
                        cycle_length=1,
                        repetitions=2,
                        tools_involved=[curr_tool],
                        description=f"Tool '{curr_tool}' failed and was retried with identical arguments without fixing error",
                    )
                )

        # Check 3: Alternating cycle detection (Cycle length 2, 3, etc.)
        tool_names = [item[0] for item in sequence]
        for cycle_len in [2, 3]:
            if len(tool_names) >= cycle_len * 2:
                # Sliding window search for periodic repeating patterns
                for start in range(len(tool_names) - cycle_len * 2 + 1):
                    unit = tool_names[start : start + cycle_len]
                    if len(set(unit)) == cycle_len:  # All distinct tools in unit
                        # Count repetitions
                        reps = 0
                        idx = start
                        while (
                            idx + cycle_len <= len(tool_names)
                            and tool_names[idx : idx + cycle_len] == unit
                        ):
                            reps += 1
                            idx += cycle_len

                        if reps >= 2:
                            cycle_desc = " -> ".join(unit)
                            pattern_obj = AgentLoopPattern(
                                loop_type="alternating_cycle",
                                cycle_length=cycle_len,
                                repetitions=reps,
                                tools_involved=unit,
                                description=f"Alternating tool cycle ({cycle_desc}) repeated {reps} times",
                            )
                            # Avoid adding duplicate pattern definitions
                            if not any(p.description == pattern_obj.description for p in patterns):
                                patterns.append(pattern_obj)

        return patterns

    def analyze_agent_run(self, run: Run) -> AgentMetrics:
        """Evaluate a complete agent execution trajectory."""
        spans = run.spans
        total_steps = len(spans)
        llm_calls = run.get_llm_calls()
        llm_count = len(llm_calls)

        tool_metrics = self.analyze_tool_calls(spans)
        detected_loops = self.detect_loops(spans)
        has_loop = len(detected_loops) > 0
        is_exceeded_step_limit = total_steps > self.max_step_limit

        # Unresolved error check
        has_unresolved_error = False
        if spans:
            last_span = spans[-1]
            if last_span.is_error or (last_span.tool_result and last_span.tool_result.error):
                has_unresolved_error = True
        if (
            tool_metrics.failed_calls > 0
            and tool_metrics.evaluations
            and tool_metrics.evaluations[-1].status == SpanStatus.ERROR
        ):
            has_unresolved_error = True

        # State drift detection (prompt explosion with low productivity)
        state_drift_detected = False
        if total_steps >= 4 and (has_loop or tool_metrics.repeated_call_count >= 2):
            state_drift_detected = True

        return AgentMetrics(
            total_steps=total_steps,
            llm_call_count=llm_count,
            tool_call_count=tool_metrics.total_tool_calls,
            has_loop=has_loop,
            detected_loops=detected_loops,
            is_exceeded_step_limit=is_exceeded_step_limit,
            max_step_limit=self.max_step_limit,
            tool_metrics=tool_metrics,
            has_unresolved_error=has_unresolved_error,
            state_drift_detected=state_drift_detected,
        )

    def analyze_trace(self, trace: Trace) -> list[AgentMetrics]:
        """Evaluate agent metrics across all runs in a trace."""
        return [self.analyze_agent_run(run) for run in trace.runs]

    def _parse_arguments(self, args: Any) -> tuple[dict[str, Any] | None, bool, str | None]:
        """Validate and parse tool arguments into a dictionary."""
        if isinstance(args, dict):
            return args, False, None

        if isinstance(args, str):
            cleaned = args.strip()
            if not cleaned:
                return {}, False, None
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed, False, None
                return None, True, f"JSON parsed to non-dict type: {type(parsed).__name__}"
            except json.JSONDecodeError as err:
                return None, True, f"Invalid JSON syntax in tool arguments: {err}"

        if args is None:
            return {}, False, None

        return None, True, f"Unsupported argument type: {type(args).__name__}"

    def _canonical_args_repr(self, args: Any) -> str:
        """Create a deterministic sorted string representation of arguments for exact equality checks."""
        if isinstance(args, dict):
            try:
                return json.dumps(args, sort_keys=True)
            except (TypeError, ValueError):
                return str(sorted(args.items()))
        return str(args)
