# Agent & Tool Failure Analysis

AI Agent systems and tool-calling models frequently encounter catastrophic trajectory failures including infinite tool loops, alternating ping-pong traps, schema validation errors, and step limit exhaustion.

---

## 1. Analyzed Agent Failure Modes

`llm_reliability.agent.AgentAnalyzer` inspects execution trajectories for:

1. **Repetitive Action Loops (`AGENT_LOOP`):**
   - Identical tool invocations with unchanged arguments executed $\ge 3$ consecutive times.
2. **Alternating Cycle Traps (`AGENT_LOOP`):**
   - 2-step or 3-step repeating cycle patterns (e.g. `ToolA` $\to$ `ToolB` $\to$ `ToolA` $\to$ `ToolB`).
3. **Failed Retry Cascades (`AGENT_LOOP`):**
   - Retrying the exact same failing tool execution without changing arguments or correcting error conditions.
4. **Tool Argument & Schema Violations (`SCHEMA_VIOLATION`):**
   - Invalid JSON strings, unparsed argument formats, missing required parameter keys, or type mismatches.
5. **Tool Execution Errors (`TOOL_ERROR`):**
   - Internal tool execution exceptions, network timeouts, database connection drops, or HTTP 500 errors.
6. **Step Limit / Budget Exhaustion (`AGENT_LOOP`):**
   - Exceeding configured maximum iteration limits (e.g. $> 10\text{ steps}$) without reaching a terminal response.
7. **Agent State Drift:**
   - Divergence from the original user goal across multi-step execution.

---

## 2. Agent Metrics Extracted

- `total_steps`: Number of steps/spans in the agent run.
- `tool_calls_count`: Total tool invocations.
- `tool_error_count`: Number of failed tool executions.
- `tool_failure_rate`: Proportion of tool errors ($0.0 \dots 1.0$).
- `loop_detected`: Boolean flag indicating repetitive loops.
- `detected_loops`: List of `AgentLoopPattern` objects detailing repeating tools, cycle length, and repetitions.
- `step_limit_exceeded`: Boolean flag indicating step budget overrun.
- `state_drift_detected`: Boolean flag indicating goal deviation.

---

## 3. Python Example

```python
from llm_reliability.agent import AgentAnalyzer
from llm_reliability.models.trace import Span, SpanKind, SpanStatus
from llm_reliability.models.execution import ToolCall, ToolResult

# Sequence of repetitive failing tool calls
spans = [
    Span(
        span_id=f"s{i}",
        name="execute_sql",
        kind=SpanKind.TOOL,
        status=SpanStatus.ERROR,
        tool_call=ToolCall(tool_name="execute_sql", arguments={"query": "SELECT * FROM orders"}),
        tool_result=ToolResult(tool_name="execute_sql", status=SpanStatus.ERROR, error="Timeout"),
    )
    for i in range(4)
]

analyzer = AgentAnalyzer()
metrics = analyzer.analyze_trajectory(spans)

print(f"Loop Detected: {metrics.loop_detected}")
print(f"Tool Failure Rate: {metrics.tool_metrics.failure_rate * 100:.1f}%")
for loop in metrics.detected_loops:
    print(f"Loop on '{loop.tool_name}' repeated {loop.repetitions} times!")
```

