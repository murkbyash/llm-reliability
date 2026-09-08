# Streaming & Token Latency Profiler

In production streaming architectures (e.g. Chatbots, SSE endpoints, WebSockets), latency bottlenecks manifest as slow initial time-to-first-token (TTFT) or mid-stream pauses (stalls) caused by server-side memory reallocation or rate throttling.

`llm_reliability.streaming` measures fine-grained token latency distributions and generation stalls.

---

## 1. Measured Streaming Metrics

`StreamingProfiler` extracts:
- `time_to_first_token_ms` (TTFT): Time elapsed from stream initiation to arrival of the first token chunk.
- `tokens_per_second` (TPS): Effective token emission throughput.
- `inter_token_latency`: Statistical breakdown (mean, median/p50, p90, p95, p99, min, max, std dev, variance, jitter).
- `stall_count`: Number of inter-token intervals exceeding the stall threshold (default: $> 500\text{ ms}$).
- `max_stall_duration_ms`: Duration of the longest stall event.
- `stalls`: Chronological list of `StallEvent` instances with timestamps and emitted text.

---

## 2. Sync and Async Generator Wrappers

Transparently wrap generators to profile chunk emission in the background:

```python
from llm_reliability import wrap_stream, wrap_async_stream, TokenLatencyProfile


def my_sync_generator():
    yield "Hello"
    yield " world"


def on_stream_done(profile: TokenLatencyProfile, full_text: str):
    print(f"Captured text: {full_text}")
    print(f"TTFT: {profile.time_to_first_token_ms:.1f}ms")
    print(f"Median Inter-Token: {profile.inter_token_latency.median_ms:.1f}ms")
    print(f"Stall Count: {profile.stall_count}")


# Wrap synchronous stream
stream = wrap_stream(my_sync_generator(), on_complete=on_stream_done, stall_threshold_ms=300.0)
for chunk in stream:
    print(chunk, end="")
```

