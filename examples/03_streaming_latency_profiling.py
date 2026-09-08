"""Example 03: High-Precision Token Streaming Latency Profiling and Stall Detection."""

import time

from llm_reliability import StreamingProfiler, wrap_stream


def main() -> None:
    print("=" * 70)
    print("Example 03: Token Streaming Latency & Mid-Stream Stall Profiling")
    print("=" * 70)

    # 1. Simulate an LLM streaming generator with TTFT and a mid-stream stall
    def simulate_token_stream():
        # TTFT: 180ms
        time.sleep(0.180)
        yield "LLM "
        # Normal inter-token deltas: 30-50ms
        time.sleep(0.040)
        yield "Reliability "
        time.sleep(0.035)
        yield "Analyzer "
        # Stall: 320ms pause (> 250ms threshold)
        time.sleep(0.320)
        yield "detects "
        time.sleep(0.045)
        yield "streaming "
        time.sleep(0.040)
        yield "stalls!"

    profiler = StreamingProfiler(stall_threshold_ms=250.0)

    # 2. Wrap streaming generator
    wrapped = wrap_stream(
        simulate_token_stream(),
        profiler=profiler,
        stall_threshold_ms=250.0,
    )

    print("\n[1] Streaming Output in Real-Time:")
    print("    >>> ", end="")
    for chunk in wrapped:
        print(chunk, end="", flush=True)
    print("\n")

    # 3. Inspect Profiler Metrics
    profile = profiler.profile
    if profile:
        print("[2] Streaming Latency Summary:")
        print(f"    * Time to First Token (TTFT) : {profile.time_to_first_token_ms:.1f} ms")
        print(f"    * Generation Throughput     : {profile.tokens_per_second:.1f} tokens/sec")
        print(f"    * Total Tokens Emitted       : {profile.total_tokens}")
        print(f"    * Median Inter-Token Latency: {profile.inter_token_latency.median_ms:.1f} ms")
        print(f"    * 95th Percentile Latency   : {profile.inter_token_latency.p95_ms:.1f} ms")
        print(f"    * Inter-Token Jitter        : {profile.inter_token_latency.jitter_ms:.1f} ms")
        print(f"    * Stalls Detected (>250ms)   : {profile.stall_count}")

        if profile.stalls:
            print("\n[3] Detected Stalling Events:")
            for s in profile.stalls:
                print(
                    f"    [!] Token #{s.token_index} ('{s.text.strip()}') paused for {s.duration_ms:.1f} ms!"
                )
    print("=" * 70)


if __name__ == "__main__":
    main()
