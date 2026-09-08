"""Tests for Streaming & Token Latency Profiler."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from llm_reliability import (
    EvidenceType,
    InterTokenLatencyStats,
    StreamingProfiler,
    TokenLatencyProfile,
    wrap_async_stream,
    wrap_stream,
)


class TestStreamingProfiler:
    """Deterministic validation of streaming token latency profiling, throughput, and stalling detection."""

    def test_streaming_profiler_statistics_and_ttft(self) -> None:
        profiler = StreamingProfiler(stall_threshold_ms=400.0)
        profiler.start(start_time_s=100.0)

        # First token arrives at t = 100.200 (TTFT = 200ms)
        profiler.record_chunk("Hello", timestamp_s=100.200)
        # Token 2 arrives at t = 100.250 (delta = 50ms)
        profiler.record_chunk(" world", timestamp_s=100.250)
        # Token 3 arrives at t = 100.310 (delta = 60ms)
        profiler.record_chunk(",", timestamp_s=100.310)
        # Token 4 arrives at t = 100.360 (delta = 50ms)
        profiler.record_chunk(" how", timestamp_s=100.360)
        # Token 5 arrives at t = 100.440 (delta = 80ms)
        profiler.record_chunk(" are", timestamp_s=100.440)
        # Token 6 arrives at t = 100.500 (delta = 60ms)
        profiler.record_chunk(" you", timestamp_s=100.500)

        profile = profiler.finish(end_time_s=100.500)

        assert isinstance(profile, TokenLatencyProfile)
        assert profile.total_chunks == 6
        assert profile.total_tokens == 6
        assert profile.time_to_first_token_ms == 200.0
        assert profile.total_duration_ms == 500.0

        # Generation duration after first token = 100.500 - 100.200 = 0.300s for 5 tokens -> TPS = 5 / 0.3 = 16.67
        assert profile.tokens_per_second == 16.67

        # Inter-token latencies: [50, 60, 50, 80, 60] -> mean = 60.0ms, median = 60.0ms, min = 50, max = 80
        stats = profile.inter_token_latency
        assert isinstance(stats, InterTokenLatencyStats)
        assert stats.mean_ms == 60.0
        assert stats.median_ms == 60.0
        assert stats.min_ms == 50.0
        assert stats.max_ms == 80.0
        assert stats.p90_ms == 80.0
        assert profile.stall_count == 0

    def test_streaming_stall_detection(self) -> None:
        profiler = StreamingProfiler(stall_threshold_ms=300.0)
        profiler.start(start_time_s=0.0)

        # Chunks with a 600ms stall between token 2 and 3
        profiler.record_chunk("Step 1:", timestamp_s=0.100)  # TTFT = 100ms
        profiler.record_chunk(" loading", timestamp_s=0.150)  # delta = 50ms
        profiler.record_chunk(" data", timestamp_s=0.750)  # delta = 600ms (STALL > 300ms)
        profiler.record_chunk(" complete.", timestamp_s=0.800)  # delta = 50ms

        profile = profiler.finish(end_time_s=0.800)

        assert profile.stall_count == 1
        assert profile.max_stall_duration_ms == 600.0
        assert len(profile.stalls) == 1
        stall = profile.stalls[0]
        assert stall.token_index == 2
        assert stall.duration_ms == 600.0
        assert stall.text == " data"

        evidences = profiler.to_evidence()
        assert len(evidences) >= 1
        assert any(e.evidence_type == EvidenceType.LATENCY_SPIKE for e in evidences)
        assert "600.0ms" in evidences[0].description

    def test_streaming_empty_and_single_token(self) -> None:
        # 0 tokens
        profiler_empty = StreamingProfiler()
        profiler_empty.start(start_time_s=10.0)
        profile_empty = profiler_empty.finish(end_time_s=10.050)
        assert profile_empty.total_chunks == 0
        assert profile_empty.total_tokens == 0
        assert profile_empty.tokens_per_second == 0.0

        # 1 token
        profiler_single = StreamingProfiler()
        profiler_single.start(start_time_s=20.0)
        profiler_single.record_chunk("OnlyToken", timestamp_s=20.150)
        profile_single = profiler_single.finish(end_time_s=20.150)
        assert profile_single.total_chunks == 1
        assert profile_single.time_to_first_token_ms == 150.0
        assert profile_single.inter_token_latency.mean_ms == 0.0
        assert len(profile_single.timeline) == 1
        assert profile_single.timeline[0].text == "OnlyToken"

    def test_wrap_stream_sync_generator(self) -> None:
        def sample_generator() -> Iterator[str]:
            yield "Alpha"
            yield " Beta"
            yield " Gamma"

        captured_profile: TokenLatencyProfile | None = None
        captured_text: str | None = None

        def on_done(p: TokenLatencyProfile, text: str) -> None:
            nonlocal captured_profile, captured_text
            captured_profile = p
            captured_text = text

        wrapped: Iterator[str] = wrap_stream(sample_generator(), on_complete=on_done)
        chunks_received = list(wrapped)

        assert chunks_received == ["Alpha", " Beta", " Gamma"]
        assert captured_text == "Alpha Beta Gamma"
        assert captured_profile is not None
        assert captured_profile.total_chunks == 3
        assert len(captured_profile.timeline) == 3

    def test_wrap_async_stream_async_generator(self) -> None:
        async def _run() -> None:
            async def sample_async_gen() -> AsyncIterator[dict[str, Any]]:
                yield {"text": "Hello"}
                yield {"text": " from"}
                yield {"text": " async"}

            captured_profile: TokenLatencyProfile | None = None
            captured_text: str | None = None

            def on_done(p: TokenLatencyProfile, text: str) -> None:
                nonlocal captured_profile, captured_text
                captured_profile = p
                captured_text = text

            stream: AsyncIterator[dict[str, Any]] = wrap_async_stream(
                sample_async_gen(), on_complete=on_done
            )
            chunks_collected: list[dict[str, Any]] = []
            async for c in stream:
                chunks_collected.append(c)

            assert len(chunks_collected) == 3
            assert captured_text == "Hello from async"
            assert captured_profile is not None
            assert captured_profile.total_chunks == 3

        asyncio.run(_run())

    def test_to_diagnostic_metrics_and_evidence(self) -> None:
        profiler = StreamingProfiler(stall_threshold_ms=250.0)
        profiler.start(start_time_s=10.0)
        profiler.record_chunk(
            "First token", timestamp_s=13.500
        )  # TTFT = 3500ms (> 3000ms triggers evidence)
        profiler.record_chunk(" second", timestamp_s=13.600)
        profiler.record_chunk(" stalled token", timestamp_s=14.100)  # 500ms stall

        metrics = profiler.to_diagnostic_metrics()
        metric_dict = {m.name: m.value for m in metrics}

        assert metric_dict["time_to_first_token_ms"] == 3500.0
        assert metric_dict["token_stall_count"] == 1.0
        assert metric_dict["max_stall_duration_ms"] == 500.0
        assert "mean_inter_token_latency_ms" in metric_dict
        assert "tokens_per_second" in metric_dict

        evidences = profiler.to_evidence()
        assert len(evidences) == 2  # Both TTFT spike and stall spike
        desc_joined = " ".join(e.description for e in evidences)
        assert "High Time-To-First-Token" in desc_joined
        assert "streaming stall" in desc_joined
