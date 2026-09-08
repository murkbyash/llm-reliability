"""Streaming profiler for measuring token latency, throughput, and generation stalls."""

import math
import statistics
import time

from llm_reliability.models.diagnosis import Evidence, Metric
from llm_reliability.models.enums import EvidenceType
from llm_reliability.streaming.models import (
    InterTokenLatencyStats,
    StallEvent,
    TokenChunk,
    TokenLatencyProfile,
)


class StreamingProfiler:
    """Measures and computes statistical distributions of streaming token emission."""

    def __init__(
        self,
        stall_threshold_ms: float = 500.0,
        target_tps: float | None = None,
    ) -> None:
        self.stall_threshold_ms = max(10.0, float(stall_threshold_ms))
        self.target_tps = target_tps
        self._start_ticks: float | None = None
        self._last_chunk_ticks: float | None = None
        self._chunks: list[TokenChunk] = []
        self._stalls: list[StallEvent] = []
        self._completed_profile: TokenLatencyProfile | None = None

    def start(self, start_time_s: float | None = None) -> None:
        """Initialize or reset streaming timer."""
        self._start_ticks = start_time_s if start_time_s is not None else time.perf_counter()
        self._last_chunk_ticks = self._start_ticks
        self._chunks.clear()
        self._stalls.clear()
        self._completed_profile = None

    def record_chunk(self, text: str, timestamp_s: float | None = None) -> TokenChunk:
        """Record an arriving token or text chunk."""
        if self._start_ticks is None:
            self.start(timestamp_s)

        assert self._start_ticks is not None
        current_ticks = timestamp_s if timestamp_s is not None else time.perf_counter()
        assert self._last_chunk_ticks is not None

        elapsed_from_start_ms = max(0.0, (current_ticks - self._start_ticks) * 1000.0)
        delta_from_prev_ms = max(0.0, (current_ticks - self._last_chunk_ticks) * 1000.0)

        chunk_idx = len(self._chunks)
        chunk = TokenChunk(
            token_index=chunk_idx,
            text=text,
            timestamp_ms=round(elapsed_from_start_ms, 3),
            delta_ms=round(delta_from_prev_ms, 3),
        )
        self._chunks.append(chunk)

        # Detect stall (only for subsequent chunks after the first token)
        if chunk_idx > 0 and delta_from_prev_ms >= self.stall_threshold_ms:
            stall = StallEvent(
                token_index=chunk_idx,
                timestamp_ms=round(elapsed_from_start_ms, 3),
                duration_ms=round(delta_from_prev_ms, 3),
                text=text,
            )
            self._stalls.append(stall)

        self._last_chunk_ticks = current_ticks
        return chunk

    def finish(self, end_time_s: float | None = None) -> TokenLatencyProfile:
        """Finalize profiling and generate the comprehensive TokenLatencyProfile."""
        if self._start_ticks is None:
            self.start()

        assert self._start_ticks is not None
        current_ticks = end_time_s if end_time_s is not None else time.perf_counter()
        total_duration_ms = max(0.0, (current_ticks - self._start_ticks) * 1000.0)

        total_chunks = len(self._chunks)
        total_tokens = total_chunks  # Approximate 1 chunk per token

        if total_chunks == 0:
            empty_stats = InterTokenLatencyStats(
                mean_ms=0.0,
                median_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_dev_ms=0.0,
                variance_ms=0.0,
                jitter_ms=0.0,
            )
            profile = TokenLatencyProfile(
                total_chunks=0,
                total_tokens=0,
                total_duration_ms=round(total_duration_ms, 3),
                time_to_first_token_ms=round(total_duration_ms, 3),
                tokens_per_second=0.0,
                inter_token_latency=empty_stats,
                stall_count=0,
                max_stall_duration_ms=0.0,
                stalls=[],
                timeline=[],
            )
            self._completed_profile = profile
            return profile

        # First chunk determines TTFT
        ttft_ms = self._chunks[0].timestamp_ms

        # Inter-token latencies for subsequent chunks (chunks 1..N)
        inter_latencies = [c.delta_ms for c in self._chunks[1:]]

        if not inter_latencies:
            stats = InterTokenLatencyStats(
                mean_ms=0.0,
                median_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_dev_ms=0.0,
                variance_ms=0.0,
                jitter_ms=0.0,
            )
        else:
            sorted_latencies = sorted(inter_latencies)
            n = len(sorted_latencies)
            mean_val = statistics.mean(sorted_latencies)
            median_val = statistics.median(sorted_latencies)
            min_val = min(sorted_latencies)
            max_val = max(sorted_latencies)
            variance_val = statistics.variance(sorted_latencies) if n > 1 else 0.0
            std_dev_val = math.sqrt(variance_val)

            # Percentiles
            def _percentile(data: list[float], pct: float) -> float:
                idx = int(math.ceil(pct * len(data))) - 1
                return data[max(0, min(idx, len(data) - 1))]

            p90_val = _percentile(sorted_latencies, 0.90)
            p95_val = _percentile(sorted_latencies, 0.95)
            p99_val = _percentile(sorted_latencies, 0.99)

            # Jitter: mean absolute successive difference
            if len(inter_latencies) > 1:
                jitter_val = statistics.mean(
                    abs(inter_latencies[i] - inter_latencies[i - 1])
                    for i in range(1, len(inter_latencies))
                )
            else:
                jitter_val = 0.0

            stats = InterTokenLatencyStats(
                mean_ms=round(mean_val, 3),
                median_ms=round(median_val, 3),
                p90_ms=round(p90_val, 3),
                p95_ms=round(p95_val, 3),
                p99_ms=round(p99_val, 3),
                min_ms=round(min_val, 3),
                max_ms=round(max_val, 3),
                std_dev_ms=round(std_dev_val, 3),
                variance_ms=round(variance_val, 3),
                jitter_ms=round(jitter_val, 3),
            )

        # Tokens per second throughput
        # If multiple tokens, compute rate from first token to last token
        if total_chunks > 1:
            stream_gen_duration_s = (self._chunks[-1].timestamp_ms - ttft_ms) / 1000.0
            if stream_gen_duration_s > 0.0:
                tps = round((total_chunks - 1) / stream_gen_duration_s, 2)
            else:
                tps = round(float(total_chunks), 2)
        elif total_duration_ms > 0.0:
            tps = round(1.0 / (total_duration_ms / 1000.0), 2)
        else:
            tps = 0.0

        max_stall = max([s.duration_ms for s in self._stalls], default=0.0)

        profile = TokenLatencyProfile(
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            total_duration_ms=round(total_duration_ms, 3),
            time_to_first_token_ms=round(ttft_ms, 3),
            tokens_per_second=tps,
            inter_token_latency=stats,
            stall_count=len(self._stalls),
            max_stall_duration_ms=round(max_stall, 3),
            stalls=list(self._stalls),
            timeline=list(self._chunks),
        )
        self._completed_profile = profile
        return profile

    @property
    def profile(self) -> TokenLatencyProfile:
        """Convenience property to access the TokenLatencyProfile."""
        return self.get_profile()

    def get_profile(self) -> TokenLatencyProfile:
        """Get the current profile or finalize if active."""
        if self._completed_profile is not None:
            return self._completed_profile
        return self.finish()

    def to_diagnostic_metrics(self) -> list[Metric]:
        """Convert profile into canonical Metric models."""
        p = self.get_profile()
        return [
            Metric(name="time_to_first_token_ms", value=p.time_to_first_token_ms),
            Metric(name="tokens_per_second", value=p.tokens_per_second),
            Metric(name="mean_inter_token_latency_ms", value=p.inter_token_latency.mean_ms),
            Metric(name="median_inter_token_latency_ms", value=p.inter_token_latency.median_ms),
            Metric(name="p95_inter_token_latency_ms", value=p.inter_token_latency.p95_ms),
            Metric(name="max_inter_token_latency_ms", value=p.inter_token_latency.max_ms),
            Metric(name="token_stall_count", value=float(p.stall_count)),
            Metric(name="max_stall_duration_ms", value=p.max_stall_duration_ms),
            Metric(name="token_latency_jitter_ms", value=p.inter_token_latency.jitter_ms),
        ]

    def to_evidence(self) -> list[Evidence]:
        """Generate structured evidence items describing streaming latency and stalls."""
        p = self.get_profile()
        evidences: list[Evidence] = []

        if p.stall_count > 0:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.LATENCY_SPIKE,
                    description=(
                        f"Detected {p.stall_count} streaming stall(s) exceeding {self.stall_threshold_ms:.1f}ms "
                        f"(maximum stall duration: {p.max_stall_duration_ms:.1f}ms)."
                    ),
                    supporting_data={
                        "stall_count": float(p.stall_count),
                        "max_stall_duration_ms": p.max_stall_duration_ms,
                        "stall_threshold_ms": self.stall_threshold_ms,
                    },
                )
            )

        if p.time_to_first_token_ms > 3000.0:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.LATENCY_SPIKE,
                    description=f"High Time-To-First-Token (TTFT) observed: {p.time_to_first_token_ms:.1f}ms.",
                    supporting_data={"time_to_first_token_ms": p.time_to_first_token_ms},
                )
            )

        return evidences
