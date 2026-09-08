"""Data models for streaming generation and token latency profiling."""

from pydantic import BaseModel, ConfigDict, Field


class TokenChunk(BaseModel):
    """Timing and payload for an individual streamed token or chunk."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    token_index: int = Field(ge=0, description="Sequential index of the token chunk")
    text: str = Field(description="Text segment emitted in this chunk")
    timestamp_ms: float = Field(ge=0.0, description="Offset timestamp from start in milliseconds")
    delta_ms: float = Field(
        ge=0.0, description="Duration elapsed since previous token in milliseconds"
    )


class StallEvent(BaseModel):
    """Details of an identified streaming stall or pause event."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    token_index: int = Field(ge=0, description="Token index where stall occurred")
    timestamp_ms: float = Field(ge=0.0, description="Timestamp when stall was detected")
    duration_ms: float = Field(ge=0.0, description="Duration of the stall in milliseconds")
    text: str = Field(default="", description="Token text emitted following the stall")


class InterTokenLatencyStats(BaseModel):
    """Statistical distribution of latency between consecutive token emissions."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    mean_ms: float = Field(ge=0.0, description="Mean inter-token latency in ms")
    median_ms: float = Field(ge=0.0, description="Median (p50) inter-token latency in ms")
    p90_ms: float = Field(ge=0.0, description="90th percentile inter-token latency in ms")
    p95_ms: float = Field(ge=0.0, description="95th percentile inter-token latency in ms")
    p99_ms: float = Field(ge=0.0, description="99th percentile inter-token latency in ms")
    min_ms: float = Field(ge=0.0, description="Minimum inter-token latency in ms")
    max_ms: float = Field(ge=0.0, description="Maximum inter-token latency in ms")
    std_dev_ms: float = Field(ge=0.0, description="Standard deviation of inter-token latency")
    variance_ms: float = Field(ge=0.0, description="Variance of inter-token latency")
    jitter_ms: float = Field(
        ge=0.0, description="Mean absolute difference between consecutive token intervals"
    )


class TokenLatencyProfile(BaseModel):
    """Comprehensive token generation latency and throughput profile."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    total_chunks: int = Field(ge=0, description="Total number of chunk events received")
    total_tokens: int = Field(ge=0, description="Estimated total tokens emitted")
    total_duration_ms: float = Field(ge=0.0, description="Total stream duration in milliseconds")
    time_to_first_token_ms: float = Field(
        ge=0.0, description="Time elapsed until first token was received (TTFT) in ms"
    )
    tokens_per_second: float = Field(
        ge=0.0, description="Generation throughput rate (tokens per second)"
    )
    inter_token_latency: InterTokenLatencyStats = Field(
        description="Statistical breakdown of inter-token latencies"
    )
    stall_count: int = Field(ge=0, description="Number of stalls exceeding stall threshold")
    max_stall_duration_ms: float = Field(
        ge=0.0, description="Duration of longest stall in milliseconds"
    )
    stalls: list[StallEvent] = Field(
        default_factory=list, description="List of detected stall events"
    )
    timeline: list[TokenChunk] = Field(
        default_factory=list, description="Full chronological chunk emission timeline"
    )
