"""Data models for Multi-Modal LLM analysis and Structured Output schema verification."""

from pydantic import BaseModel, Field

from llm_reliability.models.diagnosis import Evidence, Failure, Metric


class StructuredOutputMetrics(BaseModel):
    """Metrics assessing JSON formatting, schema conformance, and structural constraints."""

    is_valid_json: bool = Field(description="True if output is well-formed parseable JSON")
    schema_matched: bool = Field(
        default=True, description="True if output satisfies all expected JSON Schema constraints"
    )
    missing_required_keys: list[str] = Field(
        default_factory=list, description="Mandatory schema fields missing from output"
    )
    type_mismatches: list[str] = Field(
        default_factory=list,
        description="Fields with unexpected types (e.g. expected int, got str)",
    )
    unexpected_keys: list[str] = Field(
        default_factory=list, description="Additional properties present not allowed by schema"
    )
    json_parse_error: str | None = Field(
        default=None, description="Syntax error message if JSON parsing failed"
    )
    schema_violation_count: int = Field(
        default=0, ge=0, description="Total count of structural schema errors"
    )


class MultiModalMetrics(BaseModel):
    """Metrics assessing vision/multi-modal inputs, media references, and visual grounding."""

    image_count: int = Field(default=0, ge=0, description="Total number of media/image inputs")
    has_unresolved_media_reference: bool = Field(
        default=False,
        description="True if prompt references images/media that are absent or invalid",
    )
    unresolved_references: list[str] = Field(
        default_factory=list,
        description="List of unresolved, broken, or out-of-bounds media references",
    )
    visual_grounding_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for visual alignment and grounding",
    )
    media_format_errors: list[str] = Field(
        default_factory=list,
        description="Invalid MIME types, corrupt base64, or unsupported formats",
    )


class MultiModalAnalysisResult(BaseModel):
    """Container with detected failures, metrics, and evidence for multi-modal/structured evaluations."""

    failures: list[Failure] = Field(default_factory=list, description="Detected failure modes")
    metrics: list[Metric] = Field(default_factory=list, description="Computed metrics")
    evidence: list[Evidence] = Field(default_factory=list, description="Supporting evidence items")
    structured_metrics: StructuredOutputMetrics | None = Field(
        default=None, description="Detailed structured output metrics if evaluated"
    )
    multimodal_metrics: MultiModalMetrics | None = Field(
        default=None, description="Detailed multi-modal metrics if evaluated"
    )
