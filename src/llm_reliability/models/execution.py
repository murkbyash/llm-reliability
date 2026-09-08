"""Granular execution step models for LLM calls, retrievals, tools, and responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_reliability.models.enums import SpanStatus


class TokenUsage(BaseModel):
    """Token consumption and estimated cost tracking."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens consumed by input prompt")
    completion_tokens: int = Field(default=0, ge=0, description="Tokens generated in completion")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_usd: float | None = Field(default=None, ge=0.0, description="Estimated cost in USD")


class RetrievedDocument(BaseModel):
    """Single retrieved chunk/document within a retrieval operation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    doc_id: str = Field(description="Unique identifier or index of the document")
    content: str = Field(description="Text content of the retrieved chunk")
    score: float | None = Field(default=None, description="Similarity or relevance score")
    rank: int | None = Field(default=None, ge=1, description="Rank in retrieval candidate list")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary document metadata"
    )


class RetrievalStep(BaseModel):
    """Details of a vector search, hybrid retrieval, or document fetching step."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    query: str = Field(description="Search query string passed to retriever")
    documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="List of retrieved document chunks",
    )
    top_k: int | None = Field(default=None, ge=1, description="Requested candidate count")
    retriever_name: str | None = Field(default=None, description="Name or type of retriever used")
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="Retrieval latency in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Retriever configuration metadata"
    )


class LLMCall(BaseModel):
    """Details of a single LLM invocation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = Field(description="Name/identifier of the model invoked")
    prompt: str | list[dict[str, Any]] = Field(description="Prompt text or list of chat messages")
    response: str | None = Field(default=None, description="Raw model response text")
    temperature: float | None = Field(default=None, ge=0.0, description="Sampling temperature")
    token_usage: TokenUsage | None = Field(default=None, description="Token consumption breakdown")
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="Inference latency in milliseconds"
    )
    raw_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra generation parameters (top_p, max_tokens, etc.)",
    )


class ToolCall(BaseModel):
    """Invocation of an external tool or function by an agent or LLM."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tool_name: str = Field(description="Name of the tool/function called")
    arguments: dict[str, Any] | str = Field(
        default_factory=dict,
        description="Arguments passed to the tool (parsed dict or raw string)",
    )
    call_id: str | None = Field(default=None, description="Unique call ID if provided by model")


class ToolResult(BaseModel):
    """Result returned from an external tool or function execution."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tool_name: str = Field(description="Name of the tool that was executed")
    output: Any = Field(default=None, description="Payload returned by the tool")
    error: str | None = Field(default=None, description="Error message if tool execution failed")
    call_id: str | None = Field(default=None, description="Corresponding tool call ID")
    status: SpanStatus = Field(default=SpanStatus.SUCCESS, description="Execution status")
    latency_ms: float | None = Field(
        default=None, ge=0.0, description="Execution duration in milliseconds"
    )


class FinalResponse(BaseModel):
    """Final output synthesized by the LLM, RAG, or agent system."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    text: str = Field(description="Synthesized text response presented to user")
    grounded: bool | None = Field(default=None, description="Whether answer was verified grounded")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="System or model self-reported confidence score",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
