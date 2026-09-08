"""Data models and session state for the local diagnostic visualizer server."""

from pathlib import Path

from pydantic import BaseModel, Field

from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.trace import Trace


class ServerConfig(BaseModel):
    """Configuration options for Diagnostic visualizer HTTP server."""

    host: str = Field(default="127.0.0.1", description="Hostname or IP to bind server to")
    port: int = Field(default=8080, description="Port number to listen on")
    trace_dir: Path | None = Field(
        default=None, description="Optional path to directory of trace files to preload"
    )
    auto_open: bool = Field(
        default=False, description="Whether to automatically launch default browser on startup"
    )


class TraceRecord(BaseModel):
    """Container associating a normalized Trace with its computed Diagnosis."""

    trace_id: str
    trace: Trace
    diagnosis: Diagnosis
    created_at: str
    source_filename: str | None = None
