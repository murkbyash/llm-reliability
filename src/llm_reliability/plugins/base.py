"""Base interfaces and abstract classes for LLM Reliability Analyzer plugins."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_reliability.models.diagnosis import Diagnosis, Evidence, Failure, Metric
from llm_reliability.models.trace import Trace


class PluginMetadata(BaseModel):
    """Metadata describing a third-party plugin."""

    name: str = Field(description="Unique plugin identifier")
    version: str = Field(default="0.1.0", description="Plugin version string")
    author: str = Field(default="Unknown", description="Author or organization")
    description: str = Field(default="", description="Brief summary of plugin functionality")


class BasePlugin(ABC):
    """Abstract base class for all plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    def initialize(self) -> None:  # noqa: B027
        """Lifecycle hook called when plugin is registered."""

    def shutdown(self) -> None:  # noqa: B027
        """Lifecycle hook called when plugin is unloaded."""


class BaseAnalyzerPlugin(BasePlugin):
    """Abstract base class for third-party custom trace analyzers."""

    @abstractmethod
    def analyze(
        self, trace: Trace
    ) -> list[Failure] | tuple[list[Failure], list[Evidence], list[Metric]]:
        """Analyze an execution trace and return detected failures, evidence, and metrics."""
        pass


class BaseExporterPlugin(BasePlugin):
    """Abstract base class for third-party custom report and telemetry exporters."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Unique format name identifier (e.g. 'prometheus', 'datadog', 'slack')."""
        pass

    @abstractmethod
    def export(self, diagnosis: Diagnosis, output_path: Path | str | None = None) -> Any:
        """Export or format a diagnosis report to a target destination."""
        pass


class BaseMiddlewarePlugin(BasePlugin):
    """Abstract base class for trace and diagnosis interception middleware."""

    def before_diagnosis(self, trace: Trace) -> Trace:
        """Intercept and optionally modify or sanitize trace before diagnostic analysis."""
        return trace

    def after_diagnosis(self, diagnosis: Diagnosis) -> Diagnosis:
        """Intercept and optionally enrich or transform diagnosis after analysis completes."""
        return diagnosis
