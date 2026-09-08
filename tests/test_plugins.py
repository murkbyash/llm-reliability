"""Automated unit tests for plugins and extensibility SDK."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability import (
    BaseAnalyzerPlugin,
    BaseExporterPlugin,
    BaseMiddlewarePlugin,
    DiagnosticEngine,
    Evidence,
    EvidenceType,
    Failure,
    FailureCategory,
    Metric,
    PluginManager,
    PluginMetadata,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    get_plugin_manager,
    register_plugin,
)
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.trace import Run


class SampleCostAnalyzerPlugin(BaseAnalyzerPlugin):
    """Custom analyzer checking for expensive LLM calls exceeding token thresholds."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="cost-analyzer",
            version="1.0.0",
            author="Enterprise FinOps",
            description="Detects excessive token expenditures",
        )

    def analyze(
        self, trace: Trace
    ) -> list[Failure] | tuple[list[Failure], list[Evidence], list[Metric]]:
        failures: list[Failure] = []
        evidence: list[Evidence] = []
        metrics: list[Metric] = []

        total_tokens = 0
        for run in trace.runs:
            for span in run.spans:
                if span.attributes.get("token_count"):
                    total_tokens += int(span.attributes["token_count"])

        if total_tokens > 4000:
            ev = Evidence(
                evidence_type=EvidenceType.METRIC_THRESHOLD,
                description=f"Total tokens ({total_tokens}) exceeded FinOps threshold (4000).",
                supporting_data={"total_tokens": total_tokens},
            )
            evidence.append(ev)
            failures.append(
                Failure(
                    category=FailureCategory.TOKEN_ANOMALY,
                    severity=Severity.HIGH,
                    title="FinOps Token Budget Exceeded",
                    description=f"Execution consumed {total_tokens} tokens.",
                    message=f"Execution consumed {total_tokens} tokens.",
                    evidence=[ev],
                )
            )

        return failures, evidence, metrics


class PrometheusExporterPlugin(BaseExporterPlugin):
    """Custom exporter formatting diagnosis into Prometheus text metrics."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="prometheus-exporter",
            version="1.0.0",
            author="Observability Team",
            description="Exports Prometheus gauge metrics",
        )

    @property
    def format_name(self) -> str:
        return "prometheus"

    def export(self, diagnosis: Diagnosis, output_path: Path | str | None = None) -> str:
        lines = [
            "# HELP llm_reliability_failures_total Total count of detected failures",
            "# TYPE llm_reliability_failures_total gauge",
            f'llm_reliability_failures_total{{trace_id="{diagnosis.trace_id or "unknown"}"}} {len(diagnosis.failures)}',
            "# HELP llm_reliability_confidence Root cause confidence score",
            "# TYPE llm_reliability_confidence gauge",
            f'llm_reliability_confidence{{category="{diagnosis.primary_category.value}"}} {diagnosis.confidence:.2f}',
        ]
        text_output = "\n".join(lines) + "\n"
        if output_path is not None:
            Path(output_path).write_text(text_output, encoding="utf-8")
        return text_output


class TaggingMiddlewarePlugin(BaseMiddlewarePlugin):
    """Custom middleware tagging trace metadata before diagnosis and adding summary note after."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="tagging-middleware",
            version="1.0.0",
            author="DevSecOps",
            description="Tags traces with environment metadata",
        )

    def before_diagnosis(self, trace: Trace) -> Trace:
        trace_copy = trace.model_copy(deep=True)
        trace_copy.metadata["inspected_by"] = "tagging-middleware"
        return trace_copy

    def after_diagnosis(self, diagnosis: Diagnosis) -> Diagnosis:
        diag_copy = diagnosis.model_copy(deep=True)
        if diag_copy.summary:
            diag_copy.summary += " [Audited]"
        return diag_copy


class BrokenBuggyPlugin(BaseAnalyzerPlugin, BaseMiddlewarePlugin):
    """Buggy plugin intentionally throwing uncaught exceptions to test fault isolation."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(name="buggy-plugin", version="0.0.1")

    def analyze(
        self, trace: Trace
    ) -> list[Failure] | tuple[list[Failure], list[Evidence], list[Metric]]:
        raise RuntimeError("Intentional crash inside analyzer plugin!")

    def before_diagnosis(self, trace: Trace) -> Trace:
        raise ValueError("Intentional crash in pre-middleware!")

    def after_diagnosis(self, diagnosis: Diagnosis) -> Diagnosis:
        raise KeyError("Intentional crash in post-middleware!")


class TestPluginSystem:
    """Test custom analyzer, exporter, and middleware plugins with PluginManager."""

    @pytest.fixture(autouse=True)
    def clean_global_manager(self) -> Any:
        mgr = get_plugin_manager()
        mgr.clear()
        yield
        mgr.clear()

    def test_custom_analyzer_plugin_execution(self) -> None:
        manager = PluginManager()
        analyzer = SampleCostAnalyzerPlugin()
        manager.register(analyzer)

        engine = DiagnosticEngine(plugin_manager=manager)

        span = Span(
            span_id="s1",
            name="expensive_llm",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            attributes={"token_count": 5500},
        )
        run = Run(run_id="r1", trace_id="t1", spans=[span])
        trace = Trace(trace_id="t1", runs=[run])

        diagnosis = engine.diagnose(trace)
        assert len(diagnosis.failures) == 1
        assert diagnosis.failures[0].title == "FinOps Token Budget Exceeded"
        assert diagnosis.primary_category == FailureCategory.TOKEN_ANOMALY
        assert any("FinOps threshold" in ev.description for ev in diagnosis.evidence)

    def test_custom_exporter_plugin(self, tmp_path: Path) -> None:
        manager = PluginManager()
        exporter = PrometheusExporterPlugin()
        manager.register(exporter)

        diagnosis = Diagnosis(
            trace_id="trace-abc-123",
            primary_category=FailureCategory.RETRIEVAL_FAILURE,
            confidence=0.92,
            failures=[
                Failure(
                    category=FailureCategory.RETRIEVAL_FAILURE,
                    description="Retrieval error",
                )
            ],
            summary="Retrieval failed",
        )

        out_file = tmp_path / "metrics.prom"
        exported = manager.export(diagnosis, "prometheus", out_file)
        assert "llm_reliability_failures_total" in exported
        assert 'trace_id="trace-abc-123"' in exported
        assert out_file.is_file()
        assert out_file.read_text(encoding="utf-8") == exported

        with pytest.raises(ValueError, match="No exporter plugin registered"):
            manager.export(diagnosis, "datadog")

    def test_middleware_lifecycle_hooks(self) -> None:
        manager = PluginManager()
        mw = TaggingMiddlewarePlugin()
        manager.register(mw)

        engine = DiagnosticEngine(plugin_manager=manager)

        span = Span(span_id="s1", name="step", kind=SpanKind.ROOT, status=SpanStatus.SUCCESS)
        trace = Trace(trace_id="t1", runs=[Run(run_id="r1", trace_id="t1", spans=[span])])

        diagnosis = engine.diagnose(trace)
        assert diagnosis.summary is not None
        assert diagnosis.summary.endswith("[Audited]")

    def test_error_isolation_boundary(self) -> None:
        manager = PluginManager()
        buggy = BrokenBuggyPlugin()
        manager.register(buggy)

        engine = DiagnosticEngine(plugin_manager=manager)

        span = Span(span_id="s1", name="step", kind=SpanKind.ROOT, status=SpanStatus.SUCCESS)
        trace = Trace(trace_id="t1", runs=[Run(run_id="r1", trace_id="t1", spans=[span])])

        # Engine must not crash even though buggy plugin threw exceptions in all hooks
        diagnosis = engine.diagnose(trace)
        assert diagnosis is not None
        assert diagnosis.primary_category == FailureCategory.NONE

    def test_plugin_registration_management(self) -> None:
        manager = PluginManager()
        analyzer = SampleCostAnalyzerPlugin()
        manager.register(analyzer)

        assert len(manager.list_plugins()) == 1
        assert manager.get_plugin("cost-analyzer") is analyzer

        manager.unregister("cost-analyzer")
        assert len(manager.list_plugins()) == 0
        assert manager.get_plugin("cost-analyzer") is None

    def test_global_plugin_registration(self) -> None:
        analyzer = SampleCostAnalyzerPlugin()
        register_plugin(analyzer)
        mgr = get_plugin_manager()
        assert mgr.get_plugin("cost-analyzer") is analyzer

    def test_entrypoint_discovery(self) -> None:
        manager = PluginManager()
        mock_ep = MagicMock()
        mock_ep.name = "mock_analyzer"
        mock_ep.load.return_value = SampleCostAnalyzerPlugin

        with patch("importlib.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_ep]
            count = manager.discover_entrypoints()
            assert count == 1
            assert manager.get_plugin("cost-analyzer") is not None
