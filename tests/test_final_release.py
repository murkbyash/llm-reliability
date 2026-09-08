"""Final Production Polish & End-to-End Release Candidate Verification Suite."""

from pathlib import Path

import pytest

from llm_reliability import (
    BaseAnalyzerPlugin,
    DiagnosticEngine,
    DiagnosticReportGenerator,
    Evidence,
    Failure,
    FailureCategory,
    Gatekeeper,
    GatekeeperConfig,
    Metric,
    MultiModalReliabilityAnalyzer,
    PIIScrubber,
    PluginMetadata,
    RecommendationEngine,
    RedactionConfig,
    RegressionEngine,
    StructuredOutputAnalyzer,
    SyntheticTraceGenerator,
    TelemetryAnonymizer,
    Trace,
    Tracer,
    VerificationEngine,
    __version__,
    diagnose,
    register_plugin,
)
from llm_reliability.cli.main import main


class TestReleaseCandidateIntegrity:
    """Comprehensive release verification covering full pipeline integration and metadata."""

    def test_version_consistency_and_metadata(self) -> None:
        """Ensure package version string is standardized and non-empty."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        assert __version__ == "0.1.0.dev0"

    def test_end_to_end_reliability_pipeline(self) -> None:
        """Exercise full end-to-end lifecycle across all 35 architectural subsystems."""
        # 1. Synthetic Trace Generation
        generator = SyntheticTraceGenerator(seed=42)
        trace_rag = generator.generate_empty_retrieval_trace(trace_id="rc-rag-1")
        trace_clean = generator.generate_clean_trace(trace_id="rc-clean-1")

        # 2. PII Scrubbing & Anonymization
        scrubber = PIIScrubber()
        raw_text = "User admin@corp.com executed trace with key sk-proj-999"
        scrubbed = scrubber.scrub_text(raw_text)
        assert "[EMAIL]" in scrubbed.sanitized_text

        anonymizer = TelemetryAnonymizer(RedactionConfig())
        sanitized_trace = anonymizer.anonymize_trace(trace_rag)
        assert sanitized_trace.trace_id == trace_rag.trace_id

        # 3. Custom Plugin System
        class RCValidationPlugin(BaseAnalyzerPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="rc_validator", version="1.0.0")

            def analyze(self, trace: Trace) -> tuple[list[Failure], list[Evidence], list[Metric]]:
                return [], [], []

        register_plugin(RCValidationPlugin())

        # 4. Multi-Modal and Structured Output Analysis
        struct_analyzer = StructuredOutputAnalyzer()
        is_valid, parsed, _ = struct_analyzer.validate_json('{"status": "verified", "count": 10}')
        assert is_valid is True
        assert parsed["status"] == "verified"

        mm_engine = MultiModalReliabilityAnalyzer()
        mm_res = mm_engine.analyze_trace(trace_clean)
        assert len(mm_res.failures) == 0

        # 5. Core Root Cause Diagnostic Engine
        diag_engine = DiagnosticEngine()
        diagnosis_failing = diag_engine.diagnose(trace_rag)
        assert len(diagnosis_failing.failures) > 0
        assert diagnosis_failing.primary_category != FailureCategory.NONE

        diagnosis_clean = diag_engine.diagnose(trace_clean)
        assert len(diagnosis_clean.failures) == 0

        # 6. Recommendation Engine
        rec_engine = RecommendationEngine()
        recs = rec_engine.generate_recommendations(diagnosis_failing)
        assert len(recs) > 0
        assert recs[0].title is not None

        # 7. Verification & Counterfactual Simulation
        verif_engine = VerificationEngine()
        verif_report = verif_engine.verify_fix(before_source=trace_rag, after_source=trace_clean)
        assert verif_report.resolved_failures is not None

        # 8. Multi-Run Batch Regression Engine
        reg_engine = RegressionEngine()
        batch_report = reg_engine.compare_batches([trace_clean], [trace_rag])
        assert batch_report.verdict is not None

        # 9. Interactive HTML Report Generation
        report_gen = DiagnosticReportGenerator()
        html_content = report_gen.generate(diagnosis_failing, trace=trace_rag)
        assert "<!DOCTYPE html>" in html_content
        assert "LLM Reliability Diagnostic Report" in html_content

        # 10. Live Tracing SDK
        tracer = Tracer()
        with tracer.start_trace(trace_id="rc-live-trace") as t_mgr:
            with tracer.trace_llm(model="gpt-4o", prompt="Hello RC"):
                pass
        live_trace = t_mgr.trace
        assert live_trace is not None
        live_diag = diagnose(live_trace)
        assert live_diag is not None

        # 11. CI Gatekeeper Evaluation
        gatekeeper = Gatekeeper()
        gate_report = gatekeeper.evaluate(
            candidate_trace=trace_clean,
            baseline_trace=trace_clean,
            config=GatekeeperConfig(max_failure_rate=0.0),
        )
        assert gate_report.passed is True
        assert gate_report.verdict == "PASSED"

    def test_cli_subcommands_end_to_end(self, tmp_path: Path) -> None:
        """Verify all CLI subcommands (diagnose, verify, compare, gate, version)."""
        generator = SyntheticTraceGenerator(seed=123)
        t_clean = generator.generate_clean_trace("clean-cli")
        t_fail = generator.generate_empty_retrieval_trace("fail-cli")

        clean_file = tmp_path / "clean.json"
        fail_file = tmp_path / "fail.json"
        report_file = tmp_path / "report.html"

        clean_file.write_text(t_clean.model_dump_json(), encoding="utf-8")
        fail_file.write_text(t_fail.model_dump_json(), encoding="utf-8")

        # 1. Version
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

        # 2. Diagnose HTML
        assert (
            main(
                [
                    "diagnose",
                    str(fail_file),
                    "--format",
                    "html",
                    "-o",
                    str(report_file),
                ]
            )
            == 0
        )
        assert report_file.is_file()

        # 3. Verify
        assert main(["verify", str(fail_file), str(clean_file), "--format", "json"]) == 0

        # 4. Compare
        assert main(["compare", str(clean_file), str(fail_file), "--format", "json"]) == 0

        # 5. Gate
        assert (
            main(
                [
                    "gate",
                    str(clean_file),
                    "--max-failure-rate",
                    "0.0",
                ]
            )
            == 0
        )

    def test_all_documentation_files_present_and_populated(self) -> None:
        """Verify complete documentation suite is present and valid."""
        docs_dir = Path("docs")
        assert docs_dir.is_dir()
        md_files = list(docs_dir.glob("*.md"))
        assert len(md_files) >= 20
        for doc in md_files:
            content = doc.read_text(encoding="utf-8")
            assert len(content.strip()) > 50, f"Doc {doc.name} is suspiciously short"
