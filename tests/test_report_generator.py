"""Tests for Interactive HTML and Standalone Diagnostic Report Generator."""

import json
from pathlib import Path

from llm_reliability import (
    DiagnosticReportGenerator,
    LLMCall,
    RetrievalStep,
    RetrievedDocument,
    Run,
    Span,
    SpanKind,
    SpanStatus,
    ToolCall,
    ToolResult,
    Trace,
    diagnose,
    render_html_report,
    save_html_report,
)
from llm_reliability.cli.main import main


class TestDiagnosticReportGenerator:
    """Deterministic validation of offline interactive HTML report generation and CLI integration."""

    def test_render_html_report_clean_diagnosis(self) -> None:
        trace = Trace(
            trace_id="trace-clean-01",
            runs=[
                Run(
                    run_id="run-clean-01",
                    trace_id="trace-clean-01",
                    spans=[
                        Span(
                            span_id="s1",
                            name="chat_llm",
                            kind=SpanKind.LLM,
                            status=SpanStatus.SUCCESS,
                            duration_ms=120.0,
                            llm_call=LLMCall(model="gpt-4o", prompt="Hello", response="Hi there!"),
                        )
                    ],
                )
            ],
        )
        diagnosis = diagnose(trace)

        html_out = render_html_report(diagnosis, trace=trace, title="Test Clean Report")

        assert "<!DOCTYPE html>" in html_out
        assert "<title>Test Clean Report</title>" in html_out
        assert "Primary Diagnosis" in html_out
        assert "NONE" in html_out
        assert "Execution Waterfall Timeline" in html_out
        assert "chat_llm" in html_out
        assert "120.0ms" in html_out

        # Ensure 100% offline & zero CDN dependency
        assert "http://" not in html_out
        assert "https://" not in html_out

    def test_render_html_report_with_failures_and_trace(self) -> None:
        trace = Trace(
            trace_id="trace-fail-01",
            runs=[
                Run(
                    run_id="run-fail-01",
                    trace_id="trace-fail-01",
                    spans=[
                        Span(
                            span_id="s1",
                            name="vector_search",
                            kind=SpanKind.RETRIEVAL,
                            status=SpanStatus.SUCCESS,
                            duration_ms=45.0,
                            retrieval=RetrievalStep(
                                query="quantum algorithms",
                                documents=[
                                    RetrievedDocument(
                                        doc_id="d1",
                                        content="Quantum algorithms utilize superposition and entanglement.",
                                        score=0.92,
                                    )
                                ],
                            ),
                        ),
                        Span(
                            span_id="s2",
                            name="broken_tool",
                            kind=SpanKind.TOOL,
                            status=SpanStatus.ERROR,
                            error_message="NetworkTimeout: database unreachable",
                            duration_ms=500.0,
                            tool_call=ToolCall(tool_name="fetch_db", arguments={"table": "users"}),
                            tool_result=ToolResult(
                                tool_name="fetch_db",
                                status=SpanStatus.ERROR,
                                error="NetworkTimeout: database unreachable",
                            ),
                        ),
                    ],
                )
            ],
        )

        diagnosis = diagnose(trace)
        generator = DiagnosticReportGenerator(title="Failure Analysis Report")
        html_out = generator.generate(diagnosis, trace=trace)

        assert "Failure Analysis Report" in html_out
        assert "TOOL_ERROR" in html_out or "TOOL" in html_out
        assert "Execution Waterfall Timeline" in html_out
        assert "vector_search" in html_out
        assert "broken_tool" in html_out
        assert "NetworkTimeout" in html_out
        assert "Actionable Developer Remediation" in html_out
        assert "copyCode" in html_out

    def test_save_html_report_file_io(self, tmp_path: Path) -> None:
        trace = Trace(
            trace_id="trace-io-01",
            runs=[
                Run(
                    run_id="run-io-01",
                    trace_id="trace-io-01",
                    spans=[
                        Span(
                            span_id="s1",
                            name="root",
                            kind=SpanKind.ROOT,
                            status=SpanStatus.SUCCESS,
                            duration_ms=10.0,
                        )
                    ],
                )
            ],
        )
        diagnosis = diagnose(trace)

        out_file = tmp_path / "subfolder" / "diagnostic_report.html"
        saved_path = save_html_report(diagnosis, out_file, trace=trace)

        assert saved_path.is_file()
        content = saved_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "AGY" in content

    def test_cli_diagnose_html_format(self, tmp_path: Path) -> None:
        trace_data = {
            "trace_id": "trace-cli-html",
            "runs": [
                {
                    "run_id": "run-cli-html",
                    "trace_id": "trace-cli-html",
                    "spans": [
                        {
                            "span_id": "s1",
                            "name": "llm_generate",
                            "kind": "LLM",
                            "status": "SUCCESS",
                            "duration_ms": 150.0,
                            "llm_call": {
                                "model": "gpt-4o",
                                "prompt": "Explain Python typing",
                                "response": "Python typing enables static verification.",
                            },
                        }
                    ],
                }
            ],
        }
        trace_file = tmp_path / "test_trace.json"
        trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

        report_file = tmp_path / "cli_report.html"

        # Execute CLI diagnose command with --format html
        exit_code = main(
            [
                "diagnose",
                str(trace_file),
                "--format",
                "html",
                "--output",
                str(report_file),
            ]
        )

        assert exit_code == 0
        assert report_file.is_file()
        content = report_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "llm_generate" in content
