"""Tests for Terminal CLI & Formatting Engine."""

import json
from pathlib import Path

import pytest

from llm_reliability import DiagnosticEngine
from llm_reliability.cli import (
    format_diagnosis_json,
    format_diagnosis_markdown,
    format_diagnosis_text,
    main,
)
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import SpanKind
from llm_reliability.models.execution import (
    FinalResponse,
    RetrievalStep,
    RetrievedDocument,
)
from llm_reliability.models.trace import Run, Span


class TestCLIAndFormatting:
    """Deterministic tests for command-line interface and formatting functions."""

    @pytest.fixture
    def sample_trace_file(self, tmp_path: Path) -> Path:
        trace_data = {
            "trace_id": "cli-test-trace-1",
            "runs": [
                {
                    "run_id": "r1",
                    "trace_id": "cli-test-trace-1",
                    "spans": [
                        {
                            "span_id": "s1",
                            "name": "retrieval",
                            "kind": "retrieval",
                            "retrieval": {
                                "query": "test query",
                                "documents": [],
                            },
                        }
                    ],
                }
            ],
        }
        file_path = tmp_path / "sample_trace.json"
        file_path.write_text(json.dumps(trace_data), encoding="utf-8")
        return file_path

    @pytest.fixture
    def clean_trace_file(self, tmp_path: Path) -> Path:
        trace_data = {
            "trace_id": "clean-trace-1",
            "runs": [
                {
                    "run_id": "r1",
                    "trace_id": "clean-trace-1",
                    "duration_ms": 100.0,
                    "spans": [
                        {
                            "span_id": "s1",
                            "name": "retrieval",
                            "kind": "retrieval",
                            "duration_ms": 100.0,
                            "retrieval": {
                                "query": "python",
                                "documents": [
                                    {
                                        "doc_id": "d1",
                                        "content": "Python is an open source programming language with clear syntax and high developer productivity.",
                                        "score": 0.95,
                                    }
                                ],
                            },
                        }
                    ],
                    "final_response": {
                        "text": "Python is an open source programming language with clear syntax."
                    },
                }
            ],
        }
        file_path = tmp_path / "clean_trace.json"
        file_path.write_text(json.dumps(trace_data), encoding="utf-8")
        return file_path

    def test_cli_no_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main([])
        assert ret == 0
        captured = capsys.readouterr()
        assert "llm-reliability" in captured.out

    def test_cli_diagnose_text(
        self, sample_trace_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["diagnose", str(sample_trace_file), "--format", "text"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "LLM RELIABILITY ANALYZER" in captured.out
        assert "RETRIEVAL_FAILURE" in captured.out
        assert "ROOT CAUSE HYPOTHESES" in captured.out

    def test_cli_diagnose_json(
        self, sample_trace_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["diagnose", str(sample_trace_file), "--format", "json"])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["primary_category"] == "RETRIEVAL_FAILURE"
        assert len(parsed["hypotheses"]) >= 1

    def test_cli_diagnose_markdown(
        self, sample_trace_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["diagnose", str(sample_trace_file), "--format", "markdown"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "# LLM Reliability Analysis Report" in captured.out
        assert "## Root Cause Hypotheses" in captured.out
        assert "## Recommended Actions" in captured.out

    def test_cli_diagnose_output_to_file(self, sample_trace_file: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "report.md"
        ret = main(
            [
                "diagnose",
                str(sample_trace_file),
                "--format",
                "markdown",
                "--output",
                str(out_file),
            ]
        )
        assert ret == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "# LLM Reliability Analysis Report" in content

    def test_cli_verify(
        self, sample_trace_file: Path, clean_trace_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["verify", str(sample_trace_file), str(clean_trace_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "FIX VERIFICATION REPORT" in captured.out
        assert "RESOLVED FAILURES" in captured.out

    def test_cli_compare(
        self, sample_trace_file: Path, clean_trace_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = main(["compare", str(sample_trace_file), str(clean_trace_file)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "BATCH REGRESSION REPORT" in captured.out
        assert "Verdict:" in captured.out

    def test_cli_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["diagnose", "non_existent_file_path_12345.json"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_formatter_direct_calls(self) -> None:
        doc = RetrievedDocument(
            doc_id="d1",
            content="Rust ensures memory safety through ownership and lifetimes.",
            score=0.9,
        )
        step = RetrievalStep(query="rust", documents=[doc])
        span = Span(span_id="s1", name="ret", kind=SpanKind.RETRIEVAL, retrieval=step)
        run = Run(
            run_id="r1",
            trace_id="t1",
            spans=[span],
            final_response=FinalResponse(
                text="Rust ensures memory safety through ownership and lifetimes."
            ),
        )
        diagnosis: Diagnosis = DiagnosticEngine().diagnose(run)

        text_out = format_diagnosis_text(diagnosis)
        assert "LLM RELIABILITY ANALYZER" in text_out

        md_out = format_diagnosis_markdown(diagnosis)
        assert "# LLM Reliability Analysis Report" in md_out

        json_out = format_diagnosis_json(diagnosis)
        assert "primary_category" in json.loads(json_out)
