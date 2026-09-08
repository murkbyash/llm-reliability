"""Automated unit tests for CI/CD evaluation harness, gatekeeper engine, and CLI gate subcommand."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_reliability import (
    FailureCategory,
    Gatekeeper,
    GatekeeperConfig,
    LLMCall,
    Run,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    evaluate_gate,
)
from llm_reliability.cli.main import main


def create_sample_trace(
    trace_id: str,
    status: SpanStatus = SpanStatus.SUCCESS,
    category_error: str | None = None,
    duration_ms: float = 100.0,
) -> Trace:
    """Helper creating standardized Trace objects with controlled status and timing."""
    start = datetime.now(timezone.utc)
    end = start + timedelta(milliseconds=duration_ms)
    span = Span(
        span_id=f"s-{trace_id}",
        name="llm_call",
        kind=SpanKind.LLM,
        status=status,
        start_time=start,
        end_time=end,
        duration_ms=duration_ms,
        error_message=category_error,
        llm_call=LLMCall(
            prompt="Test prompt",
            response="Test response" if status == SpanStatus.SUCCESS else None,
            model="gpt-4",
        ),
    )
    run = Run(
        run_id=f"r-{trace_id}",
        trace_id=trace_id,
        start_time=start,
        end_time=end,
        duration_ms=duration_ms,
        spans=[span],
    )
    return Trace(trace_id=trace_id, runs=[run])


class TestGatekeeperEngine:
    """Test Gatekeeper reliability evaluation, SLA violations, and markdown PR comments."""

    def test_gatekeeper_pass_on_clean_candidate(self) -> None:
        trace = create_sample_trace("clean-1", SpanStatus.SUCCESS)
        gatekeeper = Gatekeeper()
        report = gatekeeper.evaluate(trace)

        assert report.passed is True
        assert report.verdict == "PASSED"
        assert len(report.violations) == 0
        assert "✅ **PASSED**" in report.markdown_summary

    def test_gatekeeper_fail_on_high_failure_rate(self) -> None:
        clean_run = create_sample_trace("clean", SpanStatus.SUCCESS).runs[0]
        failing_run = create_sample_trace("fail", SpanStatus.ERROR, "Connection timeout 504").runs[
            0
        ]
        candidate = Trace(trace_id="cand", runs=[clean_run, failing_run])

        gatekeeper = Gatekeeper()
        config = GatekeeperConfig(max_failure_rate=0.0)
        report = gatekeeper.evaluate(candidate, config=config)

        assert report.passed is False
        assert report.verdict == "FAILED"
        assert len(report.violations) == 1
        assert "failure rate (50.0%) exceeded threshold" in report.violations[0]
        assert "❌ **FAILED**" in report.markdown_summary

    def test_gatekeeper_fail_on_disallowed_category(self) -> None:
        failing_trace = create_sample_trace(
            "fail-llm", SpanStatus.ERROR, "Internal server error 500"
        )
        gatekeeper = Gatekeeper()
        config = GatekeeperConfig(
            max_failure_rate=1.0,  # allow failures generally...
            disallowed_categories=[
                FailureCategory.LLM_CALL_FAILURE
            ],  # ...but forbid LLM_CALL_FAILURE
        )
        report = gatekeeper.evaluate(failing_trace, config=config)

        assert report.passed is False
        assert report.verdict == "FAILED"
        assert any("LLM_CALL_FAILURE" in v for v in report.violations)

    def test_gatekeeper_fail_on_regression_vs_baseline(self) -> None:
        baseline = create_sample_trace("base-clean", SpanStatus.SUCCESS)
        candidate = create_sample_trace("cand-regressed", SpanStatus.ERROR, "Fatal rate limit 429")

        gatekeeper = Gatekeeper()
        config = GatekeeperConfig(max_regression_score=0.0)
        report = gatekeeper.evaluate(
            candidate_trace=candidate,
            baseline_trace=baseline,
            config=config,
        )

        assert report.passed is False
        assert report.verdict == "FAILED"
        assert report.comparison_summary is not None
        assert report.comparison_summary["regression_score"] > 0.0
        assert any("Regression score" in v for v in report.violations)

    def test_gatekeeper_fail_on_p95_latency(self) -> None:
        baseline = create_sample_trace("b", SpanStatus.SUCCESS, duration_ms=50.0)
        candidate = create_sample_trace("c", SpanStatus.SUCCESS, duration_ms=1200.0)

        gatekeeper = Gatekeeper()
        config = GatekeeperConfig(max_latency_p95_ms=500.0)
        report = gatekeeper.evaluate(
            candidate_trace=candidate,
            baseline_trace=baseline,
            config=config,
        )

        assert report.passed is False
        assert report.verdict == "FAILED"
        assert any("p95 latency" in v for v in report.violations)

    def test_evaluate_gate_convenience_function(self, tmp_path: Path) -> None:
        cand_file = tmp_path / "cand.json"
        base_file = tmp_path / "base.json"

        cand_file.write_text(
            create_sample_trace("c", SpanStatus.SUCCESS).model_dump_json(),
            encoding="utf-8",
        )
        base_file.write_text(
            create_sample_trace("b", SpanStatus.SUCCESS).model_dump_json(),
            encoding="utf-8",
        )

        report = evaluate_gate(
            candidate_file=cand_file,
            baseline_file=base_file,
            config=GatekeeperConfig(max_failure_rate=0.0),
        )
        assert report.passed is True
        assert report.verdict == "PASSED"

    def test_cli_gate_subcommand_pass_and_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clean_file = tmp_path / "clean.json"
        fail_file = tmp_path / "fail.json"
        comment_file = tmp_path / "comment.md"

        clean_file.write_text(
            create_sample_trace("c", SpanStatus.SUCCESS).model_dump_json(),
            encoding="utf-8",
        )
        fail_file.write_text(
            create_sample_trace("f", SpanStatus.ERROR, "Fatal crash").model_dump_json(),
            encoding="utf-8",
        )

        # 1. Clean trace passes CLI gate with exit code 0
        ret_clean = main(
            [
                "gate",
                str(clean_file),
                "--max-failure-rate",
                "0.0",
                "--comment-file",
                str(comment_file),
            ]
        )
        assert ret_clean == 0
        assert comment_file.is_file()
        assert "✅ **PASSED**" in comment_file.read_text(encoding="utf-8")

        # 2. Failing trace fails CLI gate with exit code 1
        ret_fail = main(
            [
                "gate",
                str(fail_file),
                "--max-failure-rate",
                "0.0",
            ]
        )
        assert ret_fail == 1

    def test_github_action_definition_file(self) -> None:
        action_file = Path(".github/actions/gatekeeper/action.yml")
        assert action_file.is_file()
        content = action_file.read_text(encoding="utf-8")
        assert "name: 'LLM Reliability Gatekeeper'" in content
        assert "candidate_trace" in content
        assert "llm-reliability gate" in content
