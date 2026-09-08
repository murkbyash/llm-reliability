"""CI/CD Gatekeeper evaluation engine enforcing reliability regression gates in pull requests."""

import logging
from pathlib import Path
from typing import Any

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.eval_harness.models import GatekeeperConfig, GatekeeperReport
from llm_reliability.models.enums import FailureCategory
from llm_reliability.models.trace import Trace
from llm_reliability.normalization.loader import load_trace
from llm_reliability.policies.engine import PolicyEngine
from llm_reliability.regression.engine import RegressionEngine

logger = logging.getLogger(__name__)


class Gatekeeper:
    """Evaluates candidate execution traces against strict reliability SLAs and regression gates."""

    def __init__(self) -> None:
        self.diagnostic_engine = DiagnosticEngine()
        self.regression_engine = RegressionEngine()
        self.policy_engine = PolicyEngine()

    def evaluate(
        self,
        candidate_trace: Trace,
        baseline_trace: Trace | None = None,
        config: GatekeeperConfig | None = None,
    ) -> GatekeeperReport:
        """Evaluate candidate trace against defined gatekeeper criteria."""
        cfg = config or GatekeeperConfig()
        violations: list[str] = []

        # 1. Diagnose all candidate runs
        total_runs = len(candidate_trace.runs)
        if total_runs == 0:
            return GatekeeperReport(
                verdict="PASSED",
                passed=True,
                candidate_summary={"total_runs": 0, "failure_rate": 0.0},
                markdown_summary="### ✅ LLM Reliability Gate: PASSED (0 runs to evaluate)",
            )

        failing_runs = 0
        detected_categories: set[FailureCategory] = set()
        grounding_scores: list[float] = []

        for run in candidate_trace.runs:
            diag = self.diagnostic_engine.diagnose_run(run)
            if diag.primary_category != FailureCategory.NONE:
                failing_runs += 1
                detected_categories.add(diag.primary_category)
                for f in diag.failures:
                    detected_categories.add(f.category)

            # Check grounding score metrics if present
            for m in diag.metrics:
                if m.name in ("grounding_faithfulness", "grounding_score", "faithfulness"):
                    if isinstance(m.value, (int, float)):
                        grounding_scores.append(float(m.value))

        failure_rate = failing_runs / total_runs

        # 2. Evaluate failure rate threshold
        if failure_rate > cfg.max_failure_rate:
            violations.append(
                f"Candidate failure rate ({failure_rate:.1%}) exceeded threshold ({cfg.max_failure_rate:.1%})."
            )

        # 3. Evaluate disallowed categories
        for cat in cfg.disallowed_categories:
            if cat in detected_categories:
                violations.append(
                    f"Disallowed failure category '{cat.value}' was detected in candidate execution."
                )

        # 4. Evaluate minimum grounding score
        if cfg.min_grounding_score is not None and grounding_scores:
            avg_grounding = sum(grounding_scores) / len(grounding_scores)
            if avg_grounding < cfg.min_grounding_score:
                violations.append(
                    f"Average grounding score ({avg_grounding:.2f}) fell below minimum required ({cfg.min_grounding_score:.2f})."
                )

        # 5. Evaluate regression against baseline if provided
        comparison_summary: dict[str, Any] | None = None
        if baseline_trace is not None and baseline_trace.runs:
            comp_report = self.regression_engine.compare_batches(baseline_trace, candidate_trace)
            comparison_summary = {
                "verdict": comp_report.verdict.value,
                "regression_score": comp_report.regression_score,
                "baseline_failure_rate": comp_report.baseline_failure_summary.failure_rate,
                "candidate_failure_rate": comp_report.candidate_failure_summary.failure_rate,
                "new_failures": [f.value for f in comp_report.new_failure_categories],
                "resolved_failures": [f.value for f in comp_report.resolved_failure_categories],
                "latency_p95_candidate_ms": comp_report.latency_candidate.p95,
            }

            if comp_report.regression_score > cfg.max_regression_score:
                violations.append(
                    f"Regression score ({comp_report.regression_score:.2f}) exceeded limit ({cfg.max_regression_score:.2f}). Newly introduced failures: {[f.value for f in comp_report.new_failure_categories]}."
                )

            if (
                cfg.max_latency_p95_ms is not None
                and comp_report.latency_candidate.p95 > cfg.max_latency_p95_ms
            ):
                violations.append(
                    f"Candidate p95 latency ({comp_report.latency_candidate.p95:.1f}ms) exceeded maximum allowable ({cfg.max_latency_p95_ms:.1f}ms)."
                )

        # 6. Evaluate custom policy file if provided
        if cfg.policy_file is not None and Path(cfg.policy_file).is_file():
            try:
                import json

                from llm_reliability.policies.models import DiagnosticPolicy

                policy_data = json.loads(Path(cfg.policy_file).read_text(encoding="utf-8"))
                policy = DiagnosticPolicy.model_validate(policy_data)
                policy_engine = PolicyEngine(policy=policy)
                policy_eval = policy_engine.evaluate(candidate_trace)
                if not policy_eval.passed:
                    for v in policy_eval.violations:
                        violations.append(f"Policy violation '{v.name}': {v.message}")
            except Exception as e:
                logger.warning(f"Failed to evaluate custom policy file '{cfg.policy_file}': {e}")

        passed = len(violations) == 0
        verdict = "PASSED" if passed else "FAILED"

        candidate_summary = {
            "total_runs": total_runs,
            "failing_runs": failing_runs,
            "failure_rate": failure_rate,
            "detected_categories": [c.value for c in detected_categories],
        }

        markdown_summary = self._generate_markdown_summary(
            verdict=verdict,
            passed=passed,
            violations=violations,
            candidate_summary=candidate_summary,
            comparison_summary=comparison_summary,
            config=cfg,
        )

        return GatekeeperReport(
            verdict=verdict,
            passed=passed,
            violations=violations,
            candidate_summary=candidate_summary,
            comparison_summary=comparison_summary,
            markdown_summary=markdown_summary,
        )

    def _generate_markdown_summary(
        self,
        verdict: str,
        passed: bool,
        violations: list[str],
        candidate_summary: dict[str, Any],
        comparison_summary: dict[str, Any] | None,
        config: GatekeeperConfig,
    ) -> str:
        """Render formatted GitHub PR comment markdown."""
        badge = "✅ **PASSED**" if passed else "❌ **FAILED**"
        total = candidate_summary.get("total_runs", 0)
        failed = candidate_summary.get("failing_runs", 0)
        fail_rate = candidate_summary.get("failure_rate", 0.0)

        lines = [
            f"## 🔍 LLM Reliability Gatekeeper Report: {badge}",
            "",
            "| Metric | Candidate Value | SLA Limit | Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Failure Rate** | {fail_rate:.1%} ({failed}/{total}) | &le; {config.max_failure_rate:.1%} | {'✅ OK' if fail_rate <= config.max_failure_rate else '❌ Violation'} |",
        ]

        if comparison_summary is not None:
            reg_score = comparison_summary.get("regression_score", 0.0)
            p95 = comparison_summary.get("latency_p95_candidate_ms", 0.0)
            lines.append(
                f"| **Regression Score** | {reg_score:.2f} | &le; {config.max_regression_score:.2f} | {'✅ OK' if reg_score <= config.max_regression_score else '❌ Violation'} |"
            )
            if config.max_latency_p95_ms is not None:
                lines.append(
                    f"| **p95 Latency** | {p95:.1f} ms | &le; {config.max_latency_p95_ms:.1f} ms | {'✅ OK' if p95 <= config.max_latency_p95_ms else '❌ Violation'} |"
                )

        if violations:
            lines.extend(
                [
                    "",
                    "### ⚠️ Unmet Gatekeeper Criteria:",
                ]
            )
            for v in violations:
                lines.append(f"- 🔴 {v}")

        lines.extend(
            [
                "",
                "---",
                "*Report generated deterministically by [LLM Reliability Analyzer](https://github.com/murkbyash/llm-reliability)*",
            ]
        )

        return "\n".join(lines)


def evaluate_gate(
    candidate_file: str | Path,
    baseline_file: str | Path | None = None,
    config: GatekeeperConfig | None = None,
) -> GatekeeperReport:
    """Convenience helper to evaluate files against gatekeeper criteria."""
    candidate_trace = load_trace(candidate_file)
    baseline_trace = load_trace(baseline_file) if baseline_file is not None else None

    gatekeeper = Gatekeeper()
    return gatekeeper.evaluate(candidate_trace, baseline_trace, config)
