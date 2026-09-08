"""Terminal and export formatters for diagnosis, verification, and regression reports."""

from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.regression.models import BatchComparisonReport
from llm_reliability.verification.models import VerificationReport


def format_diagnosis_text(diagnosis: Diagnosis) -> str:
    """Render diagnosis into a clean, human-readable terminal text presentation."""
    lines: list[str] = []
    separator = "=" * 80

    lines.append(separator)
    lines.append("  LLM RELIABILITY ANALYZER - ROOT CAUSE DIAGNOSIS REPORT")
    lines.append(separator)

    lines.append(f"  Trace ID:          {diagnosis.trace_id or 'N/A'}")
    if diagnosis.run_id:
        lines.append(f"  Run ID:            {diagnosis.run_id}")
    lines.append(f"  Primary Category:  {diagnosis.primary_category.value.upper()}")
    lines.append(f"  Severity:          [{diagnosis.severity.value.upper()}]")
    lines.append(f"  Summary:           {diagnosis.summary}")
    lines.append(separator)

    # Hypotheses
    if diagnosis.hypotheses:
        lines.append("\n[ROOT CAUSE HYPOTHESES]")
        lines.append("-" * 80)
        for h in diagnosis.hypotheses:
            conf_pct = int(h.confidence * 100)
            lines.append(f"  #{h.rank or 1} [{conf_pct}% Confidence] {h.title}")
            lines.append(f"     Rationale: {h.rationale or h.description}")
            if h.contributing_factors:
                factors_str = ", ".join(h.contributing_factors)
                lines.append(f"     Contributing Factors: {factors_str}")
            lines.append("")

    # Failures
    if diagnosis.failures:
        lines.append("[DETECTED FAILURES & EVIDENCE]")
        lines.append("-" * 80)
        for f in diagnosis.failures:
            lines.append(f"  - [{f.category.value}] {f.title}: {f.description}")
            for ev in f.evidence:
                lines.append(f"      * Evidence ({ev.evidence_type.value}): {ev.description}")
        lines.append("")

    # Recommendations
    if diagnosis.recommendations:
        lines.append("[RECOMMENDED REMEDIATION ACTIONS]")
        lines.append("-" * 80)
        for rec in diagnosis.recommendations:
            lines.append(f"  Priority {rec.priority} | [{rec.action_type or 'ACTION'}] {rec.title}")
            for d_line in rec.description.split("\n"):
                if d_line.strip():
                    lines.append(f"    {d_line.strip()}")
            if rec.rationale:
                lines.append(f"    Why: {rec.rationale}")
            lines.append("")

    # Metrics
    if diagnosis.metrics:
        lines.append("[EVALUATED RELIABILITY METRICS]")
        lines.append("-" * 80)
        for m in diagnosis.metrics:
            status = "PASS" if m.passed else "FAIL"
            thresh_str = f" (Threshold: {m.threshold})" if m.threshold is not None else ""
            lines.append(f"  [{status:<4}] {m.name:<32}: {m.value}{thresh_str}")
        lines.append("")

    lines.append(separator)
    return "\n".join(lines)


def format_diagnosis_markdown(diagnosis: Diagnosis) -> str:
    """Render diagnosis into GitHub-flavored Markdown."""
    lines: list[str] = []

    lines.append("# LLM Reliability Analysis Report\n")
    lines.append(f"- **Trace ID:** `{diagnosis.trace_id or 'N/A'}`")
    if diagnosis.run_id:
        lines.append(f"- **Run ID:** `{diagnosis.run_id}`")
    lines.append(f"- **Primary Category:** `{diagnosis.primary_category.value}`")
    lines.append(f"- **Severity:** `{diagnosis.severity.value.upper()}`\n")
    lines.append(f"> **Summary:** {diagnosis.summary}\n")

    if diagnosis.hypotheses:
        lines.append("## Root Cause Hypotheses\n")
        for h in diagnosis.hypotheses:
            conf_pct = int(h.confidence * 100)
            lines.append(f"### Rank {h.rank or 1}: {h.title} ({conf_pct}% confidence)\n")
            lines.append(f"{h.rationale or h.description}\n")
            if h.contributing_factors:
                lines.append("**Contributing factors:**")
                for cf in h.contributing_factors:
                    lines.append(f"- {cf}")
                lines.append("")

    if diagnosis.recommendations:
        lines.append("## Recommended Actions\n")
        lines.append("| Priority | Action Type | Recommendation |")
        lines.append("| :--- | :--- | :--- |")
        for r in diagnosis.recommendations:
            lines.append(
                f"| **P{r.priority}** | `{r.action_type or 'FIX'}` | **{r.title}**<br>{r.description.replace(chr(10), '<br>')} |"
            )
        lines.append("")

    if diagnosis.metrics:
        lines.append("## Diagnostic Metrics\n")
        lines.append("| Status | Metric Name | Value | Threshold |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for m in diagnosis.metrics:
            status_icon = "PASS" if m.passed else "FAIL"
            thresh = str(m.threshold) if m.threshold is not None else "-"
            lines.append(f"| {status_icon} | `{m.name}` | `{m.value}` | `{thresh}` |")
        lines.append("")

    return "\n".join(lines)


def format_diagnosis_json(diagnosis: Diagnosis) -> str:
    """Serialize diagnosis to formatted JSON string."""
    return diagnosis.model_dump_json(indent=2)


def format_verification_text(report: VerificationReport) -> str:
    """Render verification report into terminal text."""
    lines: list[str] = []
    separator = "=" * 80

    lines.append(separator)
    lines.append("  LLM RELIABILITY ANALYZER - FIX VERIFICATION REPORT")
    lines.append(separator)
    status_label = (
        "VERIFIED (IMPROVED)" if report.is_verified else "NOT VERIFIED (FAILED/REGRESSION)"
    )
    lines.append(f"  Verification Status:  [{status_label}]")
    lines.append(f"  Confidence:           {int(report.overall_confidence * 100)}%")
    lines.append(
        f"  Severity:             {report.severity_before.value.upper()} -> {report.severity_after.value.upper()}"
    )
    lines.append(f"  Summary:              {report.summary}")
    lines.append(separator)

    if report.resolved_failures:
        lines.append("\n[RESOLVED FAILURES]")
        for f in report.resolved_failures:
            lines.append(f"  + {f.value}")

    if report.new_regressions:
        lines.append("\n[NEW REGRESSIONS DETECTED]")
        for r in report.new_regressions:
            lines.append(f"  ! {r.value}")

    if report.metric_comparisons:
        lines.append("\n[METRIC COMPARISONS (BEFORE -> AFTER)]")
        lines.append("-" * 80)
        for m in report.metric_comparisons:
            improved_tag = "[IMPROVED]" if m.improved else "          "
            delta_str = f" (delta: {m.delta:+})" if m.delta is not None else ""
            lines.append(
                f"  {improved_tag} {m.name:<30}: {m.before_value} -> {m.after_value}{delta_str}"
            )

    lines.append("\n" + separator)
    return "\n".join(lines)


def format_regression_text(report: BatchComparisonReport) -> str:
    """Render batch regression comparison report into terminal text."""
    lines: list[str] = []
    separator = "=" * 80

    lines.append(separator)
    lines.append("  LLM RELIABILITY ANALYZER - BATCH REGRESSION REPORT")
    lines.append(separator)

    lines.append(f"  Verdict:              [{report.verdict.value.upper()}]")
    lines.append(f"  Regression Score:     {report.regression_score:.2f} / 1.00")
    lines.append(f"  Baseline Runs:        {report.baseline_runs_count}")
    lines.append(f"  Candidate Runs:       {report.candidate_runs_count}")
    lines.append(
        f"  Failure Rate:         {int(report.baseline_failure_summary.failure_rate * 100)}% -> "
        f"{int(report.candidate_failure_summary.failure_rate * 100)}% "
        f"({report.failure_rate_delta * 100:+.1f}%)"
    )
    lines.append(
        f"  p95 Latency:          {report.latency_baseline.p95}ms -> {report.latency_candidate.p95}ms "
        f"({report.latency_p95_delta_ms:+.1f}ms)"
    )
    lines.append(f"  Summary:              {report.summary}")
    lines.append(separator)

    if report.new_failure_categories:
        lines.append("\n[NEW REGRESSION MODES]")
        for c in report.new_failure_categories:
            lines.append(f"  ! {c.value}")

    if report.resolved_failure_categories:
        lines.append("\n[RESOLVED FAILURE MODES]")
        for c in report.resolved_failure_categories:
            lines.append(f"  + {c.value}")

    lines.append("\n" + separator)
    return "\n".join(lines)
