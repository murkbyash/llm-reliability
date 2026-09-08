"""Interactive HTML and Standalone Diagnostic Report Generator."""

import html
import json
from pathlib import Path

from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import Severity
from llm_reliability.models.trace import Span, Trace
from llm_reliability.report.template import HTML_REPORT_TEMPLATE


class DiagnosticReportGenerator:
    """Generates 100% offline, self-contained single-file HTML interactive diagnostic reports."""

    def __init__(self, title: str = "LLM Reliability Diagnostic Report") -> None:
        self.title = title

    def generate(self, diagnosis: Diagnosis, trace: Trace | None = None) -> str:
        """Render a Diagnosis object and optional Trace hierarchy into an interactive HTML string."""
        # 1. Executive Summary fields
        primary_cat = diagnosis.primary_category.value
        confidence_pct = round(diagnosis.confidence * 100.0, 1)

        # Determine primary severity
        primary_severity = diagnosis.severity if diagnosis.severity else Severity.INFO
        primary_sev_class = primary_severity.value.lower()

        total_latency_ms = 0.0
        total_spans = 0
        all_spans: list[Span] = []

        if trace and trace.runs:
            for run in trace.runs:
                all_spans.extend(run.spans)
            total_spans = len(all_spans)
            if all_spans:
                total_latency_ms = sum(float(s.duration_ms or 0.0) for s in all_spans)
        elif diagnosis.metrics:
            lat_metric = next((m for m in diagnosis.metrics if "latency" in m.name.lower()), None)
            if lat_metric and isinstance(lat_metric.value, (int, float)):
                total_latency_ms = float(lat_metric.value)

        # 2. Render Waterfall Timeline
        waterfall_section = self._render_waterfall(all_spans, total_latency_ms)

        # 3. Render Hypotheses & Evidence
        hypotheses_content = self._render_hypotheses(diagnosis)

        # 4. Render Recommendations
        recommendations_content = self._render_recommendations(diagnosis)

        # 5. Render Metrics Table
        metrics_table = self._render_metrics_table(diagnosis)

        # Populate HTML Template
        rendered_html = HTML_REPORT_TEMPLATE.format(
            title=html.escape(self.title),
            primary_category=html.escape(primary_cat),
            primary_severity_class=html.escape(primary_sev_class),
            confidence_pct=confidence_pct,
            total_latency_ms=f"{total_latency_ms:.1f}",
            total_spans=total_spans,
            failure_count=len(diagnosis.failures),
            waterfall_section=waterfall_section,
            hypotheses_content=hypotheses_content,
            recommendations_content=recommendations_content,
            metrics_table=metrics_table,
        )

        return rendered_html

    def save(
        self, diagnosis: Diagnosis, output_path: str | Path, trace: Trace | None = None
    ) -> Path:
        """Generate and save interactive HTML report to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate(diagnosis, trace=trace)
        path.write_text(content, encoding="utf-8")
        return path

    def _render_waterfall(self, spans: list[Span], total_duration_ms: float) -> str:
        if not spans:
            return ""

        html_out = [
            '<div class="section-card">',
            '  <div class="section-header">',
            '    <div class="section-title">Execution Waterfall Timeline</div>',
            "  </div>",
            '  <div class="timeline-container">',
        ]

        safe_total_duration = max(1.0, total_duration_ms)

        for sp in spans:
            dur = sp.duration_ms or 1.0
            width_pct = max(2.0, min(100.0, (dur / safe_total_duration) * 100.0))
            kind_class = f"kind-{sp.kind.value.lower()}"
            status_class = "status-error" if sp.status.value == "ERROR" else "status-success"

            span_data = {
                "name": sp.name,
                "kind": sp.kind.value,
                "status": sp.status.value,
                "duration_ms": dur,
                "error": sp.error_message,
                "prompt": sp.llm_call.prompt if sp.llm_call else None,
                "response": sp.llm_call.response if sp.llm_call else None,
                "tool_args": sp.tool_call.arguments if sp.tool_call else None,
                "tool_output": sp.tool_result.output if sp.tool_result else None,
                "query": sp.retrieval.query if sp.retrieval else None,
            }
            json_payload = html.escape(json.dumps(span_data))

            html_out.append(
                f'    <div class="timeline-row" onclick="showSpanModal({json_payload})">'
                f'      <div class="timeline-name">{html.escape(sp.name)}</div>'
                f'      <div class="timeline-bar-wrapper">'
                f'        <div class="timeline-bar {kind_class} {status_class}" style="left: 0%; width: {width_pct:.1f}%;">'
                f"          {dur:.1f}ms"
                f"        </div>"
                f"      </div>"
                f"    </div>"
            )

        html_out.extend(["  </div>", "</div>"])
        return "\n".join(html_out)

    def _render_hypotheses(self, diagnosis: Diagnosis) -> str:
        if not diagnosis.hypotheses:
            return '<p style="color: var(--text-muted); font-size: 13px;">No failure hypotheses detected.</p>'

        html_out = []
        for hyp in diagnosis.hypotheses:
            conf_pct = round(hyp.confidence * 100.0, 1)
            hyp_title = hyp.title or hyp.description or hyp.rationale or hyp.category.value
            html_out.append(
                f'<div class="hyp-card">'
                f'  <div class="hyp-header">'
                f"    <span>{html.escape(hyp_title)}</span>"
                f"    <span>Confidence: {conf_pct}%"
                f'      <div class="progress-bar-container">'
                f'        <div class="progress-bar-fill" style="width: {conf_pct}%"></div>'
                f"      </div>"
                f"    </span>"
                f"  </div>"
            )

            if hyp.evidence:
                html_out.append(
                    "  <table><thead><tr><th>Evidence Type</th><th>Description</th></tr></thead><tbody>"
                )
                for ev in hyp.evidence:
                    html_out.append(
                        f"    <tr>"
                        f'      <td><span class="badge badge-info">{html.escape(ev.evidence_type.value)}</span></td>'
                        f"      <td>{html.escape(ev.description)}</td>"
                        f"    </tr>"
                    )
                html_out.append("  </tbody></table>")

            html_out.append("</div>")

        return "\n".join(html_out)

    def _render_recommendations(self, diagnosis: Diagnosis) -> str:
        if not diagnosis.recommendations:
            return '<p style="color: var(--text-muted); font-size: 13px;">No remediation actions recommended.</p>'

        html_out = []
        for rec in diagnosis.recommendations:
            priority_label = f"P{rec.priority}"
            badge_class = (
                "badge-high"
                if rec.priority == 1
                else ("badge-medium" if rec.priority == 2 else "badge-low")
            )
            html_out.append(
                f'<div class="hyp-card">'
                f'  <div class="hyp-header">'
                f'    <span><span class="badge {badge_class}">{html.escape(priority_label)}</span> [{html.escape(rec.action_type)}] {html.escape(rec.title)}</span>'
                f"  </div>"
                f'  <p style="font-size: 13px; margin: 6px 0; color: var(--text-secondary);">{html.escape(rec.description)}</p>'
            )

            if rec.rationale:
                html_out.append(
                    f'  <p style="font-size: 12px; margin: 4px 0; color: var(--text-muted);"><em>Rationale:</em> {html.escape(rec.rationale)}</p>'
                )

            html_out.append("</div>")

        return "\n".join(html_out)

    def _render_metrics_table(self, diagnosis: Diagnosis) -> str:
        if not diagnosis.metrics:
            return '<p style="color: var(--text-muted); font-size: 13px;">No diagnostic metrics recorded.</p>'

        html_out = [
            "<table>",
            "  <thead><tr><th>Metric Name</th><th>Value</th><th>Status / Unit</th></tr></thead>",
            "  <tbody>",
        ]

        for m in diagnosis.metrics:
            unit_str = html.escape(m.unit or "")
            html_out.append(
                f"    <tr>"
                f"      <td><code>{html.escape(m.name)}</code></td>"
                f"      <td><strong>{m.value:g}</strong></td>"
                f"      <td>{unit_str}</td>"
                f"    </tr>"
            )

        html_out.extend(["  </tbody>", "</table>"])
        return "\n".join(html_out)


def render_html_report(
    diagnosis: Diagnosis,
    trace: Trace | None = None,
    title: str = "LLM Reliability Diagnostic Report",
) -> str:
    """Convenience helper to render an interactive HTML report string."""
    generator = DiagnosticReportGenerator(title=title)
    return generator.generate(diagnosis, trace=trace)


def save_html_report(
    diagnosis: Diagnosis,
    output_path: str | Path,
    trace: Trace | None = None,
    title: str = "LLM Reliability Diagnostic Report",
) -> Path:
    """Convenience helper to render and save an interactive HTML report to disk."""
    generator = DiagnosticReportGenerator(title=title)
    return generator.save(diagnosis, output_path=output_path, trace=trace)
