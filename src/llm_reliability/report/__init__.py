"""Interactive HTML and Standalone Diagnostic Report Generator module."""

from llm_reliability.report.generator import (
    DiagnosticReportGenerator,
    render_html_report,
    save_html_report,
)

__all__ = [
    "DiagnosticReportGenerator",
    "render_html_report",
    "save_html_report",
]
