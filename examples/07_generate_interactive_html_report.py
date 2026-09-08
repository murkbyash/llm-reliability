"""Example 07: Generating Offline Standalone Interactive HTML Reports."""

from pathlib import Path

from llm_reliability import (
    DiagnosticEngine,
    SyntheticTraceGenerator,
    save_html_report,
)


def main() -> None:
    print("=" * 70)
    print("Example 07: Generating 100% Offline Interactive HTML Dashboards")
    print("=" * 70)

    generator = SyntheticTraceGenerator()
    trace = generator.generate_hallucination_trace()

    engine = DiagnosticEngine()
    diagnosis = engine.diagnose(trace)

    output_path = Path("production_diagnostic_report.html")
    saved_file = save_html_report(
        diagnosis=diagnosis,
        output_path=output_path,
        trace=trace,
        title="Production Diagnostic Dashboard",
    )

    print("\n[1] Standalone HTML Dashboard successfully created:")
    print(f"    * File Location : {saved_file.resolve()}")
    print(f"    * File Size     : {saved_file.stat().st_size / 1024.0:.1f} KB")
    print("    * Dependencies  : 0 external CDNs (100% offline self-contained)")
    print("    * Features      : Dark/Light mode, waterfall timeline, span modal,")
    print("                      remediation copy buttons, confidence progress bars.")
    print("=" * 70)


if __name__ == "__main__":
    main()
