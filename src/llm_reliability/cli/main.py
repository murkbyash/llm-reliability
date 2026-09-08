"""Command line interface entrypoint for llm-reliability."""

import argparse
import sys
from collections.abc import Sequence

from llm_reliability import (
    DiagnosticEngine,
    RegressionEngine,
    VerificationEngine,
    __version__,
)
from llm_reliability.cli.formatter import (
    format_diagnosis_json,
    format_diagnosis_markdown,
    format_diagnosis_text,
    format_regression_text,
    format_verification_text,
)
from llm_reliability.normalization.loader import load_trace


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-reliability",
        description="Local-first root cause diagnostic and reliability analyzer for LLM, RAG, and Agent traces.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. diagnose
    diag_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose an execution trace or run file and output root cause findings.",
    )
    diag_parser.add_argument(
        "trace_file",
        type=str,
        help="Path to trace file (JSON, JSONL, OTEL, LangChain, etc.)",
    )
    diag_parser.add_argument(
        "--format",
        default="text",
        help="Output presentation format (default: text)",
    )
    diag_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional file path to save output report",
    )

    # 2. verify
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify whether a proposed fix resolved failures between before and after traces.",
    )
    verify_parser.add_argument(
        "before_file",
        type=str,
        help="Path to baseline/before trace file",
    )
    verify_parser.add_argument(
        "after_file",
        type=str,
        help="Path to candidate/after trace file",
    )
    verify_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    verify_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output destination",
    )

    # 3. compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two batches of execution runs for statistical regressions and latency drift.",
    )
    compare_parser.add_argument(
        "baseline_file",
        type=str,
        help="Path to baseline batch trace file",
    )
    compare_parser.add_argument(
        "candidate_file",
        type=str,
        help="Path to candidate batch trace file",
    )
    compare_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output destination",
    )

    # 4. serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the local interactive diagnostic visualizer HTTP server.",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind server to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port number to listen on (default: 8080)",
    )
    serve_parser.add_argument(
        "--trace-dir",
        "-d",
        type=str,
        default=None,
        help="Optional directory containing trace JSON files to preload",
    )
    serve_parser.add_argument(
        "--open-browser",
        "-b",
        action="store_true",
        default=False,
        help="Automatically open web browser upon server launch",
    )

    # 5. gate
    gate_parser = subparsers.add_parser(
        "gate",
        help="Evaluate execution traces against reliability SLAs and PR regression gates.",
    )
    gate_parser.add_argument(
        "candidate_file",
        type=str,
        help="Path to candidate trace file to evaluate",
    )
    gate_parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional path to baseline trace file for comparative regression evaluation",
    )
    gate_parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help="Maximum allowed failure rate ratio (0.0 to 1.0, default: 0.0)",
    )
    gate_parser.add_argument(
        "--max-regression",
        type=float,
        default=0.0,
        help="Maximum acceptable regression score against baseline (default: 0.0)",
    )
    gate_parser.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="Maximum allowable p95 latency in milliseconds",
    )
    gate_parser.add_argument(
        "--min-grounding",
        type=float,
        default=None,
        help="Minimum required average grounding / faithfulness score (0.0 to 1.0)",
    )
    gate_parser.add_argument(
        "--disallowed-categories",
        type=str,
        default=None,
        help="Comma-separated list of prohibited failure categories (e.g. HALLUCINATION,AGENT_LOOP)",
    )
    gate_parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Optional path to custom diagnostic policy file (YAML/JSON)",
    )
    gate_parser.add_argument(
        "--comment-file",
        type=str,
        default=None,
        help="Optional file path to output GitHub PR markdown comment",
    )

    return parser


def handle_gate(args: argparse.Namespace) -> int:
    """Execute gate subcommand."""
    from llm_reliability.eval_harness import GatekeeperConfig, evaluate_gate
    from llm_reliability.models.enums import FailureCategory

    disallowed: list[FailureCategory] = []
    if args.disallowed_categories:
        for cat_str in args.disallowed_categories.split(","):
            cat_str = cat_str.strip()
            if cat_str:
                try:
                    disallowed.append(FailureCategory(cat_str))
                except ValueError:
                    pass

    config = GatekeeperConfig(
        max_failure_rate=args.max_failure_rate,
        max_regression_score=args.max_regression,
        max_latency_p95_ms=args.max_latency,
        min_grounding_score=args.min_grounding,
        disallowed_categories=disallowed,
        policy_file=args.policy,
    )

    report = evaluate_gate(
        candidate_file=args.candidate_file,
        baseline_file=args.baseline,
        config=config,
    )

    if args.comment_file:
        with open(args.comment_file, "w", encoding="utf-8") as f:
            f.write(report.markdown_summary)

    print(report.markdown_summary)

    return 0 if report.passed else 1


def handle_serve(args: argparse.Namespace) -> int:
    """Execute serve subcommand."""
    from llm_reliability.server import ServerConfig, start_server

    config = ServerConfig(
        host=args.host,
        port=args.port,
        trace_dir=args.trace_dir,
        auto_open=args.open_browser,
    )
    print(
        f"Starting LLM Reliability Visualizer at http://{args.host}:{args.port} (Press Ctrl+C to stop)..."
    )
    start_server(config, background=False)
    return 0


def handle_diagnose(args: argparse.Namespace) -> int:
    """Execute diagnose subcommand."""
    trace = load_trace(args.trace_file)
    engine = DiagnosticEngine()
    diagnosis = engine.diagnose(trace)

    if args.format == "json":
        output_str = format_diagnosis_json(diagnosis)
    elif args.format == "markdown":
        output_str = format_diagnosis_markdown(diagnosis)
    elif args.format == "html":
        from llm_reliability.report.generator import render_html_report

        output_str = render_html_report(diagnosis, trace=trace)
    else:
        output_str = format_diagnosis_text(diagnosis)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
    else:
        print(output_str)
    return 0


def handle_verify(args: argparse.Namespace) -> int:
    """Execute verify subcommand."""
    engine = VerificationEngine()
    report = engine.verify_fix(args.before_file, args.after_file)

    if args.format == "json":
        output_str = report.model_dump_json(indent=2)
    else:
        output_str = format_verification_text(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
    else:
        print(output_str)
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Execute compare subcommand."""
    baseline_trace = load_trace(args.baseline_file)
    candidate_trace = load_trace(args.candidate_file)

    engine = RegressionEngine()
    report = engine.compare_batches(baseline_trace, candidate_trace)

    if args.format == "json":
        output_str = report.model_dump_json(indent=2)
    else:
        output_str = format_regression_text(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
    else:
        print(output_str)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI execution entrypoint."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "diagnose":
            return handle_diagnose(args)
        elif args.command == "verify":
            return handle_verify(args)
        elif args.command == "compare":
            return handle_compare(args)
        elif args.command == "serve":
            return handle_serve(args)
        elif args.command == "gate":
            return handle_gate(args)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
