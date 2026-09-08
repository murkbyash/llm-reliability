# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automated changelog drafting and release note generation via Release Drafter.
- Version bumping automation script (`scripts/bump_version.py`).

## [0.1.0] - 2026-08-31

### Added
- **Core Data Model & Trace Normalization**:
  - Strongly typed Pydantic v2 execution and diagnostic models (`Span`, `Run`, `Trace`, `Diagnosis`, `Hypothesis`, `Evidence`, `Metric`, `Recommendation`).
  - Multi-format ingestion normalizer (`load_trace`, `normalize_trace`, `parse_timestamp`) supporting standard schemas, OpenInference, OpenLLMetry, Arize Phoenix, and LangSmith nested run trees.
- **Multi-Domain Root Cause Analysis**:
  - **RAG Analysis Engine (`RAGAnalyzer`)**: Evaluates retrieval candidate counts, score distributions, relevance ratios, near/exact duplicate detection (Jaccard similarity), context volume, and token bloat.
  - **Grounding & Answer Support Engine (`GroundingAnalyzer`)**: Lexical claim decomposition, n-gram containment, polar/numerical contradiction detection, entity hallucination tracking, and claim support classification (`SUPPORTED`, `UNSUPPORTED`, `CONTRADICTION`).
  - **Agent & Tool Trajectory Engine (`AgentAnalyzer`)**: Exact repeat cycle detection, ping-pong alternating loops, failed retry loops, tool argument validation, step budget enforcement, and agent state drift detection.
  - **Multi-Modal & Structured Output Analyzer (`StructuredOutputAnalyzer`, `MultiModalAnalyzer`, `MultiModalReliabilityAnalyzer`)**: JSON syntax and JSON schema constraint verification, missing key detection, type mismatch validation, base64 payload corruption checks, and visual reference hallucination tracking.
- **Root Cause Diagnostic Engine (`DiagnosticEngine`)**:
  - Unified multi-layer root-cause orchestration with ranked hypotheses, mathematical confidence scores, and multi-tier severity classifications (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`).
- **Developer Recommendation Engine (`RecommendationEngine`)**:
  - Actionable prioritized remediation advice (`P1`, `P2`, `P3`) with concrete parameter adjustments and drop-in code snippets.
- **Counterfactual Simulation & Verification (`VerificationEngine`)**:
  - Simulation of top-k and score cutoff adjustments, pre-context deduplication, agent loop middleware interception, before vs after metric comparisons, and fix verification reports.
- **Multi-Run Comparative Regression Engine (`RegressionEngine`)**:
  - Batch comparative analysis comparing baseline vs candidate trace distributions (p50, p90, p95, p99 latencies, failure rates, newly introduced vs resolved failures) with automated regression verdicts (`PASSED`, `IMPROVED`, `REGRESSION`, `INCONCLUSIVE`).
- **Failure Signatures & Pattern Detection (`PatternDetectionEngine`)**:
  - Deterministic cryptographic failure fingerprints, recurrence clustering, and representative trace selection.
- **Custom Rules & Policy Engine (`PolicyEngine`)**:
  - Configurable SLA thresholds, custom rule condition operators (`<`, `<=`, `>`, `>=`, `==`, `!=`, `in`, `not_in`), strict grounding enforcement, and YAML/JSON policy loaders.
- **Synthetic Generator & Accuracy Benchmark Suite (`SyntheticTraceGenerator`, `BenchmarkRunner`)**:
  - Programmatic synthetic generator across 9 taxonomy failure categories with 100% accuracy calibration on ground-truth benchmark datasets.
- **Distributed Tracing & OpenTelemetry Importer (`OTelImporter`)**:
  - Standard OTLP JSON import, OpenInference, OpenLLMetry, and LangSmith nested trace tree conversions with span hierarchy preservation.
- **In-Memory Tracing SDK & Middleware (`Tracer`)**:
  - Decorators (`@trace`, `@trace_llm`, `@trace_tool`, `@trace_retrieval`), SDK client wrappers (`wrap_openai`, `wrap_anthropic`, `wrap_tool`), and `contextvars`-based span management.
- **Streaming & Token Latency Profiler (`StreamingProfiler`)**:
  - Real-time token latency tracking, TTFT measurement, inter-token percentiles, tokens/sec throughput, and stream stall event detection.
- **Offline Standalone Interactive HTML Reports (`DiagnosticReportGenerator`)**:
  - Self-contained single-file HTML dashboards with visual execution waterfalls, expandable span inspectors, ranked hypotheses, evidence tables, actionable remediation cards, and dark/light mode toggle.
- **Telemetry Anonymization & PII Scrubbing (`PIIScrubber`, `TelemetryAnonymizer`)**:
  - Automated regex-based scrubbing of emails, phone numbers, SSNs, credit cards, IP addresses, passwords, and API keys/bearer tokens with configurable redaction masks.
- **Plugins & Extensibility SDK (`PluginManager`, `BasePlugin`, `BaseAnalyzerPlugin`, `BaseExporterPlugin`, `BaseMiddlewarePlugin`)**:
  - Registration hooks and lifecycle execution for third-party custom analyzers, metrics exporters, and middleware.
- **Live Web Dashboard & Local Diagnostic Visualizer Server (`DiagnosticServer`, `ServerConfig`)**:
  - Built-in zero-dependency Python HTTP server providing real-time trace ingestion, REST API endpoints (`/api/traces`, `/api/diagnose`, `/api/health`, `/api/upload`), and interactive web visualizer.
- **CI/CD Evaluation Harness & Gatekeeper Action (`Gatekeeper`, `GatekeeperConfig`, `evaluate_gate`)**:
  - SLA gatekeeper enforcing failure rate limits, comparative regression limits, latency percentiles, grounding scores, and prohibited failure categories in CI pull requests with PR markdown comments.
- **Terminal CLI (`llm-reliability`)**:
  - Subcommands `diagnose`, `verify`, `compare`, `serve`, and `gate` supporting `text`, `json`, `markdown`, and `html` format outputs.
- **Developer Workflow & Distribution**:
  - 8 real-world runnable examples under `examples/`.
  - Comprehensive 25-file documentation suite under `docs/`.
  - Multi-OS GitHub Actions CI/CD matrix (`ubuntu-latest`, `macos-latest`, `windows-latest`) on Python 3.10 through 3.13.
  - Multi-stage non-root container distribution (`Dockerfile`, `.dockerignore`, `.github/workflows/docker.yml`).
  - OIDC Trusted Publishing workflows for TestPyPI and Production PyPI with PEP 740 provenance attestations.
  - Pre-commit hooks configuration and developer `Makefile`.
