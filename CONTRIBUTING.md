# Contributing to LLM Reliability Analyzer

Thanks for your interest in contributing. This project is a local-first
root-cause diagnosis engine for LLM/RAG/Agent execution traces, and
contributions of all sizes are welcome — bug reports, new failure-detection
heuristics, analyzer improvements, docs fixes, and new examples.

By participating, you're expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

```bash
git clone https://github.com/murkbyash/llm-reliability.git
cd llm-reliability
pip install -e ".[dev]"
```

This installs the package in editable mode plus the dev toolchain (`pytest`,
`ruff`, `mypy`, `build`).

## Development Workflow

A `Makefile` wraps the common tasks:

```bash
make test        # run the test suite
make lint        # ruff lint checks
make format      # ruff auto-format
make typecheck   # mypy --strict
make cov         # tests + branch coverage report
make benchmark   # run the synthetic diagnostic accuracy benchmark
make check       # lint + format check + typecheck + test, i.e. what CI runs
```

Run `make check` before opening a PR — it mirrors the `lint-and-typecheck`
and `test-matrix` jobs in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install
```

This runs `ruff`, `ruff format`, `mypy`, and `bandit` automatically on every
commit (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

## Code Standards

- **Type hints are required.** The project runs `mypy --strict`; all new
  public functions need full type annotations.
- **Pydantic v2 models** are the canonical data shape for traces, spans, and
  diagnostic output — extend [`src/llm_reliability/models/`](src/llm_reliability/models)
  rather than passing around raw dicts.
- **No comments explaining *what* code does** — code should be self
  explanatory through naming; only comment non-obvious *why*.
- **Deterministic over probabilistic.** This project's value proposition is
  reproducible, explainable diagnosis without calling out to another LLM as a
  judge. New analyzers should stick to that philosophy — rule/heuristic/metric
  based detection with an evidence trail, not opaque LLM scoring.

## Adding a New Failure Detector

Most new detection logic follows the same shape as the existing analyzers
(see [`src/llm_reliability/rag/analyzer.py`](src/llm_reliability/rag/analyzer.py)
or [`src/llm_reliability/agent/analyzer.py`](src/llm_reliability/agent/analyzer.py)):

1. Add/extend a `*Metrics` model describing what you measure.
2. Implement the detection logic in the relevant analyzer.
3. Wire it into [`DiagnosticEngine.diagnose_run`](src/llm_reliability/diagnosis/engine.py)
   so it can contribute a `Failure` and a scored `Hypothesis`.
4. Add a labeled synthetic trace to the benchmark suite in
   [`src/llm_reliability/benchmark/generator.py`](src/llm_reliability/benchmark/generator.py)
   so accuracy is measurable, not just tested for "doesn't crash."
5. Add unit tests and update the relevant guide under [`docs/`](docs).

If you'd rather not modify the core engine, consider a
[plugin](docs/plugins.md) instead (`BaseAnalyzerPlugin`).

## Tests

- New behavior needs test coverage. Run `make cov` and check the diff isn't
  regressing coverage on the files you touched.
- Prefer a small, labeled synthetic trace over a mocked object graph — traces
  are the actual unit of interest in this project.

## Submitting a Pull Request

1. Fork the repo and create a branch from `main`.
2. Make your change with tests and, if user-facing, a docs update.
3. Run `make check` locally.
4. Update [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]`.
5. Open a PR describing *why* the change is needed, not just what changed.

## Reporting Bugs / Requesting Features

Use the [issue templates](.github/ISSUE_TEMPLATE) — bug reports should
include a minimal trace (JSON) that reproduces the problem where possible,
since traces are the primary input this project reasons about.

## Security Issues

Do not open a public issue for a security vulnerability — see
[SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## Questions

See [SUPPORT.md](SUPPORT.md).
