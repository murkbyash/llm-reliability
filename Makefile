.PHONY: help install test lint format typecheck cov benchmark check clean

help:
	@echo "LLM Reliability Analyzer - Developer Tasks"
	@echo "=========================================="
	@echo "make install    : Install package and development dependencies"
	@echo "make test       : Run test suite with pytest"
	@echo "make lint       : Run ruff lint checks"
	@echo "make format     : Format codebase with ruff"
	@echo "make typecheck  : Run strict type checking with mypy"
	@echo "make cov        : Run test suite with branch coverage"
	@echo "make benchmark  : Run diagnostic calibration benchmark"
	@echo "make check      : Run full verification (lint, format check, mypy, test)"
	@echo "make clean      : Remove build artifacts and caches"

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src tests

cov:
	pytest --cov=llm_reliability --cov-report=term-missing --cov-report=xml:coverage.xml -v

benchmark:
	python -c "from llm_reliability import run_benchmark; r = run_benchmark(samples_per_category=5); print(f'Accuracy: {r.overall_accuracy*100:.1f}%, F1: {r.macro_f1:.2f}'); assert r.passed"

check: lint format typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .coverage coverage.xml .pytest_cache .mypy_cache .ruff_cache

