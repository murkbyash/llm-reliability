"""Tests verifying that all example scripts in examples/ execute cleanly."""

import importlib.util
from pathlib import Path

import pytest

EXPECTED_EXAMPLES = [
    "01_diagnose_rag_failure.py",
    "02_diagnose_agent_loop.py",
    "03_streaming_latency_profiling.py",
    "04_counterfactual_verification.py",
    "05_batch_regression_testing.py",
    "06_custom_policy_evaluation.py",
    "07_generate_interactive_html_report.py",
    "08_opentelemetry_import.py",
]


class TestExamplesSuite:
    """Validate that all standalone example scripts run without errors."""

    def test_all_example_files_exist(self, project_root: Path) -> None:
        examples_dir = project_root / "examples"
        assert examples_dir.is_dir(), "examples/ directory missing from repository"

        readme_file = examples_dir / "README.md"
        assert readme_file.is_file(), "examples/README.md missing"

        for script_name in EXPECTED_EXAMPLES:
            script_path = examples_dir / script_name
            assert script_path.is_file(), f"Example script missing: examples/{script_name}"
            content = script_path.read_text(encoding="utf-8")
            assert len(content) > 100, f"Example script examples/{script_name} is too short"
            assert 'if __name__ == "__main__":' in content, (
                f"Script {script_name} must have main block"
            )

    def test_run_example_01_rag_failure(
        self, project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        script_path = project_root / "examples" / "01_diagnose_rag_failure.py"
        spec = importlib.util.spec_from_file_location("ex01", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_02_agent_loop(self, project_root: Path) -> None:
        script_path = project_root / "examples" / "02_diagnose_agent_loop.py"
        spec = importlib.util.spec_from_file_location("ex02", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_03_streaming(self, project_root: Path) -> None:
        script_path = project_root / "examples" / "03_streaming_latency_profiling.py"
        spec = importlib.util.spec_from_file_location("ex03", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_04_verification(self, project_root: Path) -> None:
        script_path = project_root / "examples" / "04_counterfactual_verification.py"
        spec = importlib.util.spec_from_file_location("ex04", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_05_regression(self, project_root: Path) -> None:
        script_path = project_root / "examples" / "05_batch_regression_testing.py"
        spec = importlib.util.spec_from_file_location("ex05", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_06_policy(self, project_root: Path) -> None:
        script_path = project_root / "examples" / "06_custom_policy_evaluation.py"
        spec = importlib.util.spec_from_file_location("ex06", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_07_html_report(
        self, project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        script_path = project_root / "examples" / "07_generate_interactive_html_report.py"
        spec = importlib.util.spec_from_file_location("ex07", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def test_run_example_08_otel_import(self, project_root: Path) -> None:
        script_path = project_root / "examples" / "08_opentelemetry_import.py"
        spec = importlib.util.spec_from_file_location("ex08", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
