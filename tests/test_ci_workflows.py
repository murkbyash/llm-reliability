"""Tests validating GitHub Actions CI/CD workflows and automated quality gates."""

from pathlib import Path


class TestCIWorkflows:
    """Validate existence, matrix configurations, and quality steps in GitHub Actions workflows."""

    def test_workflow_files_exist(self, project_root: Path) -> None:
        workflows_dir = project_root / ".github" / "workflows"
        assert workflows_dir.is_dir(), ".github/workflows directory missing"

        expected_workflows = ["ci.yml", "benchmark.yml", "docs.yml"]
        for wf_name in expected_workflows:
            wf_path = workflows_dir / wf_name
            assert wf_path.is_file(), f"Expected workflow missing: .github/workflows/{wf_name}"
            content = wf_path.read_text(encoding="utf-8")
            assert len(content) > 50, f"Workflow {wf_name} is unexpectedly empty"

    def test_ci_matrix_configuration(self, project_root: Path) -> None:
        ci_path = project_root / ".github" / "workflows" / "ci.yml"
        assert ci_path.is_file()
        content = ci_path.read_text(encoding="utf-8")

        # Multi-OS matrix
        assert "ubuntu-latest" in content
        assert "macos-latest" in content
        assert "windows-latest" in content

        # Multi-Python matrix
        assert '"3.10"' in content
        assert '"3.11"' in content
        assert '"3.12"' in content
        assert '"3.13"' in content

        # Quality Gates
        assert "ruff check" in content
        assert "ruff format" in content
        assert "mypy" in content
        assert "pytest --cov" in content
        assert "llm-reliability --version" in content

    def test_benchmark_workflow_configuration(self, project_root: Path) -> None:
        bm_path = project_root / ".github" / "workflows" / "benchmark.yml"
        assert bm_path.is_file()
        content = bm_path.read_text(encoding="utf-8")

        assert "run_benchmark" in content
        assert "assert r.passed" in content

    def test_docs_workflow_configuration(self, project_root: Path) -> None:
        docs_path = project_root / ".github" / "workflows" / "docs.yml"
        assert docs_path.is_file()
        content = docs_path.read_text(encoding="utf-8")

        assert "pytest tests/test_docs.py" in content
