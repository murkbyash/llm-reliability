"""Automated security tests validating SAST configuration, CodeQL workflows, and SECURITY.md."""

from pathlib import Path


class TestSecurityAndAuditing:
    """Validate repository security policies, SAST configurations, and CI vulnerability workflows."""

    def test_security_md_policy(self, project_root: Path) -> None:
        security_path = project_root / "SECURITY.md"
        assert security_path.is_file(), "SECURITY.md is missing from repository root"

        content = security_path.read_text(encoding="utf-8")
        assert "Security Policy" in content
        assert "Supported Versions" in content
        assert "Reporting a Vulnerability" in content

    def test_bandit_configuration_in_pyproject(self, project_root: Path) -> None:
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")

        assert "[tool.bandit]" in content
        assert 'targets = ["src/llm_reliability"]' in content
        assert "exclude_dirs" in content
        assert "skips" in content

    def test_security_workflow_structure(self, project_root: Path) -> None:
        workflow_path = project_root / ".github" / "workflows" / "security.yml"
        assert workflow_path.is_file(), ".github/workflows/security.yml is missing"

        content = workflow_path.read_text(encoding="utf-8")

        # Key security jobs
        assert "codeql:" in content
        assert "github/codeql-action/init" in content
        assert "github/codeql-action/analyze" in content

        assert "bandit:" in content
        assert "bandit -r src/llm_reliability" in content

        assert "dependency-audit:" in content
        assert "pip-audit" in content

        # Scheduled trigger
        assert "schedule:" in content
        assert "cron:" in content

    def test_pre_commit_contains_bandit(self, project_root: Path) -> None:
        pre_commit_path = project_root / ".pre-commit-config.yaml"
        content = pre_commit_path.read_text(encoding="utf-8")

        assert "PyCQA/bandit" in content
        assert "id: bandit" in content
