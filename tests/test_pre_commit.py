"""Tests validating pre-commit hook configuration and developer workflow tasks."""

from pathlib import Path


class TestPreCommitAndErgonomics:
    """Validate pre-commit configuration and developer ergonomics."""

    def test_pre_commit_config_exists_and_valid(self, project_root: Path) -> None:
        config_path = project_root / ".pre-commit-config.yaml"
        assert config_path.is_file(), ".pre-commit-config.yaml is missing"

        content = config_path.read_text(encoding="utf-8")
        assert len(content) > 100

        # Required hooks
        expected_hooks = [
            "trailing-whitespace",
            "end-of-file-fixer",
            "check-yaml",
            "check-toml",
            "check-added-large-files",
            "ruff",
            "ruff-format",
            "mypy",
        ]
        for hook in expected_hooks:
            assert hook in content, f"Pre-commit hook '{hook}' missing from configuration"

        # Pre-commit.ci automation
        assert "ci:" in content
        assert "autoupdate_schedule: weekly" in content

    def test_makefile_exists_and_contains_tasks(self, project_root: Path) -> None:
        makefile_path = project_root / "Makefile"
        assert makefile_path.is_file(), "Makefile is missing from repository root"

        content = makefile_path.read_text(encoding="utf-8")
        expected_targets = [
            "install:",
            "test:",
            "lint:",
            "format:",
            "typecheck:",
            "cov:",
            "benchmark:",
            "check:",
            "clean:",
        ]
        for target in expected_targets:
            assert target in content, f"Makefile target '{target}' is missing"
