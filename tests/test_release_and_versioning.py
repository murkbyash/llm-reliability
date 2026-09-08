"""Automated tests validating semantic versioning, changelog structure, and release automation."""

import re
from pathlib import Path

import pytest

from scripts.bump_version import bump_version_string, parse_semver


class TestReleaseAndVersioning:
    """Validate release tagging, changelog formatting, and version bump automation."""

    def test_changelog_structure(self, project_root: Path) -> None:
        changelog_path = project_root / "CHANGELOG.md"
        assert changelog_path.is_file(), "CHANGELOG.md is missing from repository root"

        content = changelog_path.read_text(encoding="utf-8")

        # Keep a Changelog headers
        assert "# Changelog" in content
        assert "## [Unreleased]" in content
        assert "## [0.1.0]" in content
        assert "### Added" in content

        # Key capability sections documented
        assert "RAGAnalyzer" in content
        assert "GroundingAnalyzer" in content
        assert "AgentAnalyzer" in content
        assert "DiagnosticEngine" in content
        assert "VerificationEngine" in content
        assert "RegressionEngine" in content

    def test_version_consistency_across_files(self, project_root: Path) -> None:
        pyproject_path = project_root / "pyproject.toml"
        init_path = project_root / "src" / "llm_reliability" / "__init__.py"
        master_context_path = project_root / "MASTER_CONTEXT.md"

        pyproject_content = pyproject_path.read_text(encoding="utf-8")
        init_content = init_path.read_text(encoding="utf-8")
        master_context_content = master_context_path.read_text(encoding="utf-8")

        # Extract version from pyproject.toml
        match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
        assert match is not None, "Version not found in pyproject.toml"
        version = match.group(1)

        # Verify matching version in __init__.py and MASTER_CONTEXT.md
        assert f'__version__ = "{version}"' in init_content
        assert (
            f"Current version:** `{version}`" in master_context_content
            or f"`{version}`" in master_context_content
        )

    def test_release_drafter_configuration_and_workflow(self, project_root: Path) -> None:
        config_path = project_root / ".github" / "release-drafter.yml"
        assert config_path.is_file(), ".github/release-drafter.yml is missing"

        config_content = config_path.read_text(encoding="utf-8")
        assert "name-template:" in config_content
        assert "categories:" in config_content
        assert "Features" in config_content
        assert "Bug Fixes" in config_content

        workflow_path = project_root / ".github" / "workflows" / "release-drafter.yml"
        assert workflow_path.is_file(), ".github/workflows/release-drafter.yml is missing"

        workflow_content = workflow_path.read_text(encoding="utf-8")
        assert "Release Notes Drafter" in workflow_content
        assert "release-drafter/release-drafter" in workflow_content

    def test_semver_parsing_and_bumping_logic(self) -> None:
        # Test semver parsing
        major, minor, patch, pre = parse_semver("0.1.0.dev0")
        assert major == 0
        assert minor == 1
        assert patch == 0
        assert pre == "dev0"

        major2, minor2, patch2, pre2 = parse_semver("1.2.3")
        assert (major2, minor2, patch2, pre2) == (1, 2, 3, "")

        # Test bumps
        assert bump_version_string("0.1.0.dev0", "release") == "0.1.0"
        assert bump_version_string("0.1.0", "patch") == "0.1.1"
        assert bump_version_string("0.1.0", "minor") == "0.2.0"
        assert bump_version_string("0.1.0", "major") == "1.0.0"
        assert bump_version_string("0.1.0", "dev") == "0.1.0.dev0"
        assert bump_version_string("0.1.0", "rc") == "0.1.0.rc1"

        with pytest.raises(ValueError):
            parse_semver("invalid-version-string")

        with pytest.raises(ValueError):
            bump_version_string("0.1.0", "invalid_bump_type")
