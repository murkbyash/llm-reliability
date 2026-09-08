"""Tests for open-source governance file completeness and consistency."""

from pathlib import Path

GOVERNANCE_FILES = [
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
    "SECURITY.md",
    "LICENSE",
    "CHANGELOG.md",
]


class TestGovernanceFiles:
    """Validate that OSS governance/community files exist and are populated."""

    def test_governance_files_exist_and_non_empty(self, project_root: Path) -> None:
        for filename in GOVERNANCE_FILES:
            path = project_root / filename
            assert path.is_file(), f"Expected governance file missing: {filename}"
            content = path.read_text(encoding="utf-8").strip()
            assert len(content) > 200, f"{filename} is empty or too short to be meaningful"

    def test_contributing_references_dev_workflow(self, project_root: Path) -> None:
        content = (project_root / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for keyword in ["pytest", "ruff", "mypy", "Code of Conduct"]:
            assert keyword in content, f"CONTRIBUTING.md missing reference to '{keyword}'"

    def test_security_policy_has_reporting_contact(self, project_root: Path) -> None:
        content = (project_root / "SECURITY.md").read_text(encoding="utf-8")
        assert "@" in content, "SECURITY.md must list a reachable contact for vulnerability reports"

    def test_no_placeholder_or_unresolved_contact_info(self, project_root: Path) -> None:
        forbidden = ["your-email", "your.name", "TODO", "FIXME", "example@example.com"]
        for filename in GOVERNANCE_FILES:
            content = (project_root / filename).read_text(encoding="utf-8").lower()
            for token in forbidden:
                assert token.lower() not in content, f"{filename} contains unresolved placeholder '{token}'"

    def test_contact_email_consistent_across_governance_docs(self, project_root: Path) -> None:
        contact_email = "ashishuike8@gmail.com"
        security_content = (project_root / "SECURITY.md").read_text(encoding="utf-8")
        conduct_content = (project_root / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
        assert contact_email in security_content, "SECURITY.md contact email out of date"
        assert contact_email in conduct_content, "CODE_OF_CONDUCT.md contact email out of date"
