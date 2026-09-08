"""Automated tests validating packaging metadata, wheel contents, and PyPI/TestPyPI workflows."""

import subprocess
import sys
import zipfile
from pathlib import Path


class TestPackagingAndPublishPipeline:
    """Validate distribution packaging and PyPI/TestPyPI CI/CD publishing workflows."""

    def test_publish_testpypi_workflow_structure(self, project_root: Path) -> None:
        workflow_path = project_root / ".github" / "workflows" / "publish-testpypi.yml"
        assert workflow_path.is_file(), "publish-testpypi.yml workflow file is missing"

        content = workflow_path.read_text(encoding="utf-8")

        # Verify key workflow requirements
        assert "Publish to TestPyPI" in content
        assert "v*.*.*rc*" in content
        assert "testpypi-*" in content
        assert "workflow_dispatch:" in content
        assert "id-token: write" in content
        assert "pypa/gh-action-pypi-publish" in content
        assert "https://test.pypi.org/legacy/" in content
        assert "twine check --strict" in content
        assert "smoke-test:" in content

    def test_publish_production_pypi_workflow_structure(self, project_root: Path) -> None:
        workflow_path = project_root / ".github" / "workflows" / "publish-pypi.yml"
        assert workflow_path.is_file(), "publish-pypi.yml workflow file is missing"

        content = workflow_path.read_text(encoding="utf-8")

        # Verify key workflow requirements
        assert "Publish to Production PyPI" in content
        assert "release:" in content
        assert "types: [published]" in content
        assert "pre-publish-checks:" in content
        assert "id-token: write" in content
        assert "contents: write" in content
        assert "attestations: write" in content
        assert "pypa/gh-action-pypi-publish" in content
        assert "softprops/action-gh-release" in content
        assert "twine check --strict" in content
        assert "smoke-test:" in content

    def test_wheel_archive_contents(self, project_root: Path, tmp_path: Path) -> None:
        dist_dir = project_root / "dist"
        wheels = list(dist_dir.glob("*.whl")) if dist_dir.is_dir() else []

        if not wheels:
            # No pre-built wheel on disk (e.g. a fresh CI checkout) - build one
            # into a scratch directory rather than requiring a prior build step.
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
                cwd=project_root,
                check=True,
                capture_output=True,
            )
            wheels = list(tmp_path.glob("*.whl"))

        assert len(wheels) > 0, "No built wheel files found"

        wheel_file = wheels[0]
        with zipfile.ZipFile(wheel_file, "r") as z:
            names = z.namelist()

            # Ensure essential files and typing marker exist
            assert any(name.endswith("py.typed") for name in names), (
                "Missing py.typed marker in wheel"
            )
            assert any(name.endswith("llm_reliability/__init__.py") for name in names), (
                "Missing root __init__.py in wheel"
            )
            assert any(name.endswith("llm_reliability/cli/main.py") for name in names), (
                "Missing CLI entrypoint in wheel"
            )
            assert any(name.endswith("llm_reliability/diagnosis/engine.py") for name in names), (
                "Missing diagnosis engine in wheel"
            )
            assert any("entry_points.txt" in name for name in names), (
                "Missing entry_points.txt in wheel metadata"
            )

            # Ensure no test or scratch files leaked into wheel
            assert not any("tests/" in name for name in names), (
                "Test files accidentally included in wheel"
            )
            assert not any(".pytest_cache" in name for name in names), (
                "Cache files accidentally included in wheel"
            )

    def test_pyproject_toml_distribution_metadata(self, project_root: Path) -> None:
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")

        assert 'name = "llm-reliability"' in content
        assert 'version = "0.1.0.dev0"' in content
        assert (
            'license = { text = "Apache-2.0" }' in content
            or 'license = "Apache-2.0"' in content
            or "Apache" in content
        )
        assert 'llm-reliability = "llm_reliability.cli:main"' in content
        assert "Documentation" in content
        assert "Repository" in content
