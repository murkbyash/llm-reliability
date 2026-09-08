"""Tests for package installation, importability, metadata, and project configuration."""

from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

import llm_reliability


def test_package_import_and_version() -> None:
    """Verify package imports correctly and has a valid semantic version string."""
    assert hasattr(llm_reliability, "__version__")
    assert isinstance(llm_reliability.__version__, str)
    assert len(llm_reliability.__version__) > 0
    assert llm_reliability.__version__ == "0.1.0"


def test_package_all_export() -> None:
    """Verify __all__ is properly defined and contains expected symbols."""
    assert hasattr(llm_reliability, "__all__")
    assert "__version__" in llm_reliability.__all__
    assert "Trace" in llm_reliability.__all__
    assert "Diagnosis" in llm_reliability.__all__
    assert "load_trace" in llm_reliability.__all__
    assert "normalize_trace" in llm_reliability.__all__
    assert "RAGAnalyzer" in llm_reliability.__all__
    assert "RetrievalMetrics" in llm_reliability.__all__
    assert "GroundingAnalyzer" in llm_reliability.__all__
    assert "GroundingMetrics" in llm_reliability.__all__
    assert "ClaimSupport" in llm_reliability.__all__
    assert "SupportStatus" in llm_reliability.__all__
    assert "AgentAnalyzer" in llm_reliability.__all__
    assert "AgentMetrics" in llm_reliability.__all__
    assert "ToolMetrics" in llm_reliability.__all__
    assert "AgentLoopPattern" in llm_reliability.__all__
    assert "ToolCallEvaluation" in llm_reliability.__all__
    assert "DiagnosticEngine" in llm_reliability.__all__
    assert "RecommendationEngine" in llm_reliability.__all__
    assert "VerificationEngine" in llm_reliability.__all__
    assert "VerificationReport" in llm_reliability.__all__
    assert "CounterfactualResult" in llm_reliability.__all__
    assert "MetricComparison" in llm_reliability.__all__
    assert "RegressionEngine" in llm_reliability.__all__
    assert "BatchComparisonReport" in llm_reliability.__all__
    assert "RegressionVerdict" in llm_reliability.__all__
    assert "DistributionSummary" in llm_reliability.__all__
    assert "FailureRateSummary" in llm_reliability.__all__
    assert "PatternDetectionEngine" in llm_reliability.__all__
    assert "PatternAnalysisReport" in llm_reliability.__all__
    assert "FailureCluster" in llm_reliability.__all__
    assert "FailureSignature" in llm_reliability.__all__
    assert "diagnose" in llm_reliability.__all__


def test_pep561_py_typed_exists(project_root: Path) -> None:
    """Verify PEP 561 py.typed marker is present in package directory."""
    py_typed = project_root / "src" / "llm_reliability" / "py.typed"
    assert py_typed.is_file(), "py.typed marker missing from package directory"


def test_pyproject_configuration(project_root: Path) -> None:
    """Verify pyproject.toml is valid TOML and contains all mandatory packaging fields."""
    pyproject_file = project_root / "pyproject.toml"
    assert pyproject_file.is_file(), "pyproject.toml not found"

    with open(pyproject_file, "rb") as f:
        config = tomllib.load(f)

    # Build system
    assert "build-system" in config
    assert config["build-system"]["build-backend"] == "hatchling.build"
    assert "hatchling" in config["build-system"]["requires"]

    # Project metadata
    project = config.get("project", {})
    assert project.get("name") == "llm-reliability"
    assert project.get("version") == "0.1.0"
    assert project.get("requires-python") == ">=3.10"
    assert project.get("license", {}).get("text") == "Apache-2.0"
    assert "pydantic>=2.0.0" in project.get("dependencies", [])

    # Scripts / CLI entrypoints
    assert "scripts" in project
    assert project["scripts"].get("llm-reliability") == "llm_reliability.cli:main"


def test_essential_documentation_files(project_root: Path) -> None:
    """Verify essential project documentation and governance files exist and are populated."""
    for filename in ["README.md", "LICENSE", ".gitignore"]:
        file_path = project_root / filename
        assert file_path.is_file(), f"Required file {filename} missing"
        assert file_path.stat().st_size > 50, f"Required file {filename} is unexpectedly empty"
