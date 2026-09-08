"""Pytest configuration and shared fixtures for LLM Reliability Analyzer."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent
