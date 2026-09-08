#!/usr/bin/env python3
"""Semantic version bumping utility script for LLM Reliability Analyzer."""

import argparse
import re
import sys
from pathlib import Path


def parse_semver(version: str) -> tuple[int, int, int, str]:
    """Parse semver string into major, minor, patch, and prerelease tag."""
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:\.?(a|b|rc|dev)(\d+)?)?$"
    match = re.match(pattern, version.strip())
    if not match:
        raise ValueError(f"Invalid semantic version string: '{version}'")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    prerelease_type = match.group(4) or ""
    prerelease_num = match.group(5) or ""

    prerelease = f"{prerelease_type}{prerelease_num}"
    return major, minor, patch, prerelease


def bump_version_string(current_version: str, bump_type: str) -> str:
    """Compute new semantic version string given bump type."""
    major, minor, patch, prerelease = parse_semver(current_version)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "release":
        # Strip dev/rc suffix
        return f"{major}.{minor}.{patch}"
    elif bump_type.startswith("dev"):
        return f"{major}.{minor}.{patch}.dev0"
    elif bump_type.startswith("rc"):
        return f"{major}.{minor}.{patch}.rc1"
    else:
        raise ValueError(f"Unknown bump type: '{bump_type}'")


def update_file_version(file_path: Path, current_version: str, new_version: str) -> bool:
    """Replace version string in given file."""
    if not file_path.is_file():
        return False

    content = file_path.read_text(encoding="utf-8")
    if current_version not in content:
        return False

    new_content = content.replace(current_version, new_version)
    file_path.write_text(new_content, encoding="utf-8")
    return True


def get_current_version(pyproject_path: Path) -> str:
    """Read version from pyproject.toml."""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def main() -> int:
    """Run CLI version bumper."""
    parser = argparse.ArgumentParser(description="Bump package version")
    parser.add_argument(
        "bump",
        choices=["major", "minor", "patch", "release", "dev", "rc"],
        help="Type of version bump to apply",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    current = get_current_version(pyproject_path)
    new = bump_version_string(current, args.bump)

    print(f"Bumping version from {current} -> {new}")

    files_to_update = [
        project_root / "pyproject.toml",
        project_root / "src" / "llm_reliability" / "__init__.py",
        project_root / "MASTER_CONTEXT.md",
    ]

    for path in files_to_update:
        if update_file_version(path, current, new):
            print(f"  Updated: {path.name}")
        else:
            print(f"  Warning: Version not found in {path.name}")

    print("Version bump complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
