"""Tests validating containerization configurations, Dockerfile, and Docker CI workflows."""

from pathlib import Path


class TestContainerization:
    """Validate Dockerfile specifications and multi-arch Docker CI workflow."""

    def test_dockerfile_structure(self, project_root: Path) -> None:
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.is_file(), "Dockerfile is missing from repository root"

        content = dockerfile_path.read_text(encoding="utf-8")

        # Multi-stage build
        assert "AS builder" in content
        assert "AS runner" in content

        # Non-root user security practice
        assert "useradd" in content or "adduser" in content
        assert "appuser" in content
        assert "USER appuser" in content

        # CLI Entrypoint
        assert 'ENTRYPOINT ["llm-reliability"]' in content
        assert "WORKDIR /data" in content

    def test_dockerignore_rules(self, project_root: Path) -> None:
        dockerignore_path = project_root / ".dockerignore"
        assert dockerignore_path.is_file(), ".dockerignore is missing from repository root"

        content = dockerignore_path.read_text(encoding="utf-8")
        assert ".git/" in content
        assert ".venv/" in content
        assert "tests/" in content
        assert "dist/" in content

    def test_docker_workflow_structure(self, project_root: Path) -> None:
        workflow_path = project_root / ".github" / "workflows" / "docker.yml"
        assert workflow_path.is_file(), ".github/workflows/docker.yml is missing"

        content = workflow_path.read_text(encoding="utf-8")

        # Multi-architecture support
        assert "docker/setup-qemu-action" in content
        assert "docker/setup-buildx-action" in content
        assert "platforms: linux/amd64,linux/arm64" in content

        # Registry and push
        assert "ghcr.io" in content
        assert "docker/login-action" in content
        assert "docker/build-push-action" in content

        # Security scan
        assert "aquasecurity/trivy-action" in content
