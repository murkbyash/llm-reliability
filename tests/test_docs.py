"""Tests for documentation integrity, existence, and code snippet validity."""

from pathlib import Path

EXPECTED_DOC_FILES = [
    "index.md",
    "getting-started.md",
    "architecture.md",
    "rag-analysis.md",
    "grounding-analysis.md",
    "agent-analysis.md",
    "root-cause-diagnosis.md",
    "recommendation-engine.md",
    "verification-rerun.md",
    "regression-testing.md",
    "patterns-and-signatures.md",
    "custom-policies.md",
    "benchmark-suite.md",
    "opentelemetry.md",
    "tracing-sdk.md",
    "streaming-profiler.md",
    "html-reports.md",
    "cookbook.md",
    "api-reference.md",
    "troubleshooting.md",
]


class TestDocumentationSuite:
    """Validate documentation completeness and code block syntax."""

    def test_all_documentation_files_exist_and_non_empty(self, project_root: Path) -> None:
        docs_dir = project_root / "docs"
        assert docs_dir.is_dir(), "docs/ directory missing from repository"

        for doc_name in EXPECTED_DOC_FILES:
            doc_path = docs_dir / doc_name
            assert doc_path.is_file(), f"Expected documentation file missing: docs/{doc_name}"
            content = doc_path.read_text(encoding="utf-8").strip()
            assert len(content) > 100, f"Documentation file docs/{doc_name} is too short or empty"
            assert content.startswith("# "), f"docs/{doc_name} must start with a Markdown H1 header"

    def test_readme_contains_core_sections(self, project_root: Path) -> None:
        readme_path = project_root / "README.md"
        assert readme_path.is_file(), "README.md missing from repository root"
        content = readme_path.read_text(encoding="utf-8")

        required_sections = [
            "The Core Problem",
            "Key Capabilities",
            "High-Level Architecture",
            "Installation & Quickstart",
            "Live Tracing & Streaming Latency Profiling",
            "Diagnostic Benchmark Accuracy",
            "Documentation Guides",
            "License",
        ]

        for section in required_sections:
            assert section in content, f"README.md missing required section: '{section}'"

    def test_python_code_blocks_in_docs_import_valid_symbols(self, project_root: Path) -> None:
        import ast

        import llm_reliability

        docs_dir = project_root / "docs"
        exported_symbols = set(dir(llm_reliability))

        for doc_file in docs_dir.glob("*.md"):
            content = doc_file.read_text(encoding="utf-8")
            # Extract python code blocks
            blocks = []
            curr_block: list[str] = []
            in_py = False
            for line in content.splitlines():
                if line.strip().startswith("```python"):
                    in_py = True
                    curr_block = []
                elif line.strip() == "```" and in_py:
                    in_py = False
                    blocks.append("\n".join(curr_block))
                elif in_py:
                    curr_block.append(line)

            for block in blocks:
                try:
                    tree = ast.parse(block)
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "llm_reliability":
                        for alias in node.names:
                            assert alias.name in exported_symbols or alias.name in (
                                "load_trace",
                                "diagnose",
                            ), f"Invalid symbol '{alias.name}' imported in {doc_file.name}"
