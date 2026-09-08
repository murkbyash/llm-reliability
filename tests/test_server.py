"""Automated unit tests for local HTTP diagnostic visualizer server and REST APIs."""

import json
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest

from llm_reliability import DiagnosticServer, ServerConfig, start_server
from llm_reliability.cli.main import build_parser


@pytest.fixture
def running_server() -> Generator[DiagnosticServer, None, None]:
    """Fixture providing a background DiagnosticServer running on an ephemeral port."""
    config = ServerConfig(host="127.0.0.1", port=0, auto_open=False)
    server = start_server(config, background=True)
    yield server
    server.stop()


class TestDiagnosticServer:
    """Test REST endpoints, HTML dashboard rendering, trace ingestion, and CLI integration."""

    def test_health_endpoint(self, running_server: DiagnosticServer) -> None:
        url = f"http://127.0.0.1:{running_server.port}/api/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"
            assert "version" in data
            assert data["traces_count"] == 0

    def test_diagnose_post_and_trace_retrieval(self, running_server: DiagnosticServer) -> None:
        sample_trace = {
            "trace_id": "server-test-trace-1",
            "runs": [
                {
                    "run_id": "run-1",
                    "trace_id": "server-test-trace-1",
                    "spans": [
                        {
                            "span_id": "s1",
                            "name": "llm_step",
                            "kind": "llm",
                            "status": "error",
                            "error_message": "Rate limit exceeded 429",
                        }
                    ],
                }
            ],
        }

        # 1. POST /api/diagnose
        post_url = f"http://127.0.0.1:{running_server.port}/api/diagnose"
        payload = json.dumps(sample_trace).encode("utf-8")
        req = urllib.request.Request(
            post_url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res_data = json.loads(resp.read().decode("utf-8"))
            assert res_data["status"] == "success"
            assert res_data["trace_id"] == "server-test-trace-1"
            assert res_data["diagnosis"]["primary_category"] == "LLM_CALL_FAILURE"

        # 2. GET /api/traces
        list_url = f"http://127.0.0.1:{running_server.port}/api/traces"
        with urllib.request.urlopen(list_url) as resp:
            assert resp.status == 200
            traces_data = json.loads(resp.read().decode("utf-8"))
            assert traces_data["total"] == 1
            assert traces_data["traces"][0]["trace_id"] == "server-test-trace-1"
            assert traces_data["traces"][0]["root_cause"] == "LLM_CALL_FAILURE"

        # 3. GET /api/traces/<trace_id>
        get_url = f"http://127.0.0.1:{running_server.port}/api/traces/server-test-trace-1"
        with urllib.request.urlopen(get_url) as resp:
            assert resp.status == 200
            item_data = json.loads(resp.read().decode("utf-8"))
            assert item_data["trace_id"] == "server-test-trace-1"
            assert item_data["diagnosis"]["root_cause"] == "LLM_CALL_FAILURE"

        # 4. GET /api/traces/nonexistent -> 404
        bad_url = f"http://127.0.0.1:{running_server.port}/api/traces/unknown-id"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_url)
        assert exc_info.value.code == 404

    def test_dashboard_html_rendering(self, running_server: DiagnosticServer) -> None:
        root_url = f"http://127.0.0.1:{running_server.port}/"

        # Empty dashboard
        with urllib.request.urlopen(root_url) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers["Content-Type"]
            html = resp.read().decode("utf-8")
            assert "LLM Reliability Visualizer" in html
            assert "Drop trace file here" in html

        # Post trace
        post_url = f"http://127.0.0.1:{running_server.port}/api/diagnose"
        payload = json.dumps(
            {
                "trace_id": "trace-html-test",
                "runs": [
                    {
                        "run_id": "r1",
                        "trace_id": "trace-html-test",
                        "spans": [
                            {
                                "span_id": "s1",
                                "name": "root",
                                "kind": "root",
                                "status": "success",
                            }
                        ],
                    }
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            post_url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200

        # Active dashboard with report generator
        with urllib.request.urlopen(root_url) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "trace-html-test" in html or "Diagnosis" in html

    def test_preload_traces_from_directory(self, tmp_path: Path) -> None:
        trace_file1 = tmp_path / "trace1.json"
        trace_file2 = tmp_path / "trace2.json"

        trace_file1.write_text(
            json.dumps(
                {
                    "trace_id": "preload-1",
                    "runs": [
                        {
                            "run_id": "r1",
                            "trace_id": "preload-1",
                            "spans": [{"span_id": "s1", "name": "step", "status": "success"}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        trace_file2.write_text(
            json.dumps(
                {
                    "trace_id": "preload-2",
                    "runs": [
                        {
                            "run_id": "r2",
                            "trace_id": "preload-2",
                            "spans": [{"span_id": "s2", "name": "step", "status": "success"}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        config = ServerConfig(host="127.0.0.1", port=0, trace_dir=tmp_path, auto_open=False)
        server = start_server(config, background=True)
        try:
            assert len(server.get_records()) == 2
            url = f"http://127.0.0.1:{server.port}/api/traces"
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                assert data["total"] == 2
        finally:
            server.stop()

    def test_cors_options(self, running_server: DiagnosticServer) -> None:
        url = f"http://127.0.0.1:{running_server.port}/api/diagnose"
        req = urllib.request.Request(url, method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_error_handling(self, running_server: DiagnosticServer) -> None:
        # Invalid JSON -> 422
        post_url = f"http://127.0.0.1:{running_server.port}/api/diagnose"
        req = urllib.request.Request(
            post_url,
            data=b"invalid-not-json",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 422

        # 404 on unknown endpoint
        unknown_url = f"http://127.0.0.1:{running_server.port}/api/nonexistent"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(unknown_url)
        assert exc_info.value.code == 404

    def test_cli_serve_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--trace-dir",
                "/tmp/traces",
                "--open-browser",
            ]
        )
        assert args.command == "serve"
        assert args.host == "0.0.0.0"
        assert args.port == 9999
        assert args.trace_dir == "/tmp/traces"
        assert args.open_browser is True
