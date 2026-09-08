"""HTTP Request handler for the local diagnostic visualizer server."""

import json
import logging
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.normalization.normalizer import normalize_trace
from llm_reliability.report.generator import render_html_report
from llm_reliability.server.models import TraceRecord

SERVER_VERSION = "0.1.0.dev0"


logger = logging.getLogger(__name__)


class DiagnosticHTTPRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP request handler serving REST APIs and interactive HTML dashboard."""

    # Reference to parent DiagnosticServer set during server initialization
    server_instance: Any = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr HTTP access logging to keep CLI output clean."""
        pass

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        """Send JSON response with CORS headers."""
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_content: str, status: int = HTTPStatus.OK) -> None:
        """Send HTML response."""
        payload = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")

        # Health endpoint
        if path == "/api/health":
            records = self.server_instance.get_records() if self.server_instance else []
            self._send_json(
                {
                    "status": "healthy",
                    "version": SERVER_VERSION,
                    "traces_count": len(records),
                }
            )
            return

        # List traces
        if path == "/api/traces":
            records = self.server_instance.get_records() if self.server_instance else []
            summaries = [
                {
                    "trace_id": r.trace_id,
                    "created_at": r.created_at,
                    "root_cause": r.diagnosis.primary_category.value,
                    "severity": r.diagnosis.severity.value,
                    "failures_count": len(r.diagnosis.failures),
                    "summary": r.diagnosis.summary,
                    "source_filename": r.source_filename,
                }
                for r in records
            ]
            self._send_json({"traces": summaries, "total": len(summaries)})
            return

        # Single trace lookup: /api/traces/<trace_id>
        if path.startswith("/api/traces/"):
            trace_id = path[len("/api/traces/") :]
            record = self.server_instance.get_record(trace_id) if self.server_instance else None
            if record is None:
                self._send_json(
                    {"error": f"Trace with ID '{trace_id}' not found."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_json(
                {
                    "trace_id": record.trace_id,
                    "created_at": record.created_at,
                    "trace": record.trace.model_dump(),
                    "diagnosis": record.diagnosis.model_dump(),
                }
            )
            return

        # Interactive Dashboard HTML: Root "/"
        if path == "" or path == "/index.html":
            records = self.server_instance.get_records() if self.server_instance else []
            query_params = parse_qs(parsed_url.query)
            selected_id = query_params.get("trace_id", [None])[0]

            selected_record = None
            if selected_id:
                selected_record = self.server_instance.get_record(selected_id)
            elif records:
                selected_record = records[-1]  # Most recent trace

            if selected_record:
                html = render_html_report(selected_record.diagnosis, selected_record.trace)
            else:
                html = self._render_empty_dashboard_html()

            self._send_html(html)
            return

        self._send_json(
            {"error": f"Endpoint '{self.path}' not found."},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(
                {"error": "Empty request body."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        body = self.rfile.read(content_length)

        # Ingest and diagnose trace payload
        if path in ("/api/diagnose", "/api/upload"):
            try:
                raw_json = json.loads(body.decode("utf-8"))
                trace = normalize_trace(raw_json)
                engine = DiagnosticEngine()
                diagnosis = engine.diagnose(trace)

                now_iso = datetime.now(timezone.utc).isoformat()
                record = TraceRecord(
                    trace_id=trace.trace_id,
                    trace=trace,
                    diagnosis=diagnosis,
                    created_at=now_iso,
                )

                if self.server_instance:
                    self.server_instance.add_record(record)

                self._send_json(
                    {
                        "trace_id": trace.trace_id,
                        "status": "success",
                        "diagnosis": diagnosis.model_dump(),
                    }
                )
            except Exception as e:
                logger.error(f"Error processing trace in HTTP handler: {e}", exc_info=True)
                self._send_json(
                    {"error": f"Failed to ingest and diagnose trace: {str(e)}"},
                    status=HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            return

        self._send_json(
            {"error": f"POST endpoint '{self.path}' not supported."},
            status=HTTPStatus.NOT_FOUND,
        )

    def _render_empty_dashboard_html(self) -> str:
        """Render welcoming visualizer dashboard landing page when no traces are preloaded."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Reliability Analyzer — Live Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 800px;
            width: 100%;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        h1 {{
            color: #58a6ff;
            margin-top: 0;
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            background: #238636;
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .dropzone {{
            border: 2px dashed #388bfd;
            border-radius: 8px;
            padding: 40px 20px;
            text-align: center;
            background: rgba(56, 139, 253, 0.05);
            margin: 24px 0;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .dropzone:hover {{
            background: rgba(56, 139, 253, 0.12);
        }}
        .endpoint-card {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            font-family: monospace;
            font-size: 13px;
            color: #79c0ff;
            margin-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 LLM Reliability Visualizer <span class="badge">v{SERVER_VERSION}</span></h1>
        <p>Zero-cloud local observability server is running and ready to ingest execution traces.</p>


        <div class="dropzone" onclick="document.getElementById('fileInput').click()">
            <p><strong>Drop trace file here or click to upload (.json, .jsonl, OTel)</strong></p>
            <input type="file" id="fileInput" style="display:none" onchange="uploadFile(this.files[0])">
        </div>

        <h3>Active REST API Endpoints</h3>
        <div class="endpoint-card">
            POST /api/diagnose &mdash; Submit trace payload for instant root cause evaluation<br>
            GET /api/traces &mdash; List all ingested traces in current session<br>
            GET /api/health &mdash; Server health check
        </div>
    </div>

    <script>
        async function uploadFile(file) {{
            if (!file) return;
            const text = await file.text();
            try {{
                const res = await fetch('/api/diagnose', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: text
                }});
                const data = await res.json();
                if (res.ok) {{
                    window.location.reload();
                }} else {{
                    alert('Error: ' + (data.error || 'Failed to parse trace'));
                }}
            }} catch (err) {{
                alert('Upload failed: ' + err);
            }}
        }}
    </script>
</body>
</html>
"""
