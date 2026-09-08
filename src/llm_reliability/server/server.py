"""Zero-dependency local HTTP visualizer server orchestrator."""

import logging
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.normalization.loader import load_trace
from llm_reliability.server.handler import DiagnosticHTTPRequestHandler
from llm_reliability.server.models import ServerConfig, TraceRecord

logger = logging.getLogger(__name__)


class DiagnosticServer:
    """Standalone local HTTP server for interactive trace visualizer and REST APIs."""

    def __init__(self, config: ServerConfig | None = None) -> None:
        self.config = config or ServerConfig()
        self._records: dict[str, TraceRecord] = {}
        self._lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._engine = DiagnosticEngine()

        # Build custom request handler class bound to this server instance
        class BoundHandler(DiagnosticHTTPRequestHandler):
            server_instance = self

        self._server = ThreadingHTTPServer((self.config.host, self.config.port), BoundHandler)
        # Update port in case port 0 (ephemeral) was requested
        self.port = self._server.server_port
        self.host = self.config.host

        if self.config.trace_dir is not None:
            self.preload_traces(self.config.trace_dir)

    def add_record(self, record: TraceRecord) -> None:
        """Thread-safe registration of a TraceRecord."""
        with self._lock:
            self._records[record.trace_id] = record

    def get_record(self, trace_id: str) -> TraceRecord | None:
        """Thread-safe retrieval of a TraceRecord by trace_id."""
        with self._lock:
            return self._records.get(trace_id)

    def get_records(self) -> list[TraceRecord]:
        """Thread-safe list of all registered TraceRecords."""
        with self._lock:
            return list(self._records.values())

    def preload_traces(self, directory: Path | str) -> int:
        """Scan directory and load all trace JSON/JSONL files into session."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"Trace directory '{directory}' does not exist.")
            return 0

        loaded_count = 0
        for path in sorted(dir_path.glob("*.json")):
            try:
                trace = load_trace(path)
                diag = self._engine.diagnose(trace)

                record = TraceRecord(
                    trace_id=trace.trace_id,
                    trace=trace,
                    diagnosis=diag,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_filename=path.name,
                )
                self.add_record(record)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not preload trace file '{path.name}': {e}")

        return loaded_count

    def start(self, background: bool = False) -> None:
        """Start the HTTP server."""
        server_url = f"http://{self.host}:{self.port}"
        logger.info(f"Starting LLM Reliability Visualizer at {server_url}")

        if self.config.auto_open:
            webbrowser.open(server_url)

        if background:
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                name="DiagnosticServerThread",
                daemon=True,
            )
            self._server_thread.start()
        else:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                logger.info("Server stopped by user.")
            finally:
                self.stop()

    def stop(self) -> None:
        """Shutdown the HTTP server and close socket listener."""
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception as e:
            logger.warning(f"Error during server shutdown: {e}")

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)


def start_server(config: ServerConfig | None = None, background: bool = False) -> DiagnosticServer:
    """Convenience helper to initialize and start the visualizer server."""
    server = DiagnosticServer(config)
    server.start(background=background)
    return server
