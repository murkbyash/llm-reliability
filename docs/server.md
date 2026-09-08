# Live Web Dashboard & Local Diagnostic Visualizer Server

`llm-reliability` includes a lightweight, built-in local HTTP server and live web visualizer for inspecting traces, diagnosing failures, and exploring execution spans in real-time.

## CLI Usage

Launch the visualizer directly from the command line:

```bash
# Start server on default port 8080 and open browser automatically
llm-reliability serve --open-browser

# Specify custom host, port, and directory of saved traces
llm-reliability serve --host 0.0.0.0 --port 9000 --trace-dir ./saved_traces
```

---

## Python API Usage

```python
from llm_reliability import DiagnosticServer, ServerConfig, start_server

# Programmatic initialization
config = ServerConfig(host="127.0.0.1", port=8080, trace_dir="./traces")
server = DiagnosticServer(config)
```
