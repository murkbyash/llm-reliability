# Interactive HTML Diagnostic Reports

LLM Reliability Analyzer generates self-contained, standalone single-file HTML reports designed for offline inspection, CI/CD artifact storage, and engineering reviews.

---

## 1. Key Features

- **100% Offline (Zero External CDNs):** Embedded CSS3 and vanilla JavaScript guarantee that reports render reliably in air-gapped environments.
- **Execution Waterfall Timeline:** Color-coded execution duration bars proportional to trace lifetime (`LLM`, `RETRIEVAL`, `TOOL`, `CHAIN`, `AGENT`, `CUSTOM`).
- **Click-to-Inspect Span Modal:** Inspect prompts, responses, tool arguments, database outputs, and stack traces.
- **Root Cause & Confidence Gauges:** Visual confidence progress bars and evidence sub-tables.
- **Actionable Remediation Cards:** Prioritized fix advice with copy-to-clipboard code snippets.
- **Dark / Light Responsive Theme:** Built-in instant theme switcher.

---

## 2. Generating Reports via CLI

```bash
llm-reliability diagnose trace.json --format html --output report.html
```

---

## 3. Generating Reports via Python SDK

```python
from llm_reliability import diagnose, load_trace, render_html_report, save_html_report

trace = load_trace("trace.json")
diagnosis = diagnose(trace)

# 1. Render as an HTML string
html_content = render_html_report(diagnosis, trace=trace, title="Production Run Report")

# 2. Save directly to disk
save_html_report(diagnosis, "reports/production_diagnosis.html", trace=trace)
```

