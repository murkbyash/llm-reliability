# Troubleshooting & FAQ

Common questions, diagnostic debugging tips, and troubleshooting steps.

---

## 1. Frequently Asked Questions

### Q: Why does `llm-reliability` require ₹0 / $0 cloud API cost?
**A:** All parsing, statistical metric computation, failure classification, hypothesis ranking, token latency profiling, and HTML dashboard rendering are executed locally using deterministic algorithms, mathematical statistics, and lexical heuristics. No OpenAI/Anthropic/cloud API keys are required.

### Q: How does `llm-reliability` handle ungrounded responses?
**A:** `GroundingAnalyzer` breaks the generated answer into sentence-level claim units, calculates token overlap and lexical density against retrieved documents, checks for polar/numerical contradictions, and flags fabricated named entities that do not exist in the source corpus.

### Q: Can I run this in an air-gapped CI/CD environment?
**A:** Yes. The package has zero external network calls at runtime, and generated HTML reports embed all CSS and JS inline with zero CDN dependencies.

---

## 2. Common Errors & Resolutions

### `TraceParseError: Unable to parse JSON`
- **Cause:** Input file is not valid JSON or JSONL.
- **Fix:** Validate the file syntax with `python -m json.tool <file.json>`.

### `TraceValidationError: Expected trace data as dictionary or list of spans`
- **Cause:** Root element in the payload is a primitive type (e.g. integer or string).
- **Fix:** Ensure the root element is either a trace dictionary or a list of span objects.

### `UnboundLocalError / KeyError on custom span format`
- **Cause:** Trace uses non-standard span field names.
- **Fix:** Use `llm_reliability.normalize_trace(data)` or wrap your custom span parser using `Span(...)`.

