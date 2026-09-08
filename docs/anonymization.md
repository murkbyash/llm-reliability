# Telemetry Anonymization & PII Scrubbing

The telemetry anonymization engine removes Personally Identifiable Information (PII) and secret credentials from LLM traces and reports before sharing, exporting, or publishing.

## Supported PII Types

- **Email addresses**: `alice@example.com` $\to$ `[REDACTED_EMAIL]`
- **Phone numbers**: `+1-555-123-4567` $\to$ `[REDACTED_PHONE]`
- **Credit card numbers**: `4532-XXXX-XXXX-XXXX` $\to$ `[REDACTED_CREDIT_CARD]`
- **Social Security Numbers (SSN)**: `XXX-XX-XXXX` $\to$ `[REDACTED_SSN]`
- **IP addresses**: `192.168.1.1` $\to$ `[REDACTED_IP_ADDRESS]`
- **API Keys & Bearer Tokens**: `sk-proj-...` $\to$ `[REDACTED_API_KEY]`
- **Passwords**: `password=...` $\to$ `[REDACTED_PASSWORD]`

---

## Python API Usage

### 1. Direct String Scrubbing

```python
from llm_reliability import PIIScrubber, RedactionConfig, scrub_pii

# Convenience function
clean_text = scrub_pii("User email is john@corp.com with key sk-proj-12345")
print(clean_text)
```

### 2. Full Trace Anonymization

```python
from llm_reliability import TelemetryAnonymizer, load_trace, anonymize_trace

trace = load_trace("sensitive_trace.json")
sanitized_trace = anonymize_trace(trace)
```
