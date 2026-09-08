# Multi-Modal & Structured Output Reliability

The `llm-reliability` multi-modal analysis engine diagnoses structured output schema violations (JSON mode, function calling schemas, Pydantic models) and multi-modal/vision trace failures with zero paid API dependencies.

## Key Capabilities

1. **Structured Output & JSON Schema Verification**:
   - Strips markdown code blocks (```json ... ```) automatically.
   - Validates JSON grammar syntax errors.
   - Recursively verifies data structures against JSON Schema / Pydantic schema specifications.
   - Diagnoses missing required fields, type mismatches, and forbidden extra properties.
2. **Multi-Modal & Vision Attachment Inspection**:
   - Inspects input image and media attachments in span attributes or LLM call parameters.
   - Validates base64 encoding (`data:image/...;base64,...`) and URL formatting.
   - Detects visual index hallucinations (e.g. referencing "Figure 3" when only 1 image was provided).
   - Computes `visual_grounding_score`.

---

## Python API Usage

```python
from llm_reliability import StructuredOutputAnalyzer, MultiModalAnalyzer, LLMCall

# 1. Structured Output Analysis
analyzer = StructuredOutputAnalyzer()
schema = {"type": "object", "required": ["id", "val"], "properties": {"id": {"type": "string"}}}
llm_call = LLMCall(
    model="gpt-4", prompt="test", response='{"id": 123}', raw_parameters={"json_schema": schema}
)
result = analyzer.analyze(llm_call)

# 2. Multi-Modal Vision Analysis
mm_analyzer = MultiModalAnalyzer()
```
