"""Automated unit tests for Multi-Modal and Structured Output Reliability Analyzers."""

import base64

from llm_reliability import (
    DiagnosticEngine,
    FailureCategory,
    LLMCall,
    MultiModalAnalyzer,
    MultiModalReliabilityAnalyzer,
    Run,
    Span,
    SpanKind,
    SpanStatus,
    StructuredOutputAnalyzer,
    Trace,
)


class TestStructuredOutputAnalyzer:
    """Test JSON syntax extraction, markdown code block stripping, and JSON schema validation."""

    def test_structured_output_valid_json_and_fenced_markdown(self) -> None:
        analyzer = StructuredOutputAnalyzer()

        # 1. Plain JSON string
        valid_json = '{"name": "Alice", "score": 95, "active": true}'
        is_valid, parsed, err = analyzer.validate_json(valid_json)
        assert is_valid is True
        assert parsed == {"name": "Alice", "score": 95, "active": True}
        assert err is None

        # 2. Markdown fenced JSON block
        fenced_json = """```json
{
  "status": "success",
  "items": [1, 2, 3]
}
```"""
        is_valid, parsed, err = analyzer.validate_json(fenced_json)
        assert is_valid is True
        assert parsed["status"] == "success"
        assert parsed["items"] == [1, 2, 3]

    def test_structured_output_malformed_json_failure(self) -> None:
        analyzer = StructuredOutputAnalyzer()
        bad_json = '{"name": "Alice", "score": 95, active: }'
        is_valid, parsed, err = analyzer.validate_json(bad_json)

        assert is_valid is False
        assert parsed is None
        assert "JSONDecodeError" in str(err)

        # Analyze LLM call with malformed output
        llm_call = LLMCall(
            model="gpt-4",
            prompt="Generate JSON user record",
            response=bad_json,
            raw_parameters={"json_mode": True},
        )
        res = analyzer.analyze(llm_call)
        assert res.structured_metrics is not None
        assert res.structured_metrics.is_valid_json is False
        assert len(res.failures) == 1
        assert res.failures[0].category == FailureCategory.OUTPUT_FORMAT_FAILURE

    def test_structured_output_schema_conformance(self) -> None:
        analyzer = StructuredOutputAnalyzer()
        schema = {
            "type": "object",
            "required": ["user_id", "email", "age", "roles"],
            "properties": {
                "user_id": {"type": "integer"},
                "email": {"type": "string"},
                "age": {"type": "integer"},
                "roles": {"type": "array", "items": {"type": "string"}},
            },
        }

        # 1. Conforming data
        valid_data = {
            "user_id": 101,
            "email": "dev@example.com",
            "age": 28,
            "roles": ["admin", "editor"],
        }
        metrics = analyzer.validate_schema(valid_data, schema)
        assert metrics.schema_matched is True
        assert metrics.schema_violation_count == 0

        # 2. Missing required field 'age'
        missing_data = {
            "user_id": 102,
            "email": "user@example.com",
            "roles": ["viewer"],
        }
        metrics_missing = analyzer.validate_schema(missing_data, schema)
        assert metrics_missing.schema_matched is False
        assert "age" in metrics_missing.missing_required_keys
        assert metrics_missing.schema_violation_count >= 1

        # 3. Type mismatch: age is string instead of integer
        mismatch_data = {
            "user_id": 103,
            "email": "user@example.com",
            "age": "twenty-five",
            "roles": ["viewer"],
        }
        metrics_mismatch = analyzer.validate_schema(mismatch_data, schema)
        assert metrics_mismatch.schema_matched is False
        assert any("age" in tm for tm in metrics_mismatch.type_mismatches)

    def test_nested_schema_and_array_validation(self) -> None:
        analyzer = StructuredOutputAnalyzer()
        schema = {
            "type": "object",
            "required": ["company", "employees"],
            "properties": {
                "company": {"type": "string"},
                "employees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }

        invalid_nested = {
            "company": "DeepMind",
            "employees": [
                {"id": 1, "name": "Ada"},
                {"id": "two", "name": "Alan"},  # type mismatch on id
            ],
        }
        metrics = analyzer.validate_schema(invalid_nested, schema)
        assert metrics.schema_matched is False
        assert any("employees[1].id" in tm for tm in metrics.type_mismatches)


class TestMultiModalAnalyzer:
    """Test multi-modal media attachment inspection, base64 verification, and visual grounding."""

    def test_multimodal_valid_image_url_and_base64(self) -> None:
        analyzer = MultiModalAnalyzer()
        valid_b64 = base64.b64encode(b"fake-png-binary-bytes").decode("ascii")
        span = Span(
            span_id="s1",
            name="vision_step",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            attributes={
                "images": [
                    f"data:image/png;base64,{valid_b64}",
                    "https://example.com/chart.png",
                ]
            },
            llm_call=LLMCall(
                model="gpt-4-vision",
                prompt="Examine Image 1 and Image 2",
                response="Image 1 shows an architecture diagram while Image 2 displays latency charts.",
            ),
        )

        res = analyzer.analyze_span(span)
        assert len(res.failures) == 0
        assert res.multimodal_metrics is not None
        assert res.multimodal_metrics.image_count == 2
        assert res.multimodal_metrics.has_unresolved_media_reference is False
        assert res.multimodal_metrics.visual_grounding_score == 1.0

    def test_multimodal_corrupt_base64_error(self) -> None:
        analyzer = MultiModalAnalyzer()
        corrupt_b64 = "data:image/png;base64,invalid-corrupted-base64-payload!#$@%"
        span = Span(
            span_id="s2",
            name="vision_step",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            attributes={"images": [corrupt_b64]},
            llm_call=LLMCall(
                model="gpt-4-vision", prompt="Describe image", response="A sunny landscape"
            ),
        )

        res = analyzer.analyze_span(span)
        assert len(res.failures) == 1
        assert res.failures[0].category == FailureCategory.SCHEMA_VIOLATION
        assert res.multimodal_metrics is not None
        assert len(res.multimodal_metrics.media_format_errors) == 1

    def test_multimodal_out_of_bounds_image_reference(self) -> None:
        analyzer = MultiModalAnalyzer()
        span = Span(
            span_id="s3",
            name="vision_step",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            attributes={"images": ["https://example.com/single_photo.jpg"]},
            llm_call=LLMCall(
                model="gpt-4-vision",
                prompt="Inspect provided photo",
                response="Comparing Image 1 and Image 3, Image 3 displays higher variance.",
            ),
        )

        res = analyzer.analyze_span(span)
        assert len(res.failures) == 1
        assert res.failures[0].category == FailureCategory.HALLUCINATION
        assert res.multimodal_metrics is not None
        assert res.multimodal_metrics.has_unresolved_media_reference is True
        assert any("Image 3" in ref for ref in res.multimodal_metrics.unresolved_references)

    def test_diagnostic_engine_integration_with_multimodal_and_structured(self) -> None:
        engine = DiagnosticEngine()

        # 1. Structured output failure diagnosis
        schema = {
            "type": "object",
            "required": ["result_code"],
            "properties": {"result_code": {"type": "integer"}},
        }
        span_bad_schema = Span(
            span_id="s-schema",
            name="llm_step",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            llm_call=LLMCall(
                model="gpt-4",
                prompt="Return JSON result",
                response='{"error": "Missing result_code"}',
                raw_parameters={"json_schema": schema},
            ),
        )
        trace_schema = Trace(
            trace_id="t-schema",
            runs=[Run(run_id="r1", trace_id="t-schema", spans=[span_bad_schema])],
        )
        diag = engine.diagnose(trace_schema)
        assert any(f.category == FailureCategory.SCHEMA_VIOLATION for f in diag.failures)

        # 2. Multi-modal failure diagnosis via MultiModalReliabilityAnalyzer
        mm_engine = MultiModalReliabilityAnalyzer()
        mm_res = mm_engine.analyze_trace(trace_schema)
        assert mm_res.structured_metrics is not None
        assert mm_res.structured_metrics.schema_matched is False
