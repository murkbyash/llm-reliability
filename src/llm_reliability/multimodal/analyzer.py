"""Deterministic analyzers for structured outputs (JSON schema conformance) and multi-modal traces."""

import base64
import json
import re
from typing import Any

from llm_reliability.models.diagnosis import Evidence, Failure, Metric
from llm_reliability.models.enums import EvidenceType, FailureCategory, Severity
from llm_reliability.models.execution import LLMCall
from llm_reliability.models.trace import Run, Span, Trace
from llm_reliability.multimodal.models import (
    MultiModalAnalysisResult,
    MultiModalMetrics,
    StructuredOutputMetrics,
)


class StructuredOutputAnalyzer:
    """Evaluates LLM structured generation against JSON syntax rules and schema constraints."""

    @staticmethod
    def extract_json_string(text: str) -> str:
        """Strip markdown fences (```json ... ```) and extract candidate JSON string."""
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                end_idx = len(lines) - 1
                while end_idx > 0 and not lines[end_idx].strip().startswith("```"):
                    end_idx -= 1
                if end_idx > 0:
                    clean = "\n".join(lines[1:end_idx]).strip()
        return clean

    def validate_json(self, raw_text: str) -> tuple[bool, Any, str | None]:
        """Attempt to extract and parse JSON from raw text response."""
        if not raw_text or not raw_text.strip():
            return False, None, "Empty text output provided"

        extracted = self.extract_json_string(raw_text)
        try:
            parsed = json.loads(extracted)
            return True, parsed, None
        except json.JSONDecodeError as e:
            return False, None, f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno}"

    def validate_schema(
        self, data: Any, expected_schema: dict[str, Any]
    ) -> StructuredOutputMetrics:
        """Validate parsed Python data structure against expected JSON Schema specification."""
        missing_keys: list[str] = []
        type_mismatches: list[str] = []
        unexpected_keys: list[str] = []

        self._check_node(
            data=data,
            schema=expected_schema,
            path="",
            missing_keys=missing_keys,
            type_mismatches=type_mismatches,
            unexpected_keys=unexpected_keys,
        )

        total_violations = len(missing_keys) + len(type_mismatches) + len(unexpected_keys)
        return StructuredOutputMetrics(
            is_valid_json=True,
            schema_matched=total_violations == 0,
            missing_required_keys=missing_keys,
            type_mismatches=type_mismatches,
            unexpected_keys=unexpected_keys,
            schema_violation_count=total_violations,
        )

    def _check_node(
        self,
        data: Any,
        schema: dict[str, Any],
        path: str,
        missing_keys: list[str],
        type_mismatches: list[str],
        unexpected_keys: list[str],
    ) -> None:
        """Recursive node validator against JSON Schema properties."""
        expected_type = schema.get("type")

        # Type checks
        if expected_type:
            types_map: dict[str, tuple[type, ...] | type] = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }
            if expected_type in types_map:
                target_py_type = types_map[expected_type]
                # Special check for bool vs int in Python (bool is subclass of int)
                if expected_type in ("integer", "number") and isinstance(data, bool):
                    type_mismatches.append(
                        f"{path or 'root'}: expected {expected_type}, got boolean"
                    )
                    return
                if not isinstance(data, target_py_type):
                    type_mismatches.append(
                        f"{path or 'root'}: expected {expected_type}, got {type(data).__name__}"
                    )
                    return

        # Object properties & required fields
        if isinstance(data, dict):
            required = schema.get("required", [])
            for req_key in required:
                if req_key not in data:
                    full_p = f"{path}.{req_key}" if path else req_key
                    missing_keys.append(full_p)

            properties = schema.get("properties", {})
            additional_allowed = schema.get("additionalProperties", True)

            for k, v in data.items():
                full_p = f"{path}.{k}" if path else k
                if k in properties:
                    self._check_node(
                        data=v,
                        schema=properties[k],
                        path=full_p,
                        missing_keys=missing_keys,
                        type_mismatches=type_mismatches,
                        unexpected_keys=unexpected_keys,
                    )
                elif not additional_allowed:
                    unexpected_keys.append(full_p)

        # Array items
        elif isinstance(data, list) and "items" in schema:
            item_schema = schema["items"]
            for idx, item in enumerate(data):
                self._check_node(
                    data=item,
                    schema=item_schema,
                    path=f"{path}[{idx}]",
                    missing_keys=missing_keys,
                    type_mismatches=type_mismatches,
                    unexpected_keys=unexpected_keys,
                )

    def analyze(
        self,
        llm_call: LLMCall,
        expected_schema: dict[str, Any] | None = None,
    ) -> MultiModalAnalysisResult:
        """Evaluate LLM response for JSON validity and schema conformance."""
        failures: list[Failure] = []
        metrics: list[Metric] = []
        evidence: list[Evidence] = []

        schema_to_use = expected_schema
        params = llm_call.raw_parameters if llm_call.raw_parameters else {}
        if schema_to_use is None and params:
            schema_to_use = (
                params.get("json_schema") or params.get("schema") or params.get("response_schema")
            )

        requires_json = schema_to_use is not None or (
            bool(params)
            and (params.get("json_mode") is True or params.get("response_format") == "json_object")
        )

        if not requires_json:
            return MultiModalAnalysisResult()

        response_text = llm_call.response or ""
        is_valid, parsed_data, parse_err = self.validate_json(response_text)

        if not is_valid:
            struct_metrics = StructuredOutputMetrics(
                is_valid_json=False,
                schema_matched=False,
                json_parse_error=parse_err,
                schema_violation_count=1,
            )
            metrics.append(
                Metric(
                    name="is_valid_json",
                    value=0.0,
                    unit="bool",
                    details={"description": "Whether LLM generated valid JSON"},
                )
            )
            failures.append(
                Failure(
                    category=FailureCategory.OUTPUT_FORMAT_FAILURE,
                    severity=Severity.HIGH,
                    title="Malformed JSON Output",
                    description=f"LLM produced malformed non-parseable JSON: {parse_err}",
                )
            )
            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.SCHEMA_VIOLATION,
                    description="Malformed JSON output encountered",
                    supporting_data={
                        "parse_error": parse_err,
                        "raw_response_snippet": response_text[:200],
                    },
                )
            )
            return MultiModalAnalysisResult(
                failures=failures,
                metrics=metrics,
                evidence=evidence,
                structured_metrics=struct_metrics,
            )

        # JSON is syntactically valid
        metrics.append(
            Metric(
                name="is_valid_json",
                value=1.0,
                unit="bool",
                details={"description": "Whether LLM generated valid JSON"},
            )
        )

        if schema_to_use:
            struct_metrics = self.validate_schema(parsed_data, schema_to_use)
            metrics.append(
                Metric(
                    name="schema_violation_count",
                    value=float(struct_metrics.schema_violation_count),
                    unit="count",
                    details={"description": "Count of JSON Schema constraint violations"},
                )
            )

            if not struct_metrics.schema_matched:
                details = {
                    "missing_required_keys": struct_metrics.missing_required_keys,
                    "type_mismatches": struct_metrics.type_mismatches,
                    "unexpected_keys": struct_metrics.unexpected_keys,
                }
                desc_parts = []
                if struct_metrics.missing_required_keys:
                    desc_parts.append(
                        f"missing required keys: {struct_metrics.missing_required_keys}"
                    )
                if struct_metrics.type_mismatches:
                    desc_parts.append(f"type mismatches: {struct_metrics.type_mismatches}")
                if struct_metrics.unexpected_keys:
                    desc_parts.append(f"unexpected keys: {struct_metrics.unexpected_keys}")

                failures.append(
                    Failure(
                        category=FailureCategory.SCHEMA_VIOLATION,
                        severity=Severity.HIGH
                        if struct_metrics.missing_required_keys
                        else Severity.MEDIUM,
                        title="Structured Output Schema Violation",
                        description=f"Structured output schema violation: {'; '.join(desc_parts)}",
                    )
                )
                evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.SCHEMA_VIOLATION,
                        description="Schema validation failure details",
                        supporting_data=details,
                    )
                )
            return MultiModalAnalysisResult(
                failures=failures,
                metrics=metrics,
                evidence=evidence,
                structured_metrics=struct_metrics,
            )

        return MultiModalAnalysisResult(
            failures=failures,
            metrics=metrics,
            evidence=evidence,
            structured_metrics=StructuredOutputMetrics(is_valid_json=True, schema_matched=True),
        )


class MultiModalAnalyzer:
    """Evaluates Vision and Multi-Modal execution traces for media integrity and visual grounding."""

    def analyze_span(self, span: Span) -> MultiModalAnalysisResult:
        """Inspect multi-modal attachments, base64 encodings, and visual reference alignment."""
        failures: list[Failure] = []
        metrics: list[Metric] = []
        evidence: list[Evidence] = []

        # Extract media attachments from span attributes or raw_parameters
        media_list: list[Any] = []
        if span.attributes:
            media_list.extend(span.attributes.get("images", []))
            media_list.extend(span.attributes.get("media", []))
            media_list.extend(span.attributes.get("attachments", []))
        if span.llm_call and span.llm_call.raw_parameters:
            media_list.extend(span.llm_call.raw_parameters.get("images", []))
            media_list.extend(span.llm_call.raw_parameters.get("media", []))

        # Check if span is multi-modal
        is_multimodal = len(media_list) > 0 or (
            span.attributes is not None
            and (
                span.attributes.get("multimodal") is True
                or span.attributes.get("is_vision") is True
            )
        )
        if not is_multimodal:
            return MultiModalAnalysisResult()

        image_count = len(media_list)
        format_errors: list[str] = []
        unresolved_refs: list[str] = []

        # 1. Inspect media payloads (base64 and URLs)
        for idx, item in enumerate(media_list):
            if isinstance(item, str):
                if item.startswith("data:image/"):
                    # Check base64 format
                    parts = item.split(";base64,")
                    if len(parts) == 2:
                        b64_str = parts[1]
                        try:
                            base64.b64decode(b64_str, validate=True)
                        except Exception as e:
                            format_errors.append(f"Image {idx} has corrupted base64 data: {e}")
                    else:
                        format_errors.append(f"Image {idx} has invalid data URI structure")
                elif item.startswith(("http://", "https://", "file://")):
                    if len(item) < 8 or " " in item:
                        unresolved_refs.append(f"Image {idx} has malformed URI: {item}")
            elif isinstance(item, dict):
                if item.get("type") == "image_url":
                    url_obj = item.get("image_url", {})
                    url_str = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                    if not url_str or " " in url_str:
                        unresolved_refs.append(f"Image {idx} has missing or malformed image URL")

        # 2. Check visual index hallucinations in prompt / response
        text_to_scan = ""
        if span.llm_call:
            text_to_scan = f"{span.llm_call.prompt or ''} {span.llm_call.response or ''}"

        # Match patterns like "Image 3", "Figure 2", "Photo 4"
        ref_matches = re.findall(
            r"\b(?:Image|Figure|Photo|Picture)\s+#?(\d+)\b", text_to_scan, re.IGNORECASE
        )
        for match in ref_matches:
            ref_idx = int(match)
            # 1-indexed reference vs actual media count
            if ref_idx > image_count and image_count > 0:
                unresolved_refs.append(
                    f"Out-of-bounds visual reference: referenced Image {ref_idx}, but only {image_count} images were provided"
                )

        if format_errors:
            failures.append(
                Failure(
                    category=FailureCategory.SCHEMA_VIOLATION,
                    severity=Severity.HIGH,
                    title="Multi-Modal Media Format Error",
                    description=f"Multi-modal media format error: {'; '.join(format_errors)}",
                )
            )

            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.SCHEMA_VIOLATION,
                    description="Media formatting validation failure",
                    supporting_data={"format_errors": format_errors},
                )
            )

        if unresolved_refs:
            failures.append(
                Failure(
                    category=FailureCategory.HALLUCINATION,
                    severity=Severity.HIGH,
                    title="Unresolved Visual Reference",
                    description=f"Unresolved or hallucinated media reference: {'; '.join(unresolved_refs)}",
                )
            )
            evidence.append(
                Evidence(
                    evidence_type=EvidenceType.GROUNDING_DEFICIT,
                    description="Out-of-bounds or broken visual references detected",
                    supporting_data={"unresolved_references": unresolved_refs},
                )
            )

        has_unresolved = len(unresolved_refs) > 0 or len(format_errors) > 0
        grounding_score = 0.0 if has_unresolved else 1.0

        mm_metrics = MultiModalMetrics(
            image_count=image_count,
            has_unresolved_media_reference=has_unresolved,
            unresolved_references=unresolved_refs,
            visual_grounding_score=grounding_score,
            media_format_errors=format_errors,
        )

        metrics.append(
            Metric(
                name="visual_grounding_score",
                value=grounding_score,
                unit="score",
                details={"description": "Visual reference integrity and grounding score"},
            )
        )

        return MultiModalAnalysisResult(
            failures=failures,
            metrics=metrics,
            evidence=evidence,
            multimodal_metrics=mm_metrics,
        )


class MultiModalReliabilityAnalyzer:
    """Unified analyzer combining structured output validation and multimodal trace analysis."""

    def __init__(self) -> None:
        self.structured_analyzer = StructuredOutputAnalyzer()
        self.multimodal_analyzer = MultiModalAnalyzer()

    def analyze_span(
        self, span: Span, expected_schema: dict[str, Any] | None = None
    ) -> MultiModalAnalysisResult:
        """Run structured output and multimodal verification on a single span."""
        failures: list[Failure] = []
        metrics: list[Metric] = []
        evidence: list[Evidence] = []

        struct_res: MultiModalAnalysisResult | None = None
        if span.llm_call:
            struct_res = self.structured_analyzer.analyze(
                span.llm_call, expected_schema=expected_schema
            )
            failures.extend(struct_res.failures)
            metrics.extend(struct_res.metrics)
            evidence.extend(struct_res.evidence)

        mm_res = self.multimodal_analyzer.analyze_span(span)
        failures.extend(mm_res.failures)
        metrics.extend(mm_res.metrics)
        evidence.extend(mm_res.evidence)

        return MultiModalAnalysisResult(
            failures=failures,
            metrics=metrics,
            evidence=evidence,
            structured_metrics=struct_res.structured_metrics if struct_res else None,
            multimodal_metrics=mm_res.multimodal_metrics,
        )

    def analyze_run(
        self, run: Run, expected_schema: dict[str, Any] | None = None
    ) -> MultiModalAnalysisResult:
        """Run structured output and multimodal verification across all spans in a run."""
        failures: list[Failure] = []
        metrics: list[Metric] = []
        evidence: list[Evidence] = []

        last_struct: StructuredOutputMetrics | None = None
        last_mm: MultiModalMetrics | None = None

        for span in run.spans:
            res = self.analyze_span(span, expected_schema=expected_schema)
            failures.extend(res.failures)
            metrics.extend(res.metrics)
            evidence.extend(res.evidence)
            if res.structured_metrics:
                last_struct = res.structured_metrics
            if res.multimodal_metrics:
                last_mm = res.multimodal_metrics

        return MultiModalAnalysisResult(
            failures=failures,
            metrics=metrics,
            evidence=evidence,
            structured_metrics=last_struct,
            multimodal_metrics=last_mm,
        )

    def analyze_trace(
        self, trace: Trace, expected_schema: dict[str, Any] | None = None
    ) -> MultiModalAnalysisResult:
        """Run structured output and multimodal verification across all runs in a trace."""
        failures: list[Failure] = []
        metrics: list[Metric] = []
        evidence: list[Evidence] = []

        last_struct: StructuredOutputMetrics | None = None
        last_mm: MultiModalMetrics | None = None

        for run in trace.runs:
            res = self.analyze_run(run, expected_schema=expected_schema)
            failures.extend(res.failures)
            metrics.extend(res.metrics)
            evidence.extend(res.evidence)
            if res.structured_metrics:
                last_struct = res.structured_metrics
            if res.multimodal_metrics:
                last_mm = res.multimodal_metrics

        return MultiModalAnalysisResult(
            failures=failures,
            metrics=metrics,
            evidence=evidence,
            structured_metrics=last_struct,
            multimodal_metrics=last_mm,
        )
