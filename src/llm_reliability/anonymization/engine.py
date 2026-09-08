"""Deterministic PII scrubbing, credential redaction, and trace telemetry anonymization engine."""

import hashlib
import re
from typing import Any

from llm_reliability.anonymization.models import (
    PIIType,
    RedactedItem,
    RedactionConfig,
    RedactionMaskType,
    RedactionResult,
)
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.trace import Run, Span, Trace

# Standard high-accuracy deterministic regex patterns ordered by specificity
ORDERED_PII_TYPES = [
    PIIType.API_KEY,
    PIIType.BEARER_TOKEN,
    PIIType.PASSWORD,
    PIIType.CREDIT_CARD,
    PIIType.SSN,
    PIIType.EMAIL,
    PIIType.IP_ADDRESS,
    PIIType.PHONE_NUMBER,
]

DEFAULT_PATTERNS: dict[PIIType, list[str]] = {
    PIIType.API_KEY: [
        r"\bsk-[a-zA-Z0-9T3BlbkFJ]{20,}\b",
        r"\bgh[pousr]_[a-zA-Z0-9]{36,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
    ],
    PIIType.BEARER_TOKEN: [
        r"(?i)\bBearer\s+([a-zA-Z0-9\-_.~+/]+=*)",
    ],
    PIIType.PASSWORD: [
        r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^'\"\s\r\n,}]+)['\"]?",
    ],
    PIIType.CREDIT_CARD: [
        r"\b(?:\d{4}[-\s]){3}\d{4}\b",
        r"\b\d{16}\b",
    ],
    PIIType.SSN: [
        r"\b\d{3}-\d{2}-\d{4}\b",
    ],
    PIIType.EMAIL: [
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    ],
    PIIType.IP_ADDRESS: [
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    ],
    PIIType.PHONE_NUMBER: [
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        r"\b\+\d{1,3}[-.\s]\d{3,4}[-.\s]\d{4}\b",
    ],
}


class PIIScrubber:
    """High-performance deterministic regex scrubber for detecting and masking sensitive entities."""

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config = config or RedactionConfig()
        self._compiled_patterns: list[
            tuple[PIIType, re.Pattern[str], RedactionMaskType, str | None]
        ] = []
        self._compile_rules()

    def _compile_rules(self) -> None:
        """Compile active builtin patterns and custom user rules in priority order."""
        # Custom rules first
        for rule in self.config.custom_rules:
            try:
                compiled = re.compile(rule.pattern)
                self._compiled_patterns.append(
                    (rule.pii_type, compiled, rule.mask_type, rule.replacement)
                )
            except re.error:
                continue

        # Builtin rules for enabled types in priority order
        for pii_type in ORDERED_PII_TYPES:
            if pii_type in self.config.enabled_types:
                patterns = DEFAULT_PATTERNS.get(pii_type, [])
                for pattern_str in patterns:
                    try:
                        compiled = re.compile(pattern_str)
                        self._compiled_patterns.append(
                            (pii_type, compiled, self.config.default_mask_type, None)
                        )
                    except re.error:
                        continue

    def _generate_replacement(
        self,
        match_text: str,
        pii_type: PIIType,
        mask_type: RedactionMaskType,
        custom_replacement: str | None = None,
    ) -> str:
        """Generate formatted masked string according to strategy."""
        if custom_replacement is not None:
            return custom_replacement

        if mask_type == RedactionMaskType.REDACTED:
            return "[REDACTED]"
        elif mask_type == RedactionMaskType.HASH:
            salted = f"{self.config.hash_salt}:{match_text}".encode()
            digest = hashlib.sha256(salted).hexdigest()[:8]
            return f"[HASH:{digest}]"
        elif mask_type == RedactionMaskType.PARTIAL_MASK:
            if pii_type == PIIType.EMAIL and "@" in match_text:
                local_part, domain = match_text.split("@", 1)
                masked_local = local_part[0] + "***" if len(local_part) > 1 else "***"
                domain_parts = domain.split(".", 1)
                if len(domain_parts) == 2:
                    masked_domain = domain_parts[0][0] + "***." + domain_parts[1]
                else:
                    masked_domain = domain[0] + "***"
                return f"{masked_local}@{masked_domain}"

            if pii_type == PIIType.API_KEY and match_text.startswith("sk-"):
                return f"sk-...{match_text[-4:]}"

            if len(match_text) <= 4:
                return "****"
            return f"{match_text[:2]}...{match_text[-2:]}"
        else:
            return f"[{pii_type.value}]"

    def scrub_text(self, text: str) -> RedactionResult:
        """Scan text and redact all detected sensitive entities."""
        if not text:
            return RedactionResult(sanitized_text=text)

        sanitized = text
        redacted_items: list[RedactedItem] = []
        counts_by_type: dict[str, int] = {}

        # Scan for each pattern
        for pii_type, pattern, mask_type, custom_rep in self._compiled_patterns:
            matches = list(pattern.finditer(sanitized))
            # Reverse order to preserve indices during string replacements
            for match in reversed(matches):
                original_snippet = match.group(0)
                start, end = match.start(), match.end()
                replacement = self._generate_replacement(
                    original_snippet, pii_type, mask_type, custom_rep
                )

                sanitized = sanitized[:start] + replacement + sanitized[end:]

                redacted_items.append(
                    RedactedItem(
                        pii_type=pii_type,
                        original_snippet=original_snippet,
                        redacted_snippet=replacement,
                        start_char=start,
                        end_char=end,
                    )
                )
                counts_by_type[pii_type.value] = counts_by_type.get(pii_type.value, 0) + 1

        return RedactionResult(
            sanitized_text=sanitized,
            total_redacted_count=len(redacted_items),
            counts_by_type=counts_by_type,
            redacted_items=redacted_items,
        )

    def scrub_value(self, val: Any) -> Any:
        """Recursively scrub strings, lists, and dicts."""
        if isinstance(val, str):
            return self.scrub_text(val).sanitized_text
        elif isinstance(val, dict):
            new_dict: dict[str, Any] = {}
            for k, v in val.items():
                if isinstance(k, str) and any(
                    key_mask.lower() in k.lower() for key_mask in self.config.mask_attribute_keys
                ):
                    new_dict[k] = "[REDACTED_SECRET]"
                else:
                    new_dict[k] = self.scrub_value(v)
            return new_dict
        elif isinstance(val, list):
            return [self.scrub_value(item) for item in val]
        return val


class TelemetryAnonymizer:
    """Anonymizes execution traces, spans, and diagnostic reports."""

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.scrubber = PIIScrubber(config)

    def anonymize_span(self, span: Span) -> Span:
        """Return a copy of the span with all text fields, attributes, and payloads sanitized."""
        span_dict = span.model_dump()

        # Scrub input / output dictionaries
        if span_dict.get("input") is not None:
            span_dict["input"] = self.scrubber.scrub_value(span_dict["input"])
        if span_dict.get("output") is not None:
            span_dict["output"] = self.scrubber.scrub_value(span_dict["output"])
        if span_dict.get("attributes"):
            span_dict["attributes"] = self.scrubber.scrub_value(span_dict["attributes"])
        if span_dict.get("error_message"):
            span_dict["error_message"] = self.scrubber.scrub_text(
                span_dict["error_message"]
            ).sanitized_text

        # Scrub LLM Call details
        if span_dict.get("llm_call"):
            llm = span_dict["llm_call"]
            if llm.get("prompt"):
                llm["prompt"] = self.scrubber.scrub_text(llm["prompt"]).sanitized_text
            if llm.get("response"):
                llm["response"] = self.scrubber.scrub_text(llm["response"]).sanitized_text
            if llm.get("system_prompt"):
                llm["system_prompt"] = self.scrubber.scrub_text(llm["system_prompt"]).sanitized_text

        # Scrub Tool Call / Result details
        if span_dict.get("tool_call"):
            tc = span_dict["tool_call"]
            if tc.get("arguments"):
                tc["arguments"] = self.scrubber.scrub_value(tc["arguments"])
        if span_dict.get("tool_result"):
            tr = span_dict["tool_result"]
            if tr.get("output"):
                tr["output"] = self.scrubber.scrub_value(tr["output"])
            if tr.get("error"):
                tr["error"] = self.scrubber.scrub_text(tr["error"]).sanitized_text

        # Scrub Retrieval Step details
        if span_dict.get("retrieval"):
            ret = span_dict["retrieval"]
            if ret.get("query"):
                ret["query"] = self.scrubber.scrub_text(ret["query"]).sanitized_text
            if ret.get("documents"):
                for doc in ret["documents"]:
                    if doc.get("content"):
                        doc["content"] = self.scrubber.scrub_text(doc["content"]).sanitized_text
                    if doc.get("metadata"):
                        doc["metadata"] = self.scrubber.scrub_value(doc["metadata"])

        # Recursively scrub child spans
        if span_dict.get("children"):
            child_spans: list[dict[str, Any]] = []
            for child in span_dict["children"]:
                child_span = Span.model_validate(child)
                anonymized_child = self.anonymize_span(child_span)
                child_spans.append(anonymized_child.model_dump())
            span_dict["children"] = child_spans

        return Span.model_validate(span_dict)

    def anonymize_run(self, run: Run) -> Run:
        """Return a copy of the run with sanitized spans."""
        anonymized_spans = [self.anonymize_span(s) for s in run.spans]
        run_dict = run.model_dump()
        run_dict["spans"] = [s.model_dump() for s in anonymized_spans]
        if run_dict.get("metadata"):
            run_dict["metadata"] = self.scrubber.scrub_value(run_dict["metadata"])
        return Run.model_validate(run_dict)

    def anonymize_trace(self, trace: Trace) -> Trace:
        """Return an anonymized copy of the full trace."""
        anonymized_runs = [self.anonymize_run(r) for r in trace.runs]
        trace_dict = trace.model_dump()
        trace_dict["runs"] = [r.model_dump() for r in anonymized_runs]
        if trace_dict.get("metadata"):
            trace_dict["metadata"] = self.scrubber.scrub_value(trace_dict["metadata"])
        return Trace.model_validate(trace_dict)

    def anonymize_diagnosis(self, diagnosis: Diagnosis) -> Diagnosis:
        """Return an anonymized copy of a diagnosis report."""
        diag_dict = diagnosis.model_dump()
        if diag_dict.get("summary"):
            diag_dict["summary"] = self.scrubber.scrub_text(diag_dict["summary"]).sanitized_text

        # Hypotheses
        for hyp in diag_dict.get("hypotheses", []):
            if hyp.get("description"):
                hyp["description"] = self.scrubber.scrub_text(hyp["description"]).sanitized_text
            if hyp.get("title"):
                hyp["title"] = self.scrubber.scrub_text(hyp["title"]).sanitized_text

        # Evidence
        for ev in diag_dict.get("evidence", []):
            if ev.get("description"):
                ev["description"] = self.scrubber.scrub_text(ev["description"]).sanitized_text
            if ev.get("details"):
                ev["details"] = self.scrubber.scrub_value(ev["details"])
            if ev.get("supporting_data"):
                ev["supporting_data"] = self.scrubber.scrub_value(ev["supporting_data"])

        # Recommendations
        for rec in diag_dict.get("recommendations", []):
            if rec.get("title"):
                rec["title"] = self.scrubber.scrub_text(rec["title"]).sanitized_text
            if rec.get("description"):
                rec["description"] = self.scrubber.scrub_text(rec["description"]).sanitized_text

        return Diagnosis.model_validate(diag_dict)


def scrub_pii(text: str, config: RedactionConfig | None = None) -> str:
    """Convenience helper to redact PII from string text."""
    scrubber = PIIScrubber(config)
    return scrubber.scrub_text(text).sanitized_text


def anonymize_trace(trace: Trace, config: RedactionConfig | None = None) -> Trace:
    """Convenience helper to anonymize full Trace object."""
    anonymizer = TelemetryAnonymizer(config)
    return anonymizer.anonymize_trace(trace)
