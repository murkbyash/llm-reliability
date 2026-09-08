"""Automated unit tests for telemetry anonymization, PII scrubbing, and secret redaction engine."""

from datetime import datetime, timezone

from llm_reliability.anonymization import (
    PIIScrubber,
    PIIType,
    RedactionConfig,
    RedactionMaskType,
    RedactionRule,
    TelemetryAnonymizer,
    anonymize_trace,
    scrub_pii,
)
from llm_reliability.models import (
    Diagnosis,
    Evidence,
    EvidenceType,
    Failure,
    FailureCategory,
    Hypothesis,
    LLMCall,
    Recommendation,
    RetrievalStep,
    RetrievedDocument,
    Run,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    ToolCall,
    ToolResult,
    Trace,
)


class TestPIIScrubber:
    """Test deterministic regex entity detection and text masking."""

    def test_scrub_email(self) -> None:
        scrubber = PIIScrubber()
        result = scrubber.scrub_text("Please contact john.doe@example.com for info.")
        assert "[EMAIL]" in result.sanitized_text
        assert "john.doe@example.com" not in result.sanitized_text
        assert result.counts_by_type.get("EMAIL") == 1

    def test_scrub_phone_number(self) -> None:
        scrubber = PIIScrubber()
        result = scrubber.scrub_text("Call us at 555-123-4567 or +1-800-555-0199 now.")
        assert "[PHONE_NUMBER]" in result.sanitized_text
        assert "555-123-4567" not in result.sanitized_text

    def test_scrub_ip_address(self) -> None:
        scrubber = PIIScrubber()
        result = scrubber.scrub_text("Server connected from 192.168.1.100 on port 80.")
        assert "[IP_ADDRESS]" in result.sanitized_text
        assert "192.168.1.100" not in result.sanitized_text

    def test_scrub_credit_card_and_ssn(self) -> None:
        scrubber = PIIScrubber()
        result = scrubber.scrub_text("Card: 4111 2222 3333 4444, SSN: 123-45-6789.")
        assert "[CREDIT_CARD]" in result.sanitized_text
        assert "[SSN]" in result.sanitized_text
        assert "4111 2222 3333 4444" not in result.sanitized_text
        assert "123-45-6789" not in result.sanitized_text

    def test_scrub_api_keys_and_tokens(self) -> None:
        scrubber = PIIScrubber()
        result = scrubber.scrub_text(
            "OpenAI key sk-1234567890abcdef1234567890 and Bearer eyJhbGciOiJIUzI1NiJ9.test"
        )
        assert "[API_KEY]" in result.sanitized_text
        assert "[BEARER_TOKEN]" in result.sanitized_text
        assert "sk-1234567890abcdef1234567890" not in result.sanitized_text

    def test_mask_strategies(self) -> None:
        # Partial mask
        cfg_partial = RedactionConfig(default_mask_type=RedactionMaskType.PARTIAL_MASK)
        scrubber_partial = PIIScrubber(cfg_partial)
        res_partial = scrubber_partial.scrub_text(
            "Email: alice@corp.com, Key: sk-abcdef12345678901234"
        )
        assert "a***@c***.com" in res_partial.sanitized_text
        assert "sk-...1234" in res_partial.sanitized_text

        # Hash mask
        cfg_hash = RedactionConfig(default_mask_type=RedactionMaskType.HASH, hash_salt="test-salt")
        scrubber_hash = PIIScrubber(cfg_hash)
        res_hash = scrubber_hash.scrub_text("Email: bob@example.com")
        assert "[HASH:" in res_hash.sanitized_text

        # Redacted mask
        cfg_redacted = RedactionConfig(default_mask_type=RedactionMaskType.REDACTED)
        scrubber_redacted = PIIScrubber(cfg_redacted)
        res_redacted = scrubber_redacted.scrub_text("Email: bob@example.com")
        assert "[REDACTED]" in res_redacted.sanitized_text

    def test_custom_rule(self) -> None:
        custom_rule = RedactionRule(
            name="Internal Employee ID",
            pii_type=PIIType.CUSTOM,
            pattern=r"\bEMP-\d{5}\b",
            mask_type=RedactionMaskType.REPLACE_LABEL,
            replacement="[EMPLOYEE_ID]",
        )
        config = RedactionConfig(custom_rules=[custom_rule])
        scrubber = PIIScrubber(config)
        result = scrubber.scrub_text("Assigned to EMP-98765 for review.")
        assert "[EMPLOYEE_ID]" in result.sanitized_text
        assert "EMP-98765" not in result.sanitized_text

    def test_convenience_scrub_pii(self) -> None:
        sanitized = scrub_pii("Contact test@example.com or admin@domain.org")
        assert sanitized == "Contact [EMAIL] or [EMAIL]"


class TestTelemetryAnonymizer:
    """Test full trace, span, run, and diagnostic payload sanitization."""

    def test_anonymize_trace_and_spans(self) -> None:
        now = datetime.now(timezone.utc)
        span1 = Span(
            span_id="span-1",
            name="rag_query",
            kind=SpanKind.RETRIEVAL,
            status=SpanStatus.SUCCESS,
            start_time=now,
            end_time=now,
            attributes={
                "authorization": "Bearer secret-token-12345",
                "user_email": "alice@company.com",
            },
            retrieval=RetrievalStep(
                query="Find details for client at 192.168.1.50 with email client@bank.com",
                documents=[
                    RetrievedDocument(
                        doc_id="doc-1",
                        content="Account 4111 2222 3333 4444 belongs to SSN 000-11-2222.",
                        score=0.95,
                    )
                ],
            ),
        )

        span2 = Span(
            span_id="span-2",
            parent_span_id="span-1",
            name="llm_generation",
            kind=SpanKind.LLM,
            status=SpanStatus.SUCCESS,
            start_time=now,
            end_time=now,
            llm_call=LLMCall(
                prompt="Prompt containing sk-1234567890abcdef12345678 and phone 555-432-1098",
                response="Response revealing password: SecretPassword99!",
                model="gpt-4",
            ),
            tool_call=ToolCall(
                tool_name="user_lookup",
                arguments={"email": "target@domain.com", "api_key": "raw_secret_key"},
            ),
            tool_result=ToolResult(
                tool_name="user_lookup",
                output={"status": "found", "user": "target@domain.com"},
            ),
        )

        run = Run(
            run_id="run-1",
            trace_id="trace-sensitive",
            name="sensitive_execution",
            start_time=now,
            end_time=now,
            spans=[span1, span2],
            metadata={"admin_contact": "ops@cluster.local"},
        )

        sample_sensitive_trace = Trace(trace_id="trace-sensitive", runs=[run])

        anonymized = anonymize_trace(sample_sensitive_trace)

        # Verify root span sanitized
        root_span = anonymized.runs[0].spans[0]
        assert root_span.attributes["authorization"] == "[REDACTED_SECRET]"
        assert root_span.attributes["user_email"] == "[EMAIL]"
        assert root_span.retrieval is not None
        assert "192.168.1.50" not in root_span.retrieval.query
        assert "[IP_ADDRESS]" in root_span.retrieval.query
        assert "[EMAIL]" in root_span.retrieval.query
        assert "[CREDIT_CARD]" in root_span.retrieval.documents[0].content
        assert "[SSN]" in root_span.retrieval.documents[0].content

        # Verify child span sanitized
        child_span = anonymized.runs[0].spans[1]
        assert child_span.llm_call is not None
        assert "[API_KEY]" in child_span.llm_call.prompt
        assert "[PHONE_NUMBER]" in child_span.llm_call.prompt
        assert (
            child_span.llm_call.response is not None
            and "[PASSWORD]" in child_span.llm_call.response
        )
        assert child_span.tool_call is not None
        assert isinstance(child_span.tool_call.arguments, dict)
        assert child_span.tool_call.arguments["email"] == "[EMAIL]"
        assert child_span.tool_call.arguments["api_key"] == "[REDACTED_SECRET]"
        assert child_span.tool_result is not None
        assert isinstance(child_span.tool_result.output, dict)
        assert child_span.tool_result.output["user"] == "[EMAIL]"

        # Verify run metadata
        assert anonymized.runs[0].metadata is not None
        assert anonymized.runs[0].metadata["admin_contact"] == "[EMAIL]"

    def test_anonymize_diagnosis(self) -> None:
        anonymizer = TelemetryAnonymizer()

        diagnosis = Diagnosis(
            summary="Failure detected in session with user user@corp.com using key sk-abcdef12345678901234.",
            failures=[
                Failure(
                    category=FailureCategory.RETRIEVAL_FAILURE,
                    severity=Severity.HIGH,
                    description="Irrelevant context from 10.0.0.1",
                )
            ],
            hypotheses=[
                Hypothesis(
                    category=FailureCategory.RETRIEVAL_FAILURE,
                    confidence=0.92,
                    description="Retrieved email ops@corp.org rather than target context.",
                )
            ],
            evidence=[
                Evidence(
                    evidence_type=EvidenceType.RETRIEVAL_SCORE,
                    description="Evidence matched phone 555-987-6543 in snippet",
                    supporting_data={"matched_address": "192.168.0.1"},
                )
            ],
            recommendations=[
                Recommendation(
                    title="Fix query for user@corp.com",
                    description="Update filter parameter sk-1234567890abcdef12345678 to target@corp.com",
                )
            ],
            metrics=[],
        )

        anonymized_diag = anonymizer.anonymize_diagnosis(diagnosis)
        assert "[EMAIL]" in (anonymized_diag.summary or "")
        assert "[API_KEY]" in (anonymized_diag.summary or "")
        assert "user@corp.com" not in (anonymized_diag.summary or "")
        assert "sk-abcdef12345678901234" not in (anonymized_diag.summary or "")

        assert "[EMAIL]" in (anonymized_diag.hypotheses[0].description or "")
        assert "[PHONE_NUMBER]" in anonymized_diag.evidence[0].description
        assert anonymized_diag.evidence[0].supporting_data["matched_address"] == "[IP_ADDRESS]"
        assert "[EMAIL]" in anonymized_diag.recommendations[0].title
        assert "[API_KEY]" in anonymized_diag.recommendations[0].description
        assert "[EMAIL]" in anonymized_diag.recommendations[0].description
