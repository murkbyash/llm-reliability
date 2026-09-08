from enum import Enum

from pydantic import BaseModel, Field


class PIIType(str, Enum):
    """Categories of sensitive personal and credential information."""

    EMAIL = "EMAIL"
    PHONE_NUMBER = "PHONE_NUMBER"
    IP_ADDRESS = "IP_ADDRESS"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    API_KEY = "API_KEY"
    BEARER_TOKEN = "BEARER_TOKEN"
    PASSWORD = "PASSWORD"
    CUSTOM = "CUSTOM"


class RedactionMaskType(str, Enum):
    """Formatting strategy applied when masking detected sensitive entities."""

    REPLACE_LABEL = "REPLACE_LABEL"  # e.g., "[EMAIL]"
    PARTIAL_MASK = "PARTIAL_MASK"  # e.g., "j***@e***.com", "sk-...abcd"
    HASH = "HASH"  # e.g., "[HASH:a1b2c3d4]"
    REDACTED = "REDACTED"  # e.g., "[REDACTED]"


class RedactionRule(BaseModel):
    """Custom pattern matching rule for identifying and masking sensitive data."""

    name: str = Field(description="Human-readable rule name")
    pii_type: PIIType = Field(
        default=PIIType.CUSTOM, description="Classification of detected entity"
    )
    pattern: str = Field(description="Regular expression pattern to match")
    mask_type: RedactionMaskType = Field(
        default=RedactionMaskType.REPLACE_LABEL, description="Masking strategy"
    )
    replacement: str | None = Field(
        default=None, description="Explicit static replacement string if specified"
    )


class RedactionConfig(BaseModel):
    """Configuration options governing telemetry anonymization and scrubbing."""

    enabled_types: list[PIIType] = Field(
        default_factory=lambda: list(PIIType),
        description="Active PII categories to detect and redact",
    )
    default_mask_type: RedactionMaskType = Field(
        default=RedactionMaskType.REPLACE_LABEL,
        description="Default masking style across all detected entities",
    )
    custom_rules: list[RedactionRule] = Field(
        default_factory=list,
        description="User-defined regex redaction rules",
    )
    hash_salt: str = Field(
        default="llm-reliability-salt",
        description="Salt used when computing deterministic cryptographic masked hashes",
    )
    mask_attribute_keys: list[str] = Field(
        default_factory=lambda: [
            "authorization",
            "api_key",
            "apikey",
            "password",
            "secret",
            "token",
            "bearer",
            "cookie",
        ],
        description="Span attribute dictionary keys whose values are unconditionally masked",
    )


class RedactedItem(BaseModel):
    """Individual sensitive entity detected and scrubbed from text."""

    pii_type: PIIType
    original_snippet: str
    redacted_snippet: str
    start_char: int
    end_char: int


class RedactionResult(BaseModel):
    """Output containing sanitized text and telemetry metadata on scrubbed entities."""

    sanitized_text: str = Field(description="Scrubbed text with all sensitive items redacted")
    total_redacted_count: int = Field(default=0, description="Total count of scrubbed items")
    counts_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of detected entity counts by PII category",
    )
    redacted_items: list[RedactedItem] = Field(
        default_factory=list,
        description="Detailed record of redacted occurrences",
    )
