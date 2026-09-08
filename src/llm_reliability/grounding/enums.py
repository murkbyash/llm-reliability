"""Grounding and answer support status enumerations."""

from enum import Enum


class SupportStatus(str, Enum):
    """Factual support status of an answer relative to provided context."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
