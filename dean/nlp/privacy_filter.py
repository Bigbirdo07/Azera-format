from __future__ import annotations

import re
from dataclasses import dataclass


SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("SSN-like value", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone-like value", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("long ID-like number", re.compile(r"\b\d{6,}\b")),
    ("currency amount", re.compile(r"(?:\$|\bUSD\b)\s*\d+(?:,\d{3})*(?:\.\d{2})?", re.IGNORECASE)),
    ("date of birth wording", re.compile(r"\b(?:dob|date of birth|birth date)\b", re.IGNORECASE)),
)

NAME_CONTEXT_PATTERN = re.compile(
    r"\b(?:named|name is|student is|student named|person named)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
)


@dataclass(frozen=True)
class PrivacyCheck:
    allowed: bool
    reasons: list[str]


def check_local_model_request_privacy(user_request: str) -> PrivacyCheck:
    reasons = [
        label
        for label, pattern in SENSITIVE_PATTERNS
        if pattern.search(user_request)
    ]
    return PrivacyCheck(allowed=not reasons, reasons=reasons)
