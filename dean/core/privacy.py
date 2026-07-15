"""Privacy layer for student records.

Classifies columns by sensitivity, decides which columns are safe to show by
default, and flags requests that expose sensitive student-level data so the UI
can ask for confirmation before showing or exporting it. Everything here is
local metadata logic — it never sends data anywhere.
"""

from __future__ import annotations

from typing import Any

from nlp.synonym_mapper import normalize_text


# sensitivity_type -> keywords that appear in a column name.
_SENSITIVITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "contact": ("email", "e mail", "phone", "mobile", "cell", "address", "zip", "postal",
                "contact", "guardian", "emergency", "parent"),
    "financial": ("financial aid", "fafsa", "aid", "scholarship", "balance", "tuition", "payment", "loan", "grant"),
    "disciplinary": ("disciplinary", "discipline", "conduct", "infraction", "suspension", "expulsion"),
    "health": ("medical", "disability", "accommodation", "health", "mental"),
    "notes": ("notes", "note", "comment", "remark", "memo"),
    "identity_high": ("ssn", "social security", "date of birth", "dob", "birth"),
    "identity": ("student id", "id number", "first name", "last name", "full name", "name"),
}

# Types hidden from student-level tables unless explicitly requested + confirmed.
HIDDEN_TYPES = {"contact", "financial", "disciplinary", "health", "notes", "identity_high"}

# Default columns to show in a student-level list (when present).
_DEFAULT_VISIBLE_CONCEPTS = (
    "student id", "name", "department", "major", "year", "gpa", "academic status", "advisor",
)


def classify_sensitivity(column_name: str) -> tuple[bool, str]:
    """Return (is_sensitive, sensitivity_type) for a column name."""
    normalized = normalize_text(column_name)
    # Check the more specific identity_high before generic identity.
    for sensitivity_type in ("identity_high", "contact", "financial", "disciplinary", "health", "notes", "identity"):
        for keyword in _SENSITIVITY_KEYWORDS[sensitivity_type]:
            if keyword in normalized:
                if sensitivity_type == "disciplinary" and keyword == "discipline":
                    # Bare "Discipline" alone is the dean-office word for academic
                    # department (see nlp/synonym_mapper concept aliases); only
                    # treat it as sensitive when the header also carries a
                    # behavioral-record indicator. Skyward's own field is literally
                    # named "Discipline Information".
                    if not any(ind in normalized for ind in (
                        "status", "record", "conduct", "infraction", "action",
                        "warning", "probation", "information", "incident",
                        "referral", "offense",
                    )):
                        continue
                return True, sensitivity_type
    return False, "unknown"


def detect_sensitive_columns(columns: list[str]) -> dict[str, str]:
    """Map each sensitive column to its sensitivity_type."""
    result: dict[str, str] = {}
    for column in columns:
        sensitive, sensitivity_type = classify_sensitivity(column)
        if sensitive:
            result[column] = sensitivity_type
    return result


def is_hidden_by_default(column: str) -> bool:
    sensitive, sensitivity_type = classify_sensitivity(column)
    return sensitive and sensitivity_type in HIDDEN_TYPES


def get_default_visible_columns(columns: list[str]) -> list[str]:
    """Columns safe to show in a student-level table by default: the standard
    roster fields, minus anything hidden-by-default."""
    visible = [column for column in columns if not is_hidden_by_default(column)]
    preferred = []
    for concept in _DEFAULT_VISIBLE_CONCEPTS:
        for column in visible:
            if normalize_text(column) == concept and column not in preferred:
                preferred.append(column)
    extras = [c for c in visible if c not in preferred]
    return preferred + extras


def requested_sensitive_columns(user_request: str, columns: list[str]) -> list[str]:
    """Hidden-by-default columns the user explicitly named in their message.

    Matches a column only against its own name/tokens (and sensitivity keywords
    contained in that name), so "emails" flags Email but not Phone.
    """
    text = normalize_text(user_request)
    named = []
    for column in columns:
        if not is_hidden_by_default(column):
            continue
        col_norm = normalize_text(column)
        _, sensitivity_type = classify_sensitivity(column)
        # Match the full column name or a *specific* sensitivity keyword that is
        # part of this column's name. Generic shared tokens like "status" are
        # NOT used, so "housing status" doesn't flag Conduct/Financial Status.
        own_keywords = [kw for kw in _SENSITIVITY_KEYWORDS.get(sensitivity_type, ()) if kw in col_norm]
        if f" {col_norm} " in f" {text} " or any(keyword in text for keyword in own_keywords):
            named.append(column)
    return named


def redact_table(table: list[dict[str, Any]], columns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop hidden-by-default columns from a result preview.

    Returns (redacted_rows, removed_columns).
    """
    if not table:
        return table, []
    hidden = [c for c in columns if is_hidden_by_default(c)]
    if not hidden:
        return table, []
    hidden_set = set(hidden)
    redacted = [{k: v for k, v in row.items() if k not in hidden_set} for row in table]
    removed = [c for c in hidden if any(c in row for row in table)]
    return redacted, removed


def confirmation_reason(requested: list[str]) -> str:
    return (
        "This includes sensitive student-level information ("
        + ", ".join(requested)
        + "). Please confirm that you want to show these fields."
    )
