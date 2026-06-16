"""Sanitized interaction learning log.

Captures how users phrase requests and how the assistant resolved them so we
can later mine repeated patterns into deterministic rules. This is NOT the
audit log: the audit log tracks confirmed actions for accountability; this log
tracks phrasing/intent patterns for rule refinement.

Privacy contract (enforced at write time, not trusted from callers):
  - PII patterns in the user message are replaced with [REDACTED:<kind>].
  - Filter values for sensitive columns (email, name, SSN, financial, health,
    disciplinary, notes, contact, identity_high) are replaced with [REDACTED].
  - Numeric/categorical values on non-sensitive columns are preserved because
    they are the signal we want for rule mining.
  - Only result counts (rows/columns) are recorded. No row data, ever.

Log file: logs/interaction_learning.jsonl  (append-only JSONL).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import data_dir
from core.privacy import classify_sensitivity
from nlp.privacy_filter import SENSITIVE_PATTERNS, NAME_CONTEXT_PATTERN
from nlp.synonym_mapper import normalize_text


DEFAULT_LOG_PATH = data_dir("logs") / "interaction_learning.jsonl"

# Words/phrases that strongly imply the user is correcting the assistant's
# previous interpretation. Detected case-insensitively as whole phrases.
_CORRECTION_CUES = (
    "no i mean",
    "no, i mean",
    "no i meant",
    "i meant",
    "i mean ",
    "actually",
    "not that",
    "that's wrong",
    "thats wrong",
    "let me clarify",
    "let me rephrase",
    "instead i",
    "instead, i",
    "use ",
    "should be",
    "wrong interpretation",
    "wrong, ",
)


def is_correction_message(message: str) -> bool:
    """Return True if the message looks like a correction of the prior turn."""
    if not message:
        return False
    text = f" {normalize_text(message)} "
    return any(f" {cue} " in text or text.startswith(f"{cue} ") for cue in _CORRECTION_CUES)


# Prefixes we strip when re-routing a correction so the planner doesn't read
# "I mean" as the `mean`/average verb. Order matters — longer prefixes first.
_CORRECTION_STRIP_PREFIXES = (
    "no, i mean ",
    "no i mean ",
    "no actually, ",
    "no actually ",
    "no, ",
    "actually, ",
    "actually ",
    "i meant to say ",
    "i meant ",
    "let me clarify, ",
    "let me clarify ",
    "let me rephrase, ",
    "let me rephrase ",
    "instead, i mean ",
    "instead i mean ",
    "instead, ",
)


def extract_corrected_request(message: str) -> str:
    """Strip a leading correction prefix so the planner can see the real query."""
    if not message:
        return ""
    text = message.strip()
    lowered = text.lower()
    for prefix in _CORRECTION_STRIP_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def workbook_schema_hash(sheet_columns: dict[str, list[str]] | None) -> str:
    """Stable, row-free hash of the workbook schema (sheets → sorted columns)."""
    if not sheet_columns:
        return ""
    canonical = {sheet: sorted(map(str, columns)) for sheet, columns in sheet_columns.items()}
    blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def sanitize_user_message(message: str) -> tuple[str, list[str]]:
    """Replace PII in the user message with [REDACTED:<kind>] tokens.

    Returns (sanitized_text, list_of_redaction_kinds). The latter is used to
    flag the record as unsafe for rule mining if anything was redacted.
    """
    if not message:
        return "", []
    sanitized = message
    redactions: list[str] = []
    for label, pattern in SENSITIVE_PATTERNS:
        if pattern.search(sanitized):
            redactions.append(label)
            sanitized = pattern.sub(f"[REDACTED:{label}]", sanitized)
    if NAME_CONTEXT_PATTERN.search(sanitized):
        redactions.append("person name in request")
        sanitized = NAME_CONTEXT_PATTERN.sub("[REDACTED:name]", sanitized)
    return sanitized, redactions


_TEXT_SEARCH_OPS = {"contains_text", "not_contains_text"}


def sanitize_filters(filters: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Redact filter values that target sensitive columns; keep the rest.

    Exception: contains_text / not_contains_text on a sensitive free-text
    column. There, the value is the USER'S SEARCH TERM (not row data) — we
    keep it so we can mine recurring searches (M.7). The _sanitize_value
    scrubber still scrubs obvious PII (emails / long-numeric runs) defensively.
    """
    if not filters:
        return []
    safe: list[dict[str, Any]] = []
    for condition in filters:
        column = str(condition.get("column", ""))
        operator = condition.get("operator")
        sensitive, _ = classify_sensitivity(column)
        new = {"column": condition.get("column"), "operator": operator}
        if "value" in condition:
            if sensitive and operator not in _TEXT_SEARCH_OPS:
                new["value"] = "[REDACTED]"
            else:
                new["value"] = _sanitize_value(condition.get("value"))
        safe.append(new)
    return safe


_LONG_NUMBER = re.compile(r"\b\d{6,}\b")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def _sanitize_value(value: Any) -> Any:
    """Defensive: scrub any obvious PII-looking literal even on safe columns."""
    if isinstance(value, str):
        if _EMAIL.search(value) or _LONG_NUMBER.search(value):
            return "[REDACTED]"
        if len(value) > 64:
            return value[:64] + "…"
        return value
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def sanitize_validated_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Strip the validated plan to row-free, sanitized signal."""
    if not plan:
        return {}
    sort = plan.get("sort")
    return {
        "operation": plan.get("operation"),
        "sheet": plan.get("sheet"),
        "filters": sanitize_filters(plan.get("filters")),
        "group_by": plan.get("group_by") or None,
        "value_column": plan.get("value_column") or None,
        "sort": {"column": sort.get("column"), "direction": sort.get("direction")}
                if isinstance(sort, dict) else None,
        "limit": plan.get("limit"),
    }


def build_record(
    *,
    user_message: str,
    routing: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    session_id: str | None = None,
    workbook_schema_hash_value: str = "",
    confirmation_result: str | None = None,
    corrects_entry_id: str | None = None,
    correction_message: str | None = None,
    sheet_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce one sanitized log record. Caller can serialize with json.dumps."""
    routing = routing or {}
    response = response or {}
    plan = routing.get("plan") or {}
    sanitized_message, redactions = sanitize_user_message(user_message)

    result_shape: dict[str, Any] | None = None
    if response.get("response_type") == "answer":
        rows = response.get("row_count")
        columns = response.get("columns") or []
        result_shape = {
            "rows": int(rows) if isinstance(rows, int) else None,
            "columns": len(columns) if isinstance(columns, list) else None,
        }

    sensitive_requested = False
    if isinstance(response.get("removed"), list) and response["removed"]:
        sensitive_requested = True
    # Sensitive columns the user explicitly asked for are now revealed directly
    # (no confirmation gate), so flag the audit record off reveal_sensitive
    # rather than the removed-pending "show_sensitive" type.
    if routing.get("reveal_sensitive"):
        sensitive_requested = True

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id or "",
        "workbook_schema_hash": workbook_schema_hash_value,
        "user_message": sanitized_message,
        "normalized_message": normalize_text(sanitized_message),
        "plan_source": routing.get("plan_source"),
        "intent": routing.get("intent"),
        "operation": plan.get("operation"),
        "confidence": float(routing.get("confidence") or 0.0),
        "band": routing.get("band"),
        "assumption_used": routing.get("assumption_note") or "",
        "alternatives_offered": list(routing.get("alternatives") or []),
        "suggestions_offered": list(routing.get("suggestions") or []),
        "validated_plan": sanitize_validated_plan(plan),
        "result_shape": result_shape,
        "sensitive_fields_requested": sensitive_requested,
        "confirmation_required": bool(routing.get("requires_confirmation")),
        "confirmation_result": confirmation_result,
        "validation_status": routing.get("validation", {}).get("status"),
        "fallback_reason": routing.get("fallback_reason"),
        "llm_used": bool(routing.get("llm_used")),
        "conversation_llm_used": bool(response.get("conversation_llm_used")),
        "user_corrected": bool(corrects_entry_id),
        "correction_message": correction_message,
        "corrects_entry_id": corrects_entry_id,
        "pii_redactions": redactions,
        "safe_for_rule_mining": not redactions and routing.get("validation", {}).get("status", "passed") != "failed",
        "sheet_outcome": _sanitize_sheet_outcome(sheet_outcome),
    }
    return record


def _sanitize_sheet_outcome(outcome: dict[str, Any] | None) -> dict[str, Any] | None:
    """The sheet_outcome is auto-generated from the plan and contains no PII,
    but we still normalize the shape so consumers can rely on the keys."""
    if not outcome:
        return None
    return {
        "action": str(outcome.get("action") or ""),
        "sheet_name": str(outcome.get("sheet_name") or ""),
        "reason": str(outcome.get("reason") or ""),
    }


def write_record(record: dict[str, Any], *, path: Path | None = None) -> None:
    """Append one sanitized record as a single JSON line."""
    target = path or DEFAULT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def log_interaction(
    *,
    user_message: str,
    routing: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    session_id: str | None = None,
    sheet_columns: dict[str, list[str]] | None = None,
    confirmation_result: str | None = None,
    corrects_entry_id: str | None = None,
    correction_message: str | None = None,
    sheet_outcome: dict[str, Any] | None = None,
    enabled: bool = True,
    path: Path | None = None,
) -> str | None:
    """Sanitize and write one record. Returns the new entry id, or None if disabled."""
    if not enabled:
        return None
    record = build_record(
        user_message=user_message,
        routing=routing,
        response=response,
        session_id=session_id,
        workbook_schema_hash_value=workbook_schema_hash(sheet_columns),
        confirmation_result=confirmation_result,
        corrects_entry_id=corrects_entry_id,
        correction_message=correction_message,
        sheet_outcome=sheet_outcome,
    )
    write_record(record, path=path)
    return record["id"]


def read_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read all log records. Returns an empty list if the file does not exist."""
    target = path or DEFAULT_LOG_PATH
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
