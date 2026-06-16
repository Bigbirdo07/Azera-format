"""Confidence-band handling and alternative-interpretation prompts.

The planner produces a confidence score on every plan. We turn that score into
one of three behaviors:

  HIGH    (>= 0.85)  — execute cleanly, no assumption note.
  MEDIUM  (>= 0.55)  — execute the most-likely safe read-only interpretation,
                       surface the interpretation as an assumption note, and
                       offer 1–3 alternative phrasings the user can click.
  LOW     (< 0.55)   — ask a clarification question instead.

Sensitive-field requests, edits, exports, note edits, and field updates all
bypass the MEDIUM band — they go straight to clarify or confirm. We never
"assume" through a confirmation gate.

Alternatives for known vague terms ("struggling", "needs help", "at risk")
are a small deterministic dictionary; the LLM may add more when the
conversational layer is on, but only the deterministic ones are shown unless
the LLM's suggestions can be safely re-validated.
"""

from __future__ import annotations

from typing import Any

from nlp.synonym_mapper import normalize_text


HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.55


# Vague phrases that benefit from explicit assumption + alternatives.
# Keys are normalized (lowercased) phrases; values are alternative interpretations
# the user can click to re-run with a clearer definition.
VAGUE_TERM_ALTERNATIVES: dict[str, list[str]] = {
    "struggling": [
        "Now use GPA below 2.0 instead",
        "Use academic probation only",
        "Use warning or probation status",
    ],
    "struggle": [
        "Now use GPA below 2.0 instead",
        "Use academic probation only",
    ],
    "needs help": [
        "Now use GPA below 2.0 instead",
        "Use academic probation only",
    ],
    "needs attention": [
        "Use GPA below 2.5",
        "Use probation status only",
    ],
    "at risk": [
        "Use GPA below 2.5",
        "Use warning or probation status",
    ],
    "at-risk": [
        "Use GPA below 2.5",
        "Use warning or probation status",
    ],
    "doing well": [
        "Use GPA above 3.5",
        "Use Dean's list only",
    ],
    "underperforming": [
        "Use GPA below 2.0",
        "Use probation status only",
    ],
    "falling behind": [
        "Use GPA below 2.0",
        "Use warning or probation status",
    ],
    "overloaded": [
        "Group by Advisor and count",
        "Show advisors with the most students",
    ],
    "best students": [
        "Use GPA above 3.7",
        "Show top 10 by GPA",
    ],
    "top students": [
        "Use GPA above 3.7",
        "Show top 10 by GPA",
    ],
}


def classify_confidence(confidence: float) -> str:
    """Return 'high', 'medium', or 'low'."""
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def detect_vague_alternatives(message: str) -> list[str]:
    """Return alternative-interpretation prompts for any vague terms found.

    Matches whole-word phrases case-insensitively. Returns deduplicated, in
    insertion order, up to 3 alternatives.
    """
    text = f" {normalize_text(message)} "
    matched: list[str] = []
    seen: set[str] = set()
    for phrase, alternatives in VAGUE_TERM_ALTERNATIVES.items():
        if f" {phrase} " not in text:
            continue
        for alternative in alternatives:
            key = alternative.lower()
            if key in seen:
                continue
            seen.add(key)
            matched.append(alternative)
            if len(matched) >= 3:
                return matched
    return matched


def should_assume(
    *,
    confidence: float,
    intent: str,
    requires_confirmation: bool,
    pending_type: str | None,
    has_sensitive: bool,
) -> bool:
    """Decide whether to execute now and surface an assumption note.

    We assume only for medium-confidence, read-only query intents that do not
    touch sensitive fields and do not need a confirmation gate.
    """
    if intent != "query":
        return False
    if requires_confirmation:
        return False
    if pending_type in {"export", "note_edit", "field_update"}:
        return False
    if has_sensitive:
        return False
    return MEDIUM_CONFIDENCE <= confidence < HIGH_CONFIDENCE


def build_assumption_note(narration: str) -> str:
    """Wrap the deterministic narration in an explicit 'I interpreted this as' lead."""
    if not narration:
        return ""
    text = narration.strip()
    if text.lower().startswith("i interpreted") or text.lower().startswith("i understood"):
        return text
    return f"I interpreted this as: {text[0].lower() + text[1:]}" if text else ""
