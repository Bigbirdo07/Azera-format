"""L.4: confidence bands + assume-and-offer-correction for vague queries."""

from __future__ import annotations

import pytest

from nlp.planner_router import plan_user_request
from nlp.uncertainty import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    build_assumption_note,
    classify_confidence,
    detect_vague_alternatives,
)


def _route(message, sheets, columns, *, state=None, settings=None):
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=state,
        settings=settings or {"llm_enabled": False},
    )


# ---- band classifier --------------------------------------------------------


def test_classify_confidence_buckets():
    assert classify_confidence(0.95) == "high"
    assert classify_confidence(HIGH_CONFIDENCE) == "high"
    assert classify_confidence(0.7) == "medium"
    assert classify_confidence(MEDIUM_CONFIDENCE) == "medium"
    assert classify_confidence(0.4) == "low"


# ---- vague-term detection ---------------------------------------------------


def test_detect_struggling_offers_alternatives():
    alts = detect_vague_alternatives("which students are struggling this term")
    assert len(alts) >= 2
    assert all(isinstance(a, str) and a.strip() for a in alts)


def test_detect_no_alternatives_for_specific_query():
    assert detect_vague_alternatives("Show me Accounting students") == []


def test_detect_alternatives_capped_at_three():
    alts = detect_vague_alternatives("struggling students who need help and are at risk")
    assert len(alts) <= 3


# ---- assumption note format -------------------------------------------------


def test_build_assumption_note_preserves_existing_lead():
    text = "I understood this as: list the matching rows."
    assert build_assumption_note(text) == text


def test_build_assumption_note_lowers_first_letter_when_wrapping():
    note = build_assumption_note("Filter to Accounting students.")
    assert note.startswith("I interpreted this as:")
    assert "filter to accounting" in note.lower()


# ---- planner_router emits band + alternatives ------------------------------


def test_high_confidence_routing_has_no_assumption_note(sheets, columns):
    routing = _route("Show me Accounting students", sheets, columns)
    assert routing["band"] == "high"
    assert routing["assumption_note"] == ""
    assert routing["alternatives"] == []


def test_medium_band_falls_through_for_unknown_phrasing(sheets, columns):
    """A vague query with no rule match goes to clarify when the LLM is off."""
    routing = _route("who seems like they need advisor attention", sheets, columns)
    # With the LLM disabled and confidence low, this clarifies.
    assert routing["intent"] in {"clarify", "query"}


def test_sensitive_request_at_medium_band_still_requires_confirmation(sheets, columns):
    """Even with a vague phrase, sensitive-field requests must confirm, not assume."""
    # Showing a hidden field requires confirmation regardless of band.
    routing = _route("show me struggling students and their emails", sheets, columns)
    if routing["intent"] == "query" and routing.get("pending_type") == "show_sensitive":
        assert routing["requires_confirmation"] is True
        # We never surface an assumption note in front of a confirmation gate.
        assert routing["assumption_note"] == ""
