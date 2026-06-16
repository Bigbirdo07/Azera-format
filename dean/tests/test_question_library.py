"""Unit tests for nlp.question_library.

Uses tiny in-memory fixtures so the assertions don't drift if the real
knowledge/question_library.json is edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp.question_library import (
    Category,
    LIBRARY_PATH,
    Question,
    askable_categories,
    clear_cache,
    load_library,
    lookup_by_id,
    lookup_for_message,
)


# ---- fixtures --------------------------------------------------------------


@pytest.fixture()
def tiny_library() -> list[Category]:
    """Two categories, four questions — small enough to reason about."""
    return [
        Category(
            id="risk", title="Risk", blurb="",
            questions=(
                Question(
                    id="risk_low_gpa",
                    text="Show me students with GPA below 2.0",
                    requires_columns=("gpa",),
                    follow_ups=("Now only those on probation", "Export this list"),
                ),
                Question(
                    id="risk_advisor_overlap",
                    text="Which advisor has the most at-risk students?",
                    requires_columns=("advisor", "academic_status"),
                    follow_ups=("Drill into that advisor",),
                ),
            ),
        ),
        Category(
            id="finance", title="Finance", blurb="",
            questions=(
                Question(
                    id="fin_balance",
                    text="Show me students with an unpaid balance",
                    requires_columns=("balance_due",),
                    follow_ups=("Group them by year",),
                ),
                Question(
                    id="dq_open",
                    text="Summarize what looks unusual in this workbook",
                    requires_columns=(),
                    follow_ups=("Show me students with missing fields",),
                ),
            ),
        ),
    ]


_FULL_SCHEMA = ["Student ID", "GPA", "Advisor", "Academic Status", "Balance Due"]


# ---- load real file --------------------------------------------------------


def test_real_library_loads_and_is_well_formed():
    """The shipped knowledge/question_library.json parses and has the basic shape."""
    clear_cache()
    cats = load_library()
    assert cats, "real library should have at least one category"
    ids: set[str] = set()
    for category in cats:
        assert category.id and category.title
        for question in category.questions:
            assert question.id and question.text
            assert question.id not in ids, f"duplicate question id: {question.id}"
            ids.add(question.id)
    # Real library file is at the documented location.
    assert LIBRARY_PATH.exists()


# ---- askable filter --------------------------------------------------------


def test_askable_keeps_questions_with_all_required_concepts(tiny_library):
    cats = askable_categories(_FULL_SCHEMA, library=tiny_library)
    flat = {q.id for cat in cats for q in cat.questions}
    assert {"risk_low_gpa", "risk_advisor_overlap", "fin_balance", "dq_open"} <= flat


def test_askable_drops_questions_when_a_required_concept_is_missing(tiny_library):
    # No "balance" column -> the finance question should be dropped, but the
    # data-quality one (no requirements) stays.
    schema = ["Student ID", "GPA", "Advisor", "Academic Status"]
    cats = askable_categories(schema, library=tiny_library)
    flat = {q.id for cat in cats for q in cat.questions}
    assert "fin_balance" not in flat
    assert "dq_open" in flat


def test_askable_drops_categories_with_no_surviving_questions(tiny_library):
    # Remove all the concepts the "risk" category needs; that whole category
    # should disappear from the askable list.
    schema = ["Balance Due"]
    cats = askable_categories(schema, library=tiny_library)
    titles = [c.title for c in cats]
    assert "Risk" not in titles
    assert "Finance" in titles


def test_askable_resolves_concept_via_synonyms(tiny_library):
    # The fixture uses "Cumulative GPA" (a synonym) instead of literal "GPA";
    # the synonyms-based concept resolver should still find it.
    schema = ["Student ID", "Cumulative GPA", "Advisor", "Standing", "Balance"]
    cats = askable_categories(schema, library=tiny_library)
    flat = {q.id for cat in cats for q in cat.questions}
    assert "risk_low_gpa" in flat
    assert "risk_advisor_overlap" in flat
    assert "fin_balance" in flat


def test_askable_accepts_literal_column_names_in_requirements():
    """A question can declare requires_columns as a literal column name
    instead of a synonyms.json concept token."""
    lib = [Category(id="x", title="X", blurb="", questions=(
        Question(id="q1", text="A question", requires_columns=("Notes",), follow_ups=()),
    ))]
    assert askable_categories(["Notes"], library=lib)
    assert not askable_categories(["Department"], library=lib)


# ---- lookup ----------------------------------------------------------------


def test_lookup_by_id_finds_the_question(tiny_library):
    found = lookup_by_id("risk_low_gpa", library=tiny_library)
    assert found is not None
    assert found.follow_ups[0] == "Now only those on probation"


def test_lookup_by_id_returns_none_for_unknown_id(tiny_library):
    assert lookup_by_id("nope", library=tiny_library) is None


def test_lookup_for_message_exact_normalized_match(tiny_library):
    found = lookup_for_message("show me students with gpa below 2.0", library=tiny_library)
    assert found is not None
    assert found.id == "risk_low_gpa"


def test_lookup_for_message_token_subset_match(tiny_library):
    # User typed all the template tokens (in a different order with extra words).
    # The subset-based fallback should still resolve it.
    found = lookup_for_message(
        "please show me the students with gpa below 2.0 right now",
        library=tiny_library,
    )
    assert found is not None
    assert found.id == "risk_low_gpa"


def test_lookup_for_message_ignores_unrelated_long_messages(tiny_library):
    found = lookup_for_message(
        "tell me a long unrelated story about the weather and a flock of birds",
        library=tiny_library,
    )
    assert found is None


def test_lookup_for_message_empty_returns_none(tiny_library):
    assert lookup_for_message("", library=tiny_library) is None
    assert lookup_for_message("   ", library=tiny_library) is None
