"""Regression tests for natural phrasings of numeric filters in queries.

Earlier wording like 'students above a 2.0 gpa' silently parsed as a count
of all rows because the regex required a digit immediately after 'above',
blocking the article 'a/an/the' in between.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nlp.query_planner import _detect_filters, _numeric_filter
from nlp.synonym_mapper import load_json


@pytest.fixture(scope="module")
def synonyms() -> dict:
    return load_json("synonyms.json")


@pytest.fixture(scope="module")
def columns() -> list[str]:
    # Mirrors the user's real workbook (Discipline, not Department).
    return [
        "Student ID", "Name", "Year", "Discipline", "Standing", "Location",
        "Advisor", "Major", "Second Major", "GPA",
    ]


@pytest.mark.parametrize(
    "query,expected",
    [
        # Articles between operator and number
        ("how many students above a 2 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 2}),
        ("how many students with above a 2.00 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 2.0}),
        ("students above a 2.0 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 2.0}),
        ("students with the gpa above 2.0",
         {"column": "GPA", "operator": "greater_than", "value": 2.0}),
        ("students below a 2.5 gpa",
         {"column": "GPA", "operator": "less_than", "value": 2.5}),
        ("how many students above 2.0 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 2.0}),
        ("students with gpa above 2",
         {"column": "GPA", "operator": "greater_than", "value": 2}),
        # Synonyms for greater/less
        ("students higher than 2 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 2}),
        ("students lower than 2 gpa",
         {"column": "GPA", "operator": "less_than", "value": 2}),
        ("students larger than 2.5 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 2.5}),
        # At least / at most
        ("students at least 2 gpa",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2}),
        ("students at most 2.5 gpa",
         {"column": "GPA", "operator": "less_or_equal", "value": 2.5}),
        # Suffix forms
        ("students with a 2 gpa or higher",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2}),
        ("students with a 2 gpa or above",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2}),
        ("students with a 2.5 gpa or below",
         {"column": "GPA", "operator": "less_or_equal", "value": 2.5}),
        ("students with a 2 gpa and up",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2}),
        ("gpa of 2 or more",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2}),
        ("gpa of 3 or less",
         {"column": "GPA", "operator": "less_or_equal", "value": 3}),
        # Plus suffix
        ("students with 2.5+ gpa",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2.5}),
        ("students with a 3.0+ gpa",
         {"column": "GPA", "operator": "greater_or_equal", "value": 3}),
        # Symbolic comparisons
        ("students with gpa >= 2",
         {"column": "GPA", "operator": "greater_or_equal", "value": 2}),
        ("students with gpa > 2.0",
         {"column": "GPA", "operator": "greater_than", "value": 2.0}),
        ("students with gpa < 3",
         {"column": "GPA", "operator": "less_than", "value": 3}),
        ("students with gpa <= 3.5",
         {"column": "GPA", "operator": "less_or_equal", "value": 3.5}),
        # Various number magnitudes / decimals
        ("students above 0.5 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 0.5}),
        ("students above 1 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 1}),
        ("students above 1.5 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 1.5}),
        ("students above 3.7 gpa",
         {"column": "GPA", "operator": "greater_than", "value": 3.7}),
    ],
)
def test_natural_phrasings_resolve_to_numeric_filter(query, expected, columns, synonyms):
    got = _numeric_filter(query, columns, synonyms)
    assert got == expected, f"expected {expected}, got {got}"


def test_natural_phrasings_round_trip_through_detect_filters(columns, synonyms):
    filters = _detect_filters("how many students above a 2 gpa", columns, synonyms)
    assert filters == [{"column": "GPA", "operator": "greater_than", "value": 2}]


@pytest.mark.parametrize(
    "query,expected",
    [
        # _detect_filters splits on bare "and"/"or" to parse multi-clause
        # asks ("gpa below 2.0 and attendance below 90"). That split must NOT
        # fire on the "or"/"and" that's part of a comparison phrase itself
        # ("or higher", "and up", ...), or the threshold gets torn away from
        # its number and silently dropped instead of parsed.
        ("how many students have a gpa of 3.5 or higher",
         [{"column": "GPA", "operator": "greater_or_equal", "value": 3.5}]),
        ("students with a 2 gpa or above",
         [{"column": "GPA", "operator": "greater_or_equal", "value": 2}]),
        ("students with a 2.5 gpa or below",
         [{"column": "GPA", "operator": "less_or_equal", "value": 2.5}]),
        ("students with a 2 gpa and up",
         [{"column": "GPA", "operator": "greater_or_equal", "value": 2}]),
        ("gpa of 2 or more",
         [{"column": "GPA", "operator": "greater_or_equal", "value": 2}]),
    ],
)
def test_or_and_comparison_phrases_survive_clause_splitting(query, expected, columns, synonyms):
    filters = _detect_filters(query, columns, synonyms, original_text=query)
    assert filters == expected, f"expected {expected}, got {filters}"


def test_genuine_and_clause_still_splits_into_two_filters(columns, synonyms):
    query = "students with gpa below 2.0 and attendance below 90"
    columns_with_attendance = columns + ["Attendance Rate"]
    filters = _detect_filters(query, columns_with_attendance, synonyms, original_text=query)
    assert filters == [
        {"column": "GPA", "operator": "less_than", "value": 2.0},
        {"column": "Attendance Rate", "operator": "less_than", "value": 0.9},
    ]


def test_missed_days_resolves_to_days_absent_not_calendar_days(synonyms):
    from nlp.query_planner import plan_query

    cols = ["Attendance Calendar Days", "Days Present", "Days Absent", "Attendance Rate"]
    frame = pd.DataFrame({c: [0] for c in cols})
    result = plan_query(
        user_request="How many students have missed more than 15 days?",
        selected_sheet="Students", sheet_columns={"Students": cols}, frame=frame,
    )
    assert result.query["filters"] == [
        {"column": "Days Absent", "operator": "greater_than", "value": 15}
    ]


def test_spelled_out_small_number_produces_a_filter(synonyms):
    from nlp.query_planner import plan_query

    cols = ["Risk Signals", "Risk Level"]
    frame = pd.DataFrame({"Risk Signals": [1, 2], "Risk Level": ["Low", "High"]})
    result = plan_query(
        user_request="How many students have more than one risk signal?",
        selected_sheet="Students", sheet_columns={"Students": cols}, frame=frame,
    )
    assert result.query["filters"] == [
        {"column": "Risk Signals", "operator": "greater_than", "value": 1}
    ]


def test_assessment_column_name_does_not_collide_with_a_matching_major_value(synonyms):
    from nlp.query_planner import plan_query

    cols = ["SAT English", "Major"]
    frame = pd.DataFrame({"SAT English": [700, 600], "Major": ["English", "History"]})
    result = plan_query(
        user_request="How many students scored above 650 on SAT English?",
        selected_sheet="Students", sheet_columns={"Students": cols}, frame=frame,
    )
    assert result.query["filters"] == [
        {"column": "SAT English", "operator": "greater_than", "value": 650}
    ]


def test_gpa_question_does_not_collide_with_derived_risk_reason_text(synonyms):
    from nlp.query_planner import plan_query

    cols = ["GPA", "Risk Reason"]
    frame = pd.DataFrame({
        "GPA": [1.5, 3.5],
        "Risk Reason": ["GPA below 2.0", ""],
    })
    result = plan_query(
        user_request="How many students have a GPA below 2.0?",
        selected_sheet="Students", sheet_columns={"Students": cols}, frame=frame,
    )
    assert result.query["filters"] == [
        {"column": "GPA", "operator": "less_than", "value": 2.0}
    ]


def test_before_after_resolve_as_less_greater_than(columns, synonyms):
    filters = _detect_filters("how many students above a 2 gpa and before 2028", columns, synonyms)
    # Sanity: "before"/"after" register as real comparison words at all,
    # exercised directly on a concept that had zero comparison-word coverage
    # before this fix (grad_year, added to _NUMERIC_CONCEPTS this session).
    from nlp.query_planner import _numeric_filter
    cols = ["Grad Year"]
    assert _numeric_filter("grad year before 2028", cols, synonyms) == {
        "column": "Grad Year", "operator": "less_than", "value": 2028,
    }
    assert _numeric_filter("grad year after 2027", cols, synonyms) == {
        "column": "Grad Year", "operator": "greater_than", "value": 2027,
    }


def test_have_withdrawn_resolves_to_withdrawal_date_presence(synonyms):
    # "withdrawn" doesn't literally contain "withdrawal date" (the column
    # name), so this needs concept-level resolution, not the generic
    # have/has literal-substring fallback.
    cols = ["Student ID", "Name", "Withdrawal Date", "Discipline Information"]
    filters = _detect_filters("how many students have withdrawn", cols, synonyms)
    assert filters == [{"column": "Withdrawal Date", "operator": "is_not_missing"}]


def test_discipline_record_on_file_resolves_to_conduct_column_presence(synonyms):
    # "discipline record" doesn't literally contain "discipline information"
    # (the real Skyward column name) -- same gap as withdrawal above.
    cols = ["Student ID", "Name", "Withdrawal Date", "Discipline Information"]
    filters = _detect_filters(
        "how many students have a discipline record on file", cols, synonyms,
    )
    assert filters == [{"column": "Discipline Information", "operator": "is_not_missing"}]


def test_bare_of_n_equality_resolves_for_a_recognized_numeric_concept(synonyms):
    # Regression: "unexcused absence count of 0" had no comparison word at
    # all (no "above"/"or more"/">"), so it silently matched zero filters
    # and fell through to an unfiltered row count -- a confidently wrong
    # answer. Concept-resolution only, so it can't misfire on unrelated
    # "... out of 250" style phrasing (no numeric concept nearby there).
    from nlp.query_planner import _numeric_filter

    cols = ["Grade", "Unexcused Absences", "Excused Absences"]
    assert _numeric_filter(
        "how many students have an unexcused absence count of 0", cols, synonyms,
    ) == {"column": "Unexcused Absences", "operator": "equals", "value": 0}


def test_bare_of_n_equality_does_not_misfire_on_unrelated_out_of_phrasing(synonyms):
    from nlp.query_planner import _numeric_filter

    cols = ["Grade", "Unexcused Absences", "GPA", "Advisor"]
    assert _numeric_filter("top 5 students out of 250", cols, synonyms) is None
    assert _numeric_filter("average gpa of the class", cols, synonyms) is None


def test_bare_year_equality_resolves_for_a_recognized_date_concept(synonyms):
    # Regression: "entered in 2024" silently returned the unfiltered count
    # (same failure mode as the "of N" gap above, for date columns).
    from nlp.query_planner import _numeric_filter

    cols = ["Entry Date", "Withdrawal Date", "Birth Date"]
    assert _numeric_filter("how many students entered in 2024", cols, synonyms) == {
        "column": "Entry Date", "operator": "equals", "value": 2024,
    }


def test_grade_name_prefix_does_not_steal_the_adjacent_numeric_column(synonyms):
    # Regression: "9th graders have more than 3 unexcused absences" resolved
    # to {"column": "Grade", "operator": "greater_than", "value": 3} -- the
    # leading "9th graders" fragment literal-substring-matched the Grade
    # column (via "grade" inside "graders") before the adjacent, far more
    # specific "unexcused absences" concept match on the trailing side ever
    # got a chance to run. Concept-level matches on either side must now
    # outrank a generic literal-substring match on either side.
    from nlp.query_planner import _numeric_filter

    cols = ["Grade", "Unexcused Absences", "Excused Absences"]
    assert _numeric_filter(
        "how many 9th graders have more than 3 unexcused absences", cols, synonyms,
    ) == {"column": "Unexcused Absences", "operator": "greater_than", "value": 3}


def test_unexcused_absences_does_not_collide_with_excused_absences(synonyms):
    from nlp.query_planner import plan_query

    cols = ["Excused Absences", "Unexcused Absences"]
    frame = pd.DataFrame({"Excused Absences": [1, 5], "Unexcused Absences": [2, 4]})
    for label, expected_col in [("unexcused", "Unexcused Absences"), ("excused", "Excused Absences")]:
        result = plan_query(
            user_request=f"How many students have more than 3 {label} absences?",
            selected_sheet="Students", sheet_columns={"Students": cols}, frame=frame,
        )
        assert result.query["filters"] == [
            {"column": expected_col, "operator": "greater_than", "value": 3}
        ], label
