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
