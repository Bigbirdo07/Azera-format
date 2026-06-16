"""Regression tests for natural phrasings of numeric filters in queries.

Earlier wording like 'students above a 2.0 gpa' silently parsed as a count
of all rows because the regex required a digit immediately after 'above',
blocking the article 'a/an/the' in between.
"""

from __future__ import annotations

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
